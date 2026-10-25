"""Recovery pipeline executor (controller side): V -> Q_eng -> M2 -> Q_eval -> M3 under policy ba90fd66.

Invariants (Codex R87 review):
* one controller: an exclusive flock on <root>/ledger/controller.lock for the whole lifetime;
* halted / deadline / unknown or switched backend are checked BEFORE any GPU query;
* admission = two separately checked nvidia-smi queries (rc, timeout, strict parse each; 20 s total); any failure
  is unknown = no admission; after V the GPU UUID is bound and never switched;
* resource waiting uses the FIXED 600 s grid from the policy anchor; slots used and the last slot index persist in
  the ledger (432 max, never reset); the launch check right after a finished invocation is not a waiting slot;
* each invocation: gate -> durable reservation -> fresh job dir + spec -> detached host envelope (code check,
  token-owned job, timeout, ownership-verified cleanup, status file) -> poll until status observed or the reserved
  hard limit -> settle the actual wall; unverified status or cleanup halts the pipeline;
* transfers into fresh local directories with checked rsync rc and host-side SHA verification; remote absolute paths
  in receipts are mapped to the local copies and SHA-checked by the validators; receipts are never rewritten;
* no retries, no seed replacement, no second GPU. The local 3090 offload candidate is withdrawn (not attempted).
Usage: executor.py --root DIR --backends backends.json [--bundle B --bundle-sha S] [--until PHASE]
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import math
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_codex_v1 as gate  # noqa: E402
import validators as V  # noqa: E402
from ledger import Ledger  # noqa: E402

POLICY = HERE / "policy_codex_R86.json"
POLICY_SHA = "ba90fd6623cfac7af0db3b9d9c7eb3cedcb10c68e2cd122c02a088fce3d6a842"
ORDER = ["V", "Q_eng", "M2", "Q_eval", "M3"]
CLEANUP_S, POLL_S, BOOKKEEPING_S = 60, 15, 45


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc)


def host_argv(host: dict, cmd: str, timeout: float) -> list[str]:
    if host.get("ssh") is None:
        return ["bash", "-c", cmd]
    return ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={min(15, max(2, int(timeout) - 1))}", host["ssh"], cmd]


def sh(host: dict, cmd: str, timeout: float, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(host_argv(host, cmd, timeout), input=stdin, capture_output=True, text=True, timeout=timeout)


class Executor:
    def __init__(self, root: Path, backends: dict, bundle: dict | None, log):
        raw = POLICY.read_bytes()
        if hashlib.sha256(raw).hexdigest() != POLICY_SHA:
            raise SystemExit("policy hash mismatch")
        self.policy = json.loads(raw)
        self.anchor = datetime.datetime.fromisoformat(self.policy["release_anchor_utc"])
        self.deadline = datetime.datetime.fromisoformat(self.policy["deadline_utc"])
        self.root, self.backends, self.bundle, self.log = root, backends, bundle, log
        (root / "ledger").mkdir(parents=True, exist_ok=True)
        self._lock = open(root / "ledger" / "controller.lock", "a")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("another controller holds the ledger lock")
        self.ledger = Ledger(root / "ledger")

    # ---------------------------------------------------------------- admission and waiting
    def admission(self, host: dict) -> dict:
        snap = {"time": utcnow().isoformat(), "host": host["name"]}
        t_end = time.monotonic() + 20
        outs = {}
        for key, q in (("gpus", "nvidia-smi --query-gpu=index,uuid,memory.free --format=csv,noheader,nounits"),
                       ("apps", "nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits")):
            left = t_end - time.monotonic()
            if left <= 1:
                return {**snap, "ok": False, "reason": "query budget exhausted"}
            try:
                r = sh(host, q, left)
            except subprocess.TimeoutExpired:
                return {**snap, "ok": False, "reason": f"{key} timeout"}
            if r.returncode != 0:
                return {**snap, "ok": False, "reason": f"{key} rc={r.returncode}", "stderr": r.stderr[-200:]}
            outs[key] = r.stdout
        try:
            gpus = []
            for l in outs["gpus"].strip().splitlines():
                idx, uid, free = [x.strip() for x in l.split(",")]
                gpus.append({"index": int(idx), "uuid": uid, "free": int(free)})
            busy = {l.split(",")[0].strip() for l in outs["apps"].strip().splitlines() if l.strip()}
            if not gpus:
                raise ValueError("no GPU rows")
        except (ValueError, IndexError) as e:
            return {**snap, "ok": False, "reason": f"parse error: {e}"}
        bound = self.ledger.state().get("bound_gpu_uuid")
        cands = [dict(g, busy=g["uuid"] in busy) for g in gpus if g["index"] in host["eligible_gpus"]
                 and (bound is None or g["uuid"] == bound)]
        clear = [c for c in cands if not c["busy"] and c["free"] >= host["min_free_mib"]]
        return {**snap, "ok": True, "gpus": cands, "chosen": max(clear, key=lambda c: c["free"]) if clear else None}

    def preflight_refusal(self, host: dict) -> str | None:
        st = self.ledger.state()
        if st["halted"]:
            return f"halted: {st['halt_reason']}"
        if utcnow() >= self.deadline:
            return "absolute deadline expired"
        if host.get("policy_backend") not in self.policy["admission_free_MiB"]:
            return "unknown backend"
        if st.get("backend_name") and host["name"] != self.backends[st["backend_name"]]["name"]:
            return "backend switch refused"
        return None

    def wait_for_gpu(self, host: dict) -> dict | None:
        adm = self.admission(host)  # launch check (not a waiting slot)
        self.log({"event": "launch_check", **adm})
        while not (adm["ok"] and adm["chosen"] is not None):
            w = self.ledger.state().get("wait") or {"slots_used": 0, "last_slot_index": -1}
            idx = max(w["last_slot_index"] + 1, math.floor((utcnow() - self.anchor).total_seconds() / 600) + 1)
            t_slot = self.anchor + datetime.timedelta(seconds=600 * idx)
            if w["slots_used"] >= self.policy["wait"]["maximum_slots"] or t_slot >= self.deadline:
                self.log({"event": "wait_exhausted", **w})
                return None
            time.sleep(max(0.0, (t_slot - utcnow()).total_seconds()))
            if (r := self.preflight_refusal(host)):
                self.log({"event": "wait_aborted", "reason": r})
                return None
            self.ledger.update(wait={"slots_used": w["slots_used"] + 1, "last_slot_index": idx})
            adm = self.admission(host)
            self.log({"event": "slot_check", "slot": idx, **adm})
        return adm

    # ---------------------------------------------------------------- one invocation
    def invoke(self, phase: str, host: dict, job_argv: list[str], requested: int, inv: dict, extra_state=None) -> dict:
        if (r := self.preflight_refusal(host)):
            return {"status": "refused", "reason": r}
        adm = self.wait_for_gpu(host)
        if adm is None:
            return {"status": "not_run_resource_deadline"}
        st = self.ledger.state()
        state = {**st, "backend": host["policy_backend"], "admission_query_success": True,
                 "foreign_compute_processes": 0, "free_memory_MiB": adm["chosen"]["free"],
                 "requested_wall_seconds_including_cleanup": requested, **(extra_state or {})}
        verdict = gate.verdict(self.policy, state, phase, utcnow(), self.bundle, POLICY_SHA)
        self.log({"event": "gate", "phase": phase, **inv, "verdict": verdict})
        if not verdict["allowed"]:
            return {"status": "gate_refused", "verdict": verdict}
        hard, grace = verdict["max_wall_seconds_including_cleanup"], self.policy["kill_grace_seconds"]
        work = hard - grace - CLEANUP_S - BOOKKEEPING_S
        if work < 60:
            return {"status": "gate_refused", "verdict": {**verdict, "reasons": ["work budget after cleanup < 60 s"]}}
        run_id = f"{phase}-{st['attempted_invocations'][phase] + 1:03d}-{uuid.uuid4().hex[:10]}"
        jobdir = f"{host['probe_root']}/recovery_jobs/{run_id}"
        spec = {"token": uuid.uuid4().hex, "argv": [a.replace("{RUN_ID}", run_id) for a in job_argv],
                "env": {**host["env"], "CUDA_VISIBLE_DEVICES": adm["chosen"]["index"], "RECOVERY_RUN_ID": run_id},
                "work_seconds": work, "kill_grace_seconds": grace, "cleanup_seconds": CLEANUP_S, "repo": host["repo"],
                "expect": host["expect"], "log_path": f"{jobdir}/job.log", "status_path": f"{jobdir}/status.json"}
        (self.root / "specs").mkdir(exist_ok=True)
        (self.root / "specs" / f"{run_id}.json").write_text(json.dumps(spec))
        self.ledger.reserve(phase, hard, {**inv, "run_id": run_id, "host": host["name"],
                                          "gpu_uuid": adm["chosen"]["uuid"], "gpu_index": adm["chosen"]["index"]})
        t0 = time.monotonic()
        status, launch_err = None, None
        q = shlex.quote
        try:
            r = sh(host, f"test ! -e {q(jobdir)} && mkdir -p {q(jobdir)} && cat > {q(jobdir)}/spec.json && "
                         f"cd {q(host['repo'])} && (setsid nohup {q(host['env']['OPENWAM_PYTHON'])} "
                         f"scripts/probe/recovery/envelope.py {q(jobdir)}/spec.json > {q(jobdir)}/envelope.log "
                         f"2>&1 < /dev/null &) && echo LAUNCHED", 30, stdin=json.dumps(spec))
            if "LAUNCHED" not in r.stdout:
                launch_err = f"launch failed rc={r.returncode}: {r.stderr[-300:]}"
            while launch_err is None and time.monotonic() - t0 < hard - 5:
                time.sleep(POLL_S)
                try:
                    c = sh(host, f"cat {q(spec['status_path'])} 2>/dev/null", 20)
                    if c.returncode == 0 and c.stdout.strip():
                        status = json.loads(c.stdout)
                        break
                except (subprocess.TimeoutExpired, json.JSONDecodeError):
                    pass
        except Exception as e:  # noqa: BLE001 - the reservation must still be settled
            launch_err = f"{type(e).__name__}: {e}"
        wall = time.monotonic() - t0
        ok_status = (status is not None and status.get("token") == spec["token"]
                     and (status.get("cleanup") or {}).get("verified_gone") is True)
        self.ledger.settle(wall, {"run_id": run_id, "rc": None if status is None else status.get("rc"),
                                  "status_observed": status is not None, "cleanup_verified": ok_status,
                                  "launch_error": launch_err})
        if launch_err or not ok_status:
            self.ledger.halt(f"{phase} {run_id}: status/cleanup not verified ({launch_err or 'no verified status'})")
            return {"status": "halted", "run_id": run_id, "launch_error": launch_err, "job_status": status}
        return {"status": "ran", "rc": status["rc"], "wall": wall, "run_id": run_id, "jobdir": jobdir}

    # ---------------------------------------------------------------- transfers
    def fetch_verified(self, host: dict, remote_dir: str, local: Path) -> tuple[bool, dict]:
        if local.exists():
            return False, {"error": f"local target exists: {local}"}
        local.mkdir(parents=True)
        src = (remote_dir if host.get("ssh") is None else f"{host['ssh']}:{remote_dir}") + "/"
        r = subprocess.run(["rsync", "-a", "--exclude", "*.png", "--exclude", "runtime", src, str(local) + "/"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return False, {"error": f"rsync rc={r.returncode}: {r.stderr[-300:]}"}
        c = sh(host, f"cd {shlex.quote(remote_dir)} && find . -type f \\( -name '*.json' -o -name '*.pt' -o -name "
                     f"'*.jsonl' \\) -not -path './runtime/*' -print0 | xargs -0 -r sha256sum", 600)
        if c.returncode != 0:
            return False, {"error": "remote sha listing failed"}
        lines = [l for l in c.stdout.strip().splitlines() if l.strip()]
        bad = []
        for line in lines:
            h, rel = line.split(None, 1)
            p = local / (rel[2:] if rel.startswith("./") else rel)
            if not p.exists() or V.sha_file(p) != h:
                bad.append(rel)
        return (not bad and bool(lines)), {"n_files": len(lines), "mismatch": bad[:20]}

    def push_new(self, host: dict, local: Path, remote: str) -> bool:
        q = shlex.quote(remote)
        r = sh(host, f"test ! -e {q} && mkdir -p {shlex.quote(str(Path(remote).parent))} && echo FREE", 20)
        if "FREE" not in r.stdout:
            return False
        dst = remote if host.get("ssh") is None else f"{host['ssh']}:{remote}"
        if subprocess.run(["rsync", "-a", str(local), dst], capture_output=True).returncode != 0:
            return False
        c = sh(host, f"sha256sum {q}", 20)
        return c.returncode == 0 and c.stdout.split()[0] == V.sha_file(local)

    # ---------------------------------------------------------------- stages
    def backend_manifest(self, host: dict) -> tuple[str | None, dict]:
        out = f"{host['probe_root']}/recovery_backend_manifest_{uuid.uuid4().hex[:8]}.json"
        env = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in host["env"].items())
        try:
            r = sh(host, f"mkdir -p {shlex.quote(host['probe_root'])} && cd {shlex.quote(host['repo'])} && {env} "
                         f"{shlex.quote(host['env']['OPENWAM_PYTHON'])} scripts/probe/recovery/backend_manifest.py "
                         f"{shlex.quote(out)}", 3600)
        except subprocess.TimeoutExpired:
            return None, {"error": "backend manifest timeout"}
        local = self.root / Path(out).name
        dst = out if host.get("ssh") is None else f"{host['ssh']}:{out}"
        if r.returncode != 0 or subprocess.run(["rsync", "-a", dst, str(local)]).returncode != 0:
            return None, {"error": f"backend manifest failed rc={r.returncode}", "stdout": r.stdout[-500:]}
        body = json.loads(local.read_text())
        host["expect"] = {"git_head": body["openwam"]["head"], "tracked_digest": body["openwam"]["tracked_digest"]}
        return V.sha_file(local), body

    def stage_V(self) -> bool:
        st = self.ledger.state()
        if st["receipts"].get("V"):
            return True
        if "local_3090_offload_once" not in st["validation_attempts"]:
            self.ledger.update(validation_attempts=st["validation_attempts"] + ["local_3090_offload_once"],
                               local_candidate="withdrawn_not_attempted (R87: online offload residency not implemented)")
        cand = "remote_reference_once"
        if cand in self.ledger.state()["validation_attempts"]:
            self.log({"event": "V_candidate_already_attempted", "candidate": cand})
            return False
        host = self.backends[cand]
        if (r := self.preflight_refusal(host)):
            self.log({"event": "V_refused", "reason": r})
            return False
        bm_sha, bm = self.backend_manifest(host)
        if bm_sha is None or not bm.get("ok"):
            self.ledger.halt(f"backend manifest not ok: {bm.get('error') or bm.get('errors')}")
            return False
        self.ledger.update(validation_attempts=self.ledger.state()["validation_attempts"] + [cand])
        argv = [host["env"]["OPENWAM_PYTHON"], "scripts/probe/recovery/v_backend.py",
                "--out", f"{host['probe_root']}/recovery_V/{{RUN_ID}}", "--run-id", "{RUN_ID}",
                "--source-trace", host["v_source_trace"], "--source-trace-sha", host["v_source_trace_sha"],
                "--reference", host["v_reference"], "--reference-sha", V.V_REFERENCE_SHA]
        res = self.invoke("V", host, argv, 1800, {"candidate": cand},
                          {"validation_candidate": cand, "candidate_already_attempted": False})
        self.log({"event": "V_invocation", **{k: v for k, v in res.items() if k != "verdict"}})
        if res["status"] != "ran":
            return False
        local = self.root / "V" / res["run_id"]
        ok_fetch, info = self.fetch_verified(host, f"{host['probe_root']}/recovery_V/{res['run_id']}", local)
        rec = V.validate_V(local / "v_result.json", res["run_id"], res["rc"], bm_sha, self.root / "receipt_V.json",
                           fetch_ok=ok_fetch)
        passed = json.loads(Path(rec["path"]).read_text())["engineering_pass"]
        self.log({"event": "V_result", "pass": passed, "fetch": info})
        if passed:
            inv = self.ledger.state()["invocations"][-1]
            self.ledger.update(receipts={**self.ledger.state()["receipts"], "V": rec}, backend=host["policy_backend"],
                               backend_manifest_sha256=bm_sha, backend_name=cand, bound_gpu_uuid=inv["gpu_uuid"],
                               host_expect=host["expect"])
        return passed

    def _host(self) -> dict:
        st = self.ledger.state()
        host = dict(self.backends[st["backend_name"]])
        host["expect"] = st["host_expect"]
        return host

    def run_qualification(self, phase: str) -> bool:
        st = self.ledger.state()
        if st["receipts"].get(phase):
            return True
        host = self._host()
        cands, accepted = [], 0
        for k in range(self.policy["qualification_candidate_caps"][phase]):
            seed = self.policy["qualification_start_scene_seeds"][phase] + k
            remote = f"{host['probe_root']}/recovery_{phase}/cand_{seed}"
            argv = [host["env"]["ROBOTWIN_PYTHON"], "scripts/probe/recovery/job.py", "qualify", "--out", remote,
                    "--seed-param", str(V.SEED_PARAM[phase]), "--start-seed", str(seed)]
            res = self.invoke(phase, host, argv, self.policy["per_invocation_cap_seconds"][phase],
                              {"candidate_seed": seed}, {"backend_manifest_sha256": st["backend_manifest_sha256"]})
            if res["status"] != "ran":
                if not self.ledger.state()["halted"]:
                    self.ledger.halt(f"{phase} candidate {seed} not run: {res['status']}")
                return False
            local = self.root / phase / f"cand_{seed}"
            ok_fetch, info = self.fetch_verified(host, remote, local)
            cands.append({"seed": seed, "dir": str(local), "rc": res["rc"], "run_id": res["run_id"], "fetch_ok": ok_fetch})
            verdict = V.classify_candidate(local, res["rc"], seed) if ok_fetch else {"verdict": "infra"}
            self.log({"event": "candidate", "phase": phase, "seed": seed, "verdict": verdict["verdict"], "fetch": info})
            if verdict["verdict"] == "infra":
                break
            accepted += verdict["verdict"] == "accepted"
            if accepted == V.TARGET[phase]:
                break
        outd = self.root / f"{phase}_sealed"
        outd.mkdir(exist_ok=False)
        rec = V.validate_Q(phase, cands, st["backend_manifest_sha256"], outd, self.root / f"receipt_{phase}.json")
        body = json.loads(Path(rec["path"]).read_text())
        if body["engineering_pass"]:
            if not self.push_new(host, Path(body["manifest_path"]), f"{host['probe_root']}/recovery_{phase}/manifest.json"):
                self.ledger.halt(f"{phase} manifest push failed or target existed")
                return False
            self.ledger.update(receipts={**self.ledger.state()["receipts"], phase: rec})
            return True
        self.ledger.halt(f"{phase} qualification did not pass ({body.get('infra_failures')} infra)")
        return False

    def run_episodes(self, phase: str) -> bool:
        st = self.ledger.state()
        if st["receipts"].get(phase):
            return True
        host = self._host()
        qphase = "Q_eng" if phase == "M2" else "Q_eval"
        q = json.loads(Path(st["receipts"][qphase]["path"]).read_text())
        remote_manifest = f"{host['probe_root']}/recovery_{qphase}/manifest.json"
        order = json.loads((self.root / f"{qphase}_sealed" / "arm_order.json").read_text())
        grid, stop = [], False
        for ordn, seed in enumerate(q["accepted_scene_seeds"]):
            for arm in order[ordn]:
                remote = f"{host['probe_root']}/recovery_{phase}/ep{ordn:02d}_{arm}"
                cell = {"ordinal": ordn, "seed": seed, "arm": arm, "status": "not_run", "rc": None, "run_id": None,
                        "dir": None, "fetch_ok": False, "remote_dir": remote}
                if not stop:
                    argv = [host["env"]["ROBOTWIN_PYTHON"], "scripts/probe/recovery/job.py", "episode", "--out", remote,
                            "--arm", arm, "--manifest", remote_manifest, "--episode", str(ordn)]
                    res = self.invoke(phase, host, argv, self.policy["per_invocation_cap_seconds"][phase],
                                      {"ordinal": ordn, "seed": seed, "arm": arm},
                                      {"backend_manifest_sha256": st["backend_manifest_sha256"]})
                    cell.update(status=res["status"], rc=res.get("rc"), run_id=res.get("run_id"))
                    if res["status"] == "ran":
                        local = self.root / phase / f"ep{ordn:02d}_{arm}"
                        cell["fetch_ok"], cell["fetch"] = self.fetch_verified(host, remote, local)
                        cell["dir"] = str(local)
                    if phase == "M2" and (res["status"] != "ran" or res.get("rc") != 0 or not cell["fetch_ok"]):
                        if not self.ledger.state()["halted"]:
                            self.ledger.halt(f"M2 cell {ordn}/{arm} failed: {res['status']} rc={res.get('rc')}")
                        stop = True
                    if res["status"] in ("halted", "refused", "not_run_resource_deadline", "gate_refused"):
                        stop = True
                grid.append(cell)
                self.log({"event": "episode", "phase": phase, **{k: v for k, v in cell.items() if k != "fetch"}})
        rec = V.validate_episodes(phase, q, grid, st["backend_manifest_sha256"],
                                  (self.bundle or {}).get("scientific_bundle_sha256"),
                                  host["policy_backend"] == "local_3090", self.root / f"receipt_{phase}.json")
        self.ledger.update(receipts={**self.ledger.state()["receipts"], phase: rec})
        body = json.loads(Path(rec["path"]).read_text())
        if phase == "M2" and not body["engineering_pass"]:
            if not self.ledger.state()["halted"]:
                self.ledger.halt(f"M2 engineering gate failed: {body['problems'][:5]}")
            return False
        return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--backends", type=Path, required=True)
    ap.add_argument("--bundle", type=Path)
    ap.add_argument("--bundle-sha")
    ap.add_argument("--until", choices=ORDER, default="M3")
    a = ap.parse_args()
    a.root.mkdir(parents=True, exist_ok=True)
    bundle = gate.read_pinned({"path": str(a.bundle), "sha256": a.bundle_sha}) if a.bundle else None
    logf = open(a.root / "executor.log.jsonl", "a")

    def log(row):
        logf.write(json.dumps({"t": utcnow().isoformat(), **row}, default=str) + "\n")
        logf.flush()

    ex = Executor(a.root, json.loads(a.backends.read_text()), bundle, log)
    if ex.ledger.state()["halted"]:
        log({"event": "halted_at_start", "reason": ex.ledger.state()["halt_reason"]})
        return 3
    steps = {"V": ex.stage_V, "Q_eng": lambda: ex.run_qualification("Q_eng"), "M2": lambda: ex.run_episodes("M2"),
             "Q_eval": lambda: ex.run_qualification("Q_eval"), "M3": lambda: ex.run_episodes("M3")}
    for phase in ORDER[: ORDER.index(a.until) + 1]:
        if phase != "V" and bundle is None:
            log({"event": "stop_no_joint_bundle", "before": phase})
            return 0
        ok = steps[phase]()
        log({"event": "stage_done", "phase": phase, "ok": ok})
        if not ok:
            return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
