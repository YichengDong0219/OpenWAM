"""Recovery pipeline executor (controller side): V -> Q_eng -> M2 -> Q_eval -> M3 under policy ba90fd66.

A jointly signed root bundle is REQUIRED for every phase (V included); before anything else the executor verifies the
bundle SHA, its policy SHA, both reviewers and every bound controller file. Remote expectations (code bytes, assets,
environments, backend manifest) are taken from the bundle only.

Per invocation: preflight (halt / deadline / backend) -> CPU runtime identity check against the bundle -> admission
(two strictly parsed nvidia-smi tables, 20 s total, unknown = refuse; GPU UUID bound after V) with persistent 600 s
grid waiting -> Codex gate -> durable reservation -> fresh job dir + spec carrying a relative budget -> detached host
envelope (code-byte check, token-owned job, bounded kill/reap, ownership-verified cleanup, status file) -> polling
that shares the single absolute remaining budget -> settle on verified status, otherwise an UNRESOLVED allocation
(reservation kept, pipeline halted). Process / transfer / engineering failures stop M2 and M3 at once; remaining
cells stay not_run and the stage is reported incomplete. No retries, no seed replacement, no second GPU.
Usage: executor.py --root DIR --backends backends.json --bundle B --bundle-sha S [--until PHASE]
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import math
import re
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
CLEANUP_S, POLL_S, LAUNCH_SLACK_S, SSH_MAX_S = 60, 15, 30, 20
UUID_RE = re.compile(r"^GPU-[0-9a-fA-F-]{8,}$")


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc)


def host_argv(host: dict, cmd: str, timeout: float) -> list[str]:
    if host.get("ssh") is None:
        return ["bash", "-c", cmd]
    return ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={min(15, max(2, int(timeout) - 1))}", host["ssh"], cmd]


def sh(host: dict, cmd: str, timeout: float, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(host_argv(host, cmd, timeout), input=stdin, capture_output=True, text=True, timeout=timeout)


def parse_gpu_tables(gpus_txt: str, apps_txt: str):
    """Strict parse; any malformed row raises ValueError (-> unknown)."""
    gpus, seen_idx, seen_uuid = [], set(), set()
    for l in gpus_txt.strip().splitlines():
        f = [x.strip() for x in l.split(",")]
        if len(f) != 3 or not f[0].isdigit() or not UUID_RE.match(f[1]) or not f[2].isdigit():
            raise ValueError(f"bad GPU row {l!r}")
        idx, uid, free = int(f[0]), f[1], int(f[2])
        if idx in seen_idx or uid in seen_uuid:
            raise ValueError("duplicate GPU index/uuid")
        seen_idx.add(idx); seen_uuid.add(uid)
        gpus.append({"index": idx, "uuid": uid, "free": free})
    if not gpus:
        raise ValueError("no GPU rows")
    busy = set()
    for l in apps_txt.strip().splitlines():
        if not l.strip():
            continue
        f = [x.strip() for x in l.split(",")]
        if len(f) != 2 or f[0] not in seen_uuid or not f[1].isdigit() or int(f[1]) <= 0:
            raise ValueError(f"bad process row {l!r}")
        busy.add(f[0])
    return gpus, busy


class Executor:
    def __init__(self, root: Path, backends: dict, bundle: dict, bundle_sha: str, log):
        raw = POLICY.read_bytes()
        if hashlib.sha256(raw).hexdigest() != POLICY_SHA:
            raise SystemExit("policy hash mismatch")
        self.policy = json.loads(raw)
        self.anchor = datetime.datetime.fromisoformat(self.policy["release_anchor_utc"])
        self.deadline = datetime.datetime.fromisoformat(self.policy["deadline_utc"])
        self.root, self.backends, self.bundle, self.bundle_sha, self.log = root, backends, bundle, bundle_sha, log
        (root / "ledger").mkdir(parents=True, exist_ok=True)
        self._lock = open(root / "ledger" / "controller.lock", "a")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("another controller holds the ledger lock")
        self.ledger = Ledger(root / "ledger")

    # ---------------------------------------------------------------- bundle
    def bundle_problems(self) -> list[str]:
        b, out = self.bundle, []
        if b.get("policy_sha256") != POLICY_SHA:
            out.append("bundle policy sha")
        if set(b.get("reviewers", [])) != {"codex", "claude"}:
            out.append("bundle reviewers")
        if not b.get("bound_files") or not b.get("validators") or not b.get("runtime_expect"):
            out.append("bundle incomplete")
        for path, h in (b.get("bound_files") or {}).items():
            p = Path(path)
            if not p.is_absolute() or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != h:
                out.append(f"bound file changed or not absolute: {path}")
        if any(v != V.validator_sha() for v in (b.get("validators") or {}).values()):
            out.append("validator sha differs from the running validators.py")
        if b.get("selection_algorithm_sha256") != V.selection_algorithm_sha():
            out.append("selection algorithm sha differs")
        return out

    def host(self, name: str) -> dict:
        h = dict(self.backends[name])
        rx = self.bundle["runtime_expect"]
        h["expect"] = {"openwam_head": rx["openwam_code"]["head"], "openwam_bytes_digest": rx["openwam_code"]["bytes_digest"],
                       "robotwin_head": rx["robotwin_code"]["head"], "robotwin_bytes_digest": rx["robotwin_code"]["bytes_digest"]}
        return h

    def runtime_check(self, host: dict) -> tuple[bool, dict]:
        """CPU identity probe on the host before any reservation; must equal the signed runtime_expect exactly."""
        env = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in host["env"].items())
        try:
            r = sh(host, f"cd {shlex.quote(host['repo'])} && {env} {shlex.quote(host['env']['OPENWAM_PYTHON'])} "
                         f"scripts/probe/recovery/runtime_identity.py", 600)
            got = json.loads(r.stdout) if r.returncode == 0 else {"rc": r.returncode, "stderr": r.stderr[-300:]}
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
            got = {"error": type(e).__name__}
        exp = self.bundle["runtime_expect"]
        diff = sorted(k for k in set(exp) | set(got) if k not in ("errors", "ok", "format") and exp.get(k) != got.get(k))
        ok = got.get("ok") is True and not diff
        return ok, {"diff": diff}

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
            gpus, busy = parse_gpu_tables(outs["gpus"], outs["apps"])
        except ValueError as e:
            return {**snap, "ok": False, "reason": f"parse: {e}"}
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
        ok, info = self.runtime_check(host)
        self.log({"event": "runtime_check", "phase": phase, "ok": ok, **info})
        if not ok:
            self.ledger.halt(f"runtime identity differs from the signed root before {phase}: {info['diff'][:5]}")
            return {"status": "halted", "reason": "runtime drift"}
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
        if hard - LAUNCH_SLACK_S - grace - CLEANUP_S - 5 < 60:
            return {"status": "gate_refused", "verdict": {**verdict, "reasons": ["work budget after cleanup < 60 s"]}}
        run_id = f"{phase}-{st['attempted_invocations'][phase] + 1:03d}-{uuid.uuid4().hex[:10]}"
        jobdir = f"{host['probe_root']}/recovery_jobs/{run_id}"
        spec = {"token": uuid.uuid4().hex, "argv": [a.replace("{RUN_ID}", run_id) for a in job_argv],
                "env": {**host["env"], "CUDA_VISIBLE_DEVICES": adm["chosen"]["index"], "RECOVERY_RUN_ID": run_id,
                        "PYTHONPATH": f"{host['repo']}:{host['repo']}/benchmarks/robotwin"},
                "kill_grace_seconds": grace, "cleanup_seconds": CLEANUP_S, "repo": host["repo"], "expect": host["expect"],
                "log_path": f"{jobdir}/job.log", "status_path": f"{jobdir}/status.json"}
        self.ledger.reserve(phase, hard, {**inv, "run_id": run_id, "host": host["name"],
                                          "gpu_uuid": adm["chosen"]["uuid"], "gpu_index": adm["chosen"]["index"]})
        t0 = time.monotonic()
        end = t0 + hard

        def left():
            return end - time.monotonic()

        spec["budget_seconds"] = round(left() - LAUNCH_SLACK_S, 1)
        (self.root / "specs").mkdir(exist_ok=True)
        (self.root / "specs" / f"{run_id}.json").write_text(json.dumps(spec))
        status, launch_err = None, None
        q = shlex.quote
        try:
            r = sh(host, f"test ! -e {q(jobdir)} && mkdir -p {q(jobdir)} && cat > {q(jobdir)}/spec.json && "
                         f"cd {q(host['repo'])} && (setsid nohup {q(host['env']['OPENWAM_PYTHON'])} "
                         f"scripts/probe/recovery/envelope.py {q(jobdir)}/spec.json > {q(jobdir)}/envelope.log "
                         f"2>&1 < /dev/null &) && echo LAUNCHED", min(LAUNCH_SLACK_S, max(1, left() - 1)),
                   stdin=json.dumps(spec))
            if "LAUNCHED" not in r.stdout:
                launch_err = f"launch failed rc={r.returncode}: {r.stderr[-300:]}"
            while launch_err is None and left() > 2:
                ssh_t = min(SSH_MAX_S, left() - 1)
                time.sleep(max(0.0, min(POLL_S, left() - ssh_t - 1)))
                ssh_t = min(SSH_MAX_S, left() - 1)
                if ssh_t < 1:
                    break
                try:
                    c = sh(host, f"cat {q(spec['status_path'])} 2>/dev/null", ssh_t)
                    if c.returncode == 0 and c.stdout.strip():
                        status = json.loads(c.stdout)
                        break
                except (subprocess.TimeoutExpired, json.JSONDecodeError):
                    pass
        except Exception as e:  # noqa: BLE001 - the reservation must still be resolved
            launch_err = f"{type(e).__name__}: {e}"
        wall = time.monotonic() - t0
        verified = (status is not None and status.get("token") == spec["token"]
                    and (status.get("cleanup") or {}).get("verified_gone") is True)
        if not verified:
            self.ledger.unresolved(f"{phase} {run_id}: no verified end/cleanup", {"launch_error": launch_err,
                                   "status": status, "elapsed_s": round(wall, 1)})
            return {"status": "halted", "run_id": run_id, "launch_error": launch_err, "job_status": status}
        self.ledger.settle(wall, {"run_id": run_id, "rc": status.get("rc"), "cleanup_verified": True,
                                  "timed_out": status.get("timed_out")})
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
        try:
            c = sh(host, f"cd {shlex.quote(remote_dir)} && find . -type f \\( -name '*.json' -o -name '*.pt' -o -name "
                         f"'*.jsonl' \\) -not -path './runtime/*' -print0 | xargs -0 -r sha256sum", 600)
        except subprocess.TimeoutExpired:
            return False, {"error": "remote sha listing timeout"}
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

    # ---------------------------------------------------------------- job command lines (overridable in tests)
    def argv_v(self, host):
        return [host["env"]["OPENWAM_PYTHON"], "scripts/probe/recovery/v_backend.py",
                "--out", f"{host['probe_root']}/recovery_V/{{RUN_ID}}", "--run-id", "{RUN_ID}",
                "--source-trace", host["v_source_trace"], "--source-trace-sha", V.V_SOURCE_SHA,
                "--reference", host["v_reference"], "--reference-sha", V.V_REFERENCE_SHA]

    def argv_qualify(self, host, phase, seed, out):
        return [host["env"]["ROBOTWIN_PYTHON"], "scripts/probe/recovery/job.py", "qualify", "--out", out,
                "--seed-param", str(V.SEED_PARAM[phase]), "--start-seed", str(seed)]

    def argv_episode(self, host, out, arm, manifest, manifest_sha, ordinal, seed):
        return [host["env"]["ROBOTWIN_PYTHON"], "scripts/probe/recovery/job.py", "episode", "--out", out, "--arm", arm,
                "--manifest", manifest, "--expected-manifest-sha", manifest_sha, "--episode", str(ordinal),
                "--seed", str(seed)]

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
        return V.sha_file(local), json.loads(local.read_text())

    # ---------------------------------------------------------------- stages
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
        host = self.host(cand)
        if (r := self.preflight_refusal(host)):
            self.log({"event": "V_refused", "reason": r})
            return False
        bm_sha, bm = self.backend_manifest(host)
        if bm_sha is None or not bm.get("ok") or bm_sha != self.bundle.get("backend_manifest_sha256"):
            self.ledger.halt(f"backend manifest {bm_sha} not ok or differs from the signed root "
                             f"{self.bundle.get('backend_manifest_sha256')}")
            return False
        self.ledger.update(backend_manifest_sha256=bm_sha)
        self.ledger.update(validation_attempts=self.ledger.state()["validation_attempts"] + [cand])
        res = self.invoke("V", host, self.argv_v(host), 1800, {"candidate": cand},
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
                               backend_name=cand, bound_gpu_uuid=inv["gpu_uuid"])
        else:
            self.ledger.halt("V did not pass")
        return passed

    def run_qualification(self, phase: str) -> bool:
        st = self.ledger.state()
        if st["receipts"].get(phase):
            return True
        host = self.host(st["backend_name"])
        cands, accepted = [], 0
        for k in range(self.policy["qualification_candidate_caps"][phase]):
            seed = self.policy["qualification_start_scene_seeds"][phase] + k
            remote = f"{host['probe_root']}/recovery_{phase}/cand_{seed}"
            res = self.invoke(phase, host, self.argv_qualify(host, phase, seed, remote),
                              self.policy["per_invocation_cap_seconds"][phase], {"candidate_seed": seed},
                              {"backend_manifest_sha256": st["backend_manifest_sha256"]})
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
        host = self.host(st["backend_name"])
        qphase = "Q_eng" if phase == "M2" else "Q_eval"
        q = json.loads(Path(st["receipts"][qphase]["path"]).read_text())
        remote_manifest = f"{host['probe_root']}/recovery_{qphase}/manifest.json"
        order = json.loads((self.root / f"{qphase}_sealed" / "arm_order.json").read_text())
        if V.sha_file(self.root / f"{qphase}_sealed" / "arm_order.json") != q["arm_order_sha256"]:
            self.ledger.halt(f"{phase}: arm order file differs from the sealed receipt")
            return False
        grid, stop_reason = [], None
        for ordn, seed in enumerate(q["accepted_scene_seeds"]):
            for arm in order[ordn]:
                remote = f"{host['probe_root']}/recovery_{phase}/ep{ordn:02d}_{arm}"
                cell = {"ordinal": ordn, "seed": seed, "arm": arm, "status": "not_run", "rc": None, "run_id": None,
                        "dir": None, "fetch_ok": False, "remote_dir": remote}
                if stop_reason is None:
                    res = self.invoke(phase, host, self.argv_episode(host, remote, arm, remote_manifest,
                                                                     q["manifest_sha256"], ordn, seed),
                                      self.policy["per_invocation_cap_seconds"][phase],
                                      {"ordinal": ordn, "seed": seed, "arm": arm},
                                      {"backend_manifest_sha256": st["backend_manifest_sha256"]})
                    cell.update(status=res["status"], rc=res.get("rc"), run_id=res.get("run_id"))
                    if res["status"] == "ran":
                        local = self.root / phase / f"ep{ordn:02d}_{arm}"
                        cell["fetch_ok"], cell["fetch"] = self.fetch_verified(host, remote, local)
                        cell["dir"] = str(local)
                    # policy failures are rc 0 (success=False) and continue; anything else stops all new calls
                    if res["status"] != "ran":
                        stop_reason = f"cell {ordn}/{arm}: {res['status']}"
                    elif res.get("rc") != 0:
                        stop_reason = f"cell {ordn}/{arm}: process rc={res.get('rc')}"
                    elif not cell["fetch_ok"]:
                        stop_reason = f"cell {ordn}/{arm}: transfer not verified"
                    if stop_reason and not self.ledger.state()["halted"]:
                        self.ledger.halt(f"{phase} stopped: {stop_reason}")
                grid.append(cell)
                self.log({"event": "episode", "phase": phase, **{k: v for k, v in cell.items() if k != "fetch"}})
        rec = V.validate_episodes(phase, q, grid, st["backend_manifest_sha256"], self.bundle.get("scientific_bundle_sha256"),
                                  host["policy_backend"] == "local_3090", self.root / f"receipt_{phase}.json",
                                  stage_stop_reason=stop_reason)
        self.ledger.update(receipts={**self.ledger.state()["receipts"], phase: rec})
        body = json.loads(Path(rec["path"]).read_text())
        if stop_reason is not None:
            return False
        if phase == "M2" and not body["engineering_pass"]:
            self.ledger.halt(f"M2 engineering gate failed: {body['problems'][:5]}")
            return False
        return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--backends", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--bundle-sha", required=True)
    ap.add_argument("--until", choices=ORDER, default="M3")
    a = ap.parse_args()
    a.root.mkdir(parents=True, exist_ok=True)
    bundle = gate.read_pinned({"path": str(a.bundle), "sha256": a.bundle_sha})
    logf = open(a.root / "executor.log.jsonl", "a")

    def log(row):
        logf.write(json.dumps({"t": utcnow().isoformat(), **row}, default=str) + "\n")
        logf.flush()

    ex = make_executor(a.root, json.loads(a.backends.read_text()), bundle, a.bundle_sha, log)
    return run_pipeline(ex, a.until, log)


def make_executor(root, backends, bundle, bundle_sha, log):
    return Executor(root, backends, bundle, bundle_sha, log)


def run_pipeline(ex: Executor, until: str, log) -> int:
    probs = ex.bundle_problems()
    if probs:
        log({"event": "bundle_rejected", "problems": probs})
        return 2
    if ex.ledger.state()["halted"]:
        log({"event": "halted_at_start", "reason": ex.ledger.state()["halt_reason"]})
        return 3
    steps = {"V": ex.stage_V, "Q_eng": lambda: ex.run_qualification("Q_eng"), "M2": lambda: ex.run_episodes("M2"),
             "Q_eval": lambda: ex.run_qualification("Q_eval"), "M3": lambda: ex.run_episodes("M3")}
    for phase in ORDER[: ORDER.index(until) + 1]:
        ok = steps[phase]()
        log({"event": "stage_done", "phase": phase, "ok": ok})
        if not ok:
            return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
