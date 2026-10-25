"""Recovery pipeline executor: V -> Q_eng -> M2 -> Q_eval -> M3 under the Codex R86 resource policy.

Runs on the controller machine. Before EVERY invocation: admission query on the target host/GPU (20 s timeout,
failure = unknown = no admission) -> Codex gate verdict (pinned policy, joint bundle for non-V phases) -> durable
reservation of the full allowed wall -> launch under `timeout -k <grace> <work>` -> settle actual monotonic wall.
When the GPU is not clear it waits for the next 600 s slot (at most 432 slots, never past the absolute deadline).
Process/engineering failures halt the pipeline; nothing is retried, no seed is replaced, no second GPU is used.

Usage: executor.py --root DIR --backends backends.json [--bundle B --bundle-sha S] [--until PHASE]
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_codex_v1 as gate  # noqa: E402
import validators as V  # noqa: E402
from ledger import Ledger  # noqa: E402

POLICY = HERE / "policy_codex_R86.json"
POLICY_SHA = "ba90fd6623cfac7af0db3b9d9c7eb3cedcb10c68e2cd122c02a088fce3d6a842"
ORDER = ["V", "Q_eng", "M2", "Q_eval", "M3"]


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc)


def sh(host: dict, cmd: str, timeout: float) -> subprocess.CompletedProcess:
    argv = ["bash", "-lc", cmd] if host.get("ssh") is None else ["ssh", "-o", "BatchMode=yes", "-o",
                                                                   "ConnectTimeout=15", host["ssh"], cmd]
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


class Executor:
    def __init__(self, root: Path, backends: dict, bundle: dict | None, log):
        raw = POLICY.read_bytes()
        if hashlib.sha256(raw).hexdigest() != POLICY_SHA:
            raise SystemExit("policy hash mismatch")
        self.policy = json.loads(raw)
        self.root, self.backends, self.bundle, self.log = root, backends, bundle, log
        self.ledger = Ledger(root / "ledger")
        self.slots_used = 0

    # ---------------------------------------------------------------- admission and waiting
    def admission(self, host: dict) -> dict:
        snap = {"time": utcnow().isoformat(), "host": host["name"]}
        q = ("nvidia-smi --query-gpu=index,uuid,memory.free --format=csv,noheader,nounits; echo ===; "
             "nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits")
        try:
            r = sh(host, q, 20)
        except subprocess.TimeoutExpired:
            return {**snap, "ok": False, "reason": "timeout"}
        if r.returncode != 0 or "===" not in r.stdout:
            return {**snap, "ok": False, "reason": f"rc={r.returncode}", "stderr": r.stderr[-300:]}
        gpus_txt, apps_txt = r.stdout.split("===")
        busy = {l.split(",")[0].strip() for l in apps_txt.strip().splitlines() if l.strip()}
        cands = []
        for l in gpus_txt.strip().splitlines():
            idx, uuid, free = [x.strip() for x in l.split(",")]
            if int(idx) in host["eligible_gpus"]:
                cands.append({"index": int(idx), "uuid": uuid, "free": int(free), "busy": uuid in busy})
        clear = [c for c in cands if not c["busy"] and c["free"] >= host["min_free_mib"]]
        best = max(clear, key=lambda c: c["free"]) if clear else None
        return {**snap, "ok": True, "gpus": cands, "chosen": best}

    def wait_for_gpu(self, host: dict) -> dict | None:
        deadline = datetime.datetime.fromisoformat(self.policy["deadline_utc"])
        while True:
            adm = self.admission(host)
            self.log({"event": "admission", **adm, "slot": self.slots_used})
            if adm["ok"] and adm["chosen"] is not None:
                return adm
            self.slots_used += 1
            nxt = utcnow() + datetime.timedelta(seconds=self.policy["wait"]["interval_seconds"])
            if self.slots_used >= self.policy["wait"]["maximum_slots"] or nxt >= deadline:
                self.log({"event": "wait_exhausted", "slots": self.slots_used})
                return None
            time.sleep(self.policy["wait"]["interval_seconds"])

    # ---------------------------------------------------------------- one invocation
    def invoke(self, phase: str, host: dict, job_argv: list[str], requested: int, inv: dict, extra_state=None,
               inv_out: str | None = None):
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
        hard, work, grace = verdict["max_wall_seconds_including_cleanup"], verdict["max_work_seconds"], self.policy["kill_grace_seconds"]
        self.ledger.reserve(phase, hard, {**inv, "host": host["name"], "gpu_uuid": adm["chosen"]["uuid"],
                                          "gpu_index": adm["chosen"]["index"]})
        env = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in {**host["env"], "CUDA_VISIBLE_DEVICES": adm["chosen"]["index"]}.items())
        logdir = f"{host['probe_root']}/recovery_logs"
        joblog = f"{logdir}/{phase}_{int(time.time())}_{self.ledger.state()['attempted_invocations'][phase]}.log"
        # output goes to a host-side log (an orphaned grandchild must not keep a controller pipe open)
        cmd = (f"mkdir -p {shlex.quote(logdir)} && cd {shlex.quote(host['repo'])} && {env} timeout -k {grace} {max(1, work)} "
               + " ".join(shlex.quote(x) for x in job_argv) + f" > {shlex.quote(joblog)} 2>&1; echo RC=$?")
        t0 = time.monotonic()
        try:
            r = sh(host, cmd, hard + 120)
            m = [l for l in r.stdout.splitlines() if l.startswith("RC=")]
            rc = int(m[-1][3:]) if m else f"no_rc(ssh={r.returncode})"
            tail = r.stderr[-500:]
        except subprocess.TimeoutExpired:
            rc, tail = "controller_timeout", ""
        # kill any of OUR recorded process groups that survived (SIGKILL after grace skips job.py's handler), verify
        own_gone = True
        if inv_out:
            pg = shlex.quote(f"{inv_out}/pgids.txt")
            clean = (f"if [ -f {pg} ]; then for g in $(cat {pg}); do kill -TERM -$g 2>/dev/null; done; sleep 5; "
                     f"for g in $(cat {pg}); do kill -KILL -$g 2>/dev/null; done; sleep 2; "
                     f"for g in $(cat {pg}); do pgrep -g $g >/dev/null && echo ALIVE_$g; done; fi; echo CLEANUP_DONE")
            try:
                c = sh(host, clean, 90)
                own_gone = "CLEANUP_DONE" in c.stdout and "ALIVE_" not in c.stdout
            except subprocess.TimeoutExpired:
                own_gone = False
        wall = time.monotonic() - t0
        self.ledger.settle(wall, {"rc": rc, "own_process_clear": bool(own_gone), "stderr_tail": tail, "job_log": joblog})
        if not own_gone:
            self.ledger.halt(f"own processes of {phase} invocation {inv} not verified gone")
        return {"status": "ran", "rc": rc, "wall": wall}

    # ---------------------------------------------------------------- stages
    def stage_V(self) -> bool:
        st = self.ledger.state()
        if st["receipts"].get("V"):
            return True
        for cand in self.policy["validation_candidates"]:
            if cand in st["validation_attempts"]:
                continue
            host = self.backends[cand]
            out = f"{host['probe_root']}/recovery_V_{cand}"
            argv = [host["env"]["OPENWAM_PYTHON"], "scripts/probe/recovery/v_backend.py", "--out", out,
                    "--source-trace", host["v_source_trace"], "--source-trace-sha", host["v_source_trace_sha"],
                    "--reference", host["v_reference"], "--reference-sha", "8e124efdf1dc1210c8bdbd81cec2678015d19bff5168c211aa1a82d57780aa38",
                    "--residency", host["residency"]]
            self.ledger.update(validation_attempts=st["validation_attempts"] + [cand])
            res = self.invoke("V", host, argv, 1800, {"candidate": cand},
                              {"validation_candidate": cand, "candidate_already_attempted": False})
            st = self.ledger.state()
            if res["status"] != "ran":
                self.log({"event": "V_not_run", "candidate": cand, **res})
                return False
            local_copy = self.fetch(host, f"{out}/v_result.json", self.root / f"V_{cand}_v_result.json")
            bm = self.backend_manifest(host)
            rec = V.validate_V(local_copy, bm, self.root / f"receipt_V_{cand}.json") if local_copy else None
            ok = rec is not None and json.loads(Path(rec["path"]).read_text())["engineering_pass"]
            self.log({"event": "V_result", "candidate": cand, "pass": ok})
            if ok:
                self.ledger.update(receipts={**st["receipts"], "V": rec}, backend=host["policy_backend"],
                                   backend_manifest_sha256=bm, backend_name=cand)
                return True
        return False

    def run_qualification(self, phase: str) -> bool:
        st = self.ledger.state()
        if st["receipts"].get(phase):
            return True
        host = self.backends[st["backend_name"]]
        cands, accepted = [], 0
        for k in range(self.policy["qualification_candidate_caps"][phase]):
            seed = self.policy["qualification_start_scene_seeds"][phase] + k
            out = f"{host['probe_root']}/recovery_{phase}/cand_{seed}"
            argv = [host["env"]["ROBOTWIN_PYTHON"], "scripts/probe/recovery/job.py", "qualify", "--out", out,
                    "--seed-param", str(V.SEED_PARAM[phase]), "--start-seed", str(seed)]
            res = self.invoke(phase, host, argv, self.policy["per_invocation_cap_seconds"][phase],
                              {"candidate_seed": seed}, {"backend_manifest_sha256": st["backend_manifest_sha256"]},
                              inv_out=out)
            if res["status"] != "ran":
                self.ledger.halt(f"{phase} candidate {seed} not run: {res['status']}"); return False
            local = self.fetch_dir(host, out, self.root / f"{phase}/cand_{seed}")
            cands.append({"seed": seed, "dir": str(local), "rc": res["rc"]})
            verdict = V.classify_candidate(local, res["rc"])["verdict"]
            self.log({"event": "candidate", "phase": phase, "seed": seed, "verdict": verdict})
            if verdict == "infra":
                self.ledger.halt(f"{phase} candidate {seed} infrastructure failure"); break
            accepted += verdict == "accepted"
            if accepted == V.TARGET[phase]:
                break
        outd = self.root / phase
        outd.mkdir(exist_ok=True)
        rec = V.validate_Q(phase, cands, st["backend_manifest_sha256"], outd, self.root / f"receipt_{phase}.json")
        body = json.loads(Path(rec["path"]).read_text())
        if body["engineering_pass"]:
            self.push(host, Path(body["manifest_path"]), f"{host['probe_root']}/recovery_{phase}/manifest.json")
            self.ledger.update(receipts={**self.ledger.state()["receipts"], phase: rec})
            return True
        self.ledger.halt(f"{phase} qualification did not pass: {body.get('infra_failures')} infra, "
                         f"{len(body.get('candidates', []))} candidates")
        return False

    def run_episodes(self, phase: str) -> bool:
        st = self.ledger.state()
        if st["receipts"].get(phase):
            return True
        host = self.backends[st["backend_name"]]
        qphase = "Q_eng" if phase == "M2" else "Q_eval"
        q = json.loads(Path(st["receipts"][qphase]["path"]).read_text())
        remote_manifest = f"{host['probe_root']}/recovery_{qphase}/manifest.json"
        order = json.loads((self.root / qphase / "arm_order.json").read_text())
        episodes = []
        for ordn, seed in enumerate(q["accepted_scene_seeds"]):
            for arm in order[ordn]:
                out = f"{host['probe_root']}/recovery_{phase}/ep{ordn:02d}_{arm}"
                argv = [host["env"]["ROBOTWIN_PYTHON"], "scripts/probe/recovery/job.py", "episode", "--out", out,
                        "--arm", arm, "--manifest", remote_manifest, "--episode", str(ordn)]
                res = self.invoke(phase, host, argv, self.policy["per_invocation_cap_seconds"][phase],
                                  {"ordinal": ordn, "seed": seed, "arm": arm},
                                  {"backend_manifest_sha256": st["backend_manifest_sha256"]}, inv_out=out)
                if res["status"] != "ran":
                    self.log({"event": "episode_not_run", "phase": phase, "ordinal": ordn, "arm": arm, **res})
                    if phase == "M2":
                        self.ledger.halt(f"M2 episode {ordn}/{arm} not run: {res['status']}"); return False
                    episodes.append({"ordinal": ordn, "seed": seed, "arm": arm, "dir": None, "rc": res["status"]})
                    continue
                local = self.fetch_dir(host, out, self.root / f"{phase}/ep{ordn:02d}_{arm}")
                episodes.append({"ordinal": ordn, "seed": seed, "arm": arm, "dir": str(local), "rc": res["rc"]})
                if phase == "M2" and res["rc"] != 0:
                    self.ledger.halt(f"M2 episode {ordn}/{arm} failed rc={res['rc']}")
                    break
        rec = V.validate_episodes(phase, q, [e for e in episodes if e["dir"]], st["backend_manifest_sha256"],
                                  (self.bundle or {}).get("scientific_bundle_sha256"),
                                  host["policy_backend"] == "local_3090", self.root / f"receipt_{phase}.json")
        self.ledger.update(receipts={**self.ledger.state()["receipts"], phase: rec})
        body = json.loads(Path(rec["path"]).read_text())
        if phase == "M2" and not body["engineering_pass"]:
            self.ledger.halt(f"M2 engineering gate failed: {body['problems'][:5]}")
            return False
        return True

    # ---------------------------------------------------------------- file transfer / manifests
    def fetch(self, host, remote: str, local: Path):
        local.parent.mkdir(parents=True, exist_ok=True)
        src = remote if host.get("ssh") is None else f"{host['ssh']}:{remote}"
        r = subprocess.run(["rsync", "-a", src, str(local)], capture_output=True, text=True)
        return local if r.returncode == 0 and local.exists() else None

    def fetch_dir(self, host, remote: str, local: Path) -> Path:
        local.mkdir(parents=True, exist_ok=True)
        src = (remote if host.get("ssh") is None else f"{host['ssh']}:{remote}") + "/"
        subprocess.run(["rsync", "-a", "--exclude", "*.png", src, str(local) + "/"], capture_output=True, text=True)
        return local

    def push(self, host, local: Path, remote: str):
        if host.get("ssh") is None:
            Path(remote).parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["cp", "-n", str(local), remote], check=True)
        else:
            subprocess.run(["ssh", host["ssh"], f"mkdir -p {shlex.quote(str(Path(remote).parent))}"], check=True)
            subprocess.run(["rsync", "-a", "--ignore-existing", str(local), f"{host['ssh']}:{remote}"], check=True)

    def backend_manifest(self, host) -> str:
        r = sh(host, f"cd {shlex.quote(host['repo'])} && git rev-parse HEAD && git status --porcelain --untracked-files=no "
                     f"&& sha256sum assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboTwin-Full/*.safetensors "
                     f"configs/deploy_probe.yaml && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader "
                     f"&& {host['env']['OPENWAM_PYTHON']} -c 'import torch;print(torch.__version__,torch.version.cuda)'", 900)
        body = {"host": host["name"], "policy_backend": host["policy_backend"], "residency": host["residency"],
                "probe": r.stdout, "rc": r.returncode}
        p = self.root / f"backend_manifest_{host['name']}.json"
        p.write_text(json.dumps(body, indent=1, sort_keys=True))
        return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--backends", type=Path, required=True)
    ap.add_argument("--bundle", type=Path)
    ap.add_argument("--bundle-sha")
    ap.add_argument("--until", choices=ORDER, default="M3")
    a = ap.parse_args()
    a.root.mkdir(parents=True, exist_ok=True)
    bundle = None
    if a.bundle:
        bundle = gate.read_pinned({"path": str(a.bundle), "sha256": a.bundle_sha})
    logf = open(a.root / "executor.log.jsonl", "a")

    def log(row):
        logf.write(json.dumps({"t": utcnow().isoformat(), **row}, default=str) + "\n"); logf.flush()

    ex = Executor(a.root, json.loads(a.backends.read_text()), bundle, log)
    if ex.ledger.state()["halted"]:
        log({"event": "halted_at_start", "reason": ex.ledger.state()["halt_reason"]}); return 3
    steps = {"V": ex.stage_V, "Q_eng": lambda: ex.run_qualification("Q_eng"), "M2": lambda: ex.run_episodes("M2"),
             "Q_eval": lambda: ex.run_qualification("Q_eval"), "M3": lambda: ex.run_episodes("M3")}
    for phase in ORDER[: ORDER.index(a.until) + 1]:
        if phase != "V" and bundle is None:
            log({"event": "stop_no_joint_bundle", "before": phase}); return 0
        ok = steps[phase]()
        log({"event": "stage_done", "phase": phase, "ok": ok})
        if not ok:
            return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
