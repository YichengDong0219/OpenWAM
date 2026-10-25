"""Host-side envelope for one recovery invocation (runs detached on the GPU host; the controller only polls status).

spec.json: {token, argv, env, kill_grace_seconds, cleanup_seconds, repo, budget_seconds, expect: {openwam_head,
            openwam_bytes_digest, robotwin_head, robotwin_bytes_digest}, log_path, status_path}
1. bounded code check: SHA-256 over the on-disk bytes of every tracked file of the worktree and RoboTwin (expect values
   come from the jointly signed root, exported by the controller), probes with timeouts and rc checks;
2. hard end = envelope start (host clock) + budget_seconds (the controller already subtracted the launch slack, so no
   cross-host clock comparison is needed); the work window ends at hard end - kill_grace - cleanup - 5; refuse if < 60 s;
3. start the job in a new session with RECOVERY_JOB_TOKEN=<token>; on timeout TERM, then KILL after the grace, and
   reap with a bounded wait;
4. clean every recorded process group after re-verifying uid + starttime + token, then kill any token survivor;
5. ALWAYS write status.json atomically (also on any exception).
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path


def starttime(pid: int):
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
    except (OSError, IndexError, ValueError):
        return None


def proc_token(pid: int):
    try:
        env = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError:
        return None
    for kv in env:
        if kv.startswith(b"RECOVERY_JOB_TOKEN="):
            return kv.split(b"=", 1)[1].decode()
    return None


def owned(pid: int, st0, token: str) -> bool:
    try:
        uid = os.stat(f"/proc/{pid}").st_uid
    except OSError:
        return False
    return uid == os.getuid() and st0 is not None and starttime(pid) == st0 and proc_token(pid) == token


def survivors(token: str) -> list[int]:
    out = []
    for d in os.listdir("/proc"):
        if d.isdigit() and int(d) != os.getpid():
            try:
                if os.stat(f"/proc/{d}").st_uid == os.getuid() and proc_token(int(d)) == token:
                    out.append(int(d))
            except OSError:
                pass
    return out


def main() -> int:
    spec = json.loads(Path(sys.argv[1]).read_text())
    status_path = Path(spec["status_path"])
    t0 = time.monotonic()
    hard_end_mono = t0 + float(spec["budget_seconds"])
    st = {"token": spec["token"], "start_utc": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "rc": None,
          "cleanup": {"groups": [], "verified_gone": False}}

    def finish(**kw):
        st.update(kw, end_utc=time.strftime("%Y-%m-%dT%H:%M:%S%z"), wall_s=round(time.monotonic() - t0, 2))
        tmp = status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=1)); os.replace(tmp, status_path)

    groups: dict[int, int | None] = {}
    job = None
    try:
        sys.path.insert(0, str(Path(spec["repo"]) / "scripts/probe/recovery"))
        import runtime_identity as ri
        ri.errors.clear()
        env_rt = Path(spec["env"]["ROBOTWIN_PATH"])
        ow, rw = ri.code_digest(Path(spec["repo"])), ri.code_digest(env_rt)
        st["host_code"] = {"openwam": ow, "robotwin": rw, "errors": list(ri.errors)}
        exp = spec["expect"]
        if ri.errors or ow["head"] != exp["openwam_head"] or ow["bytes_digest"] != exp["openwam_bytes_digest"] \
                or rw["head"] != exp["robotwin_head"] or rw["bytes_digest"] != exp["robotwin_bytes_digest"]:
            st["cleanup"]["verified_gone"] = True
            finish(rc="code_mismatch")
            return 2
        work = hard_end_mono - spec["kill_grace_seconds"] - spec["cleanup_seconds"] - 5 - time.monotonic()
        if work < 60:
            st["cleanup"]["verified_gone"] = True
            finish(rc="insufficient_time", work_s=round(work, 1))
            return 2
        pg_file = Path(spec["log_path"]).with_suffix(".pgids")
        env = dict(os.environ, **{k: str(v) for k, v in spec["env"].items()}, RECOVERY_JOB_TOKEN=spec["token"],
                   JOB_PGIDS=str(pg_file))
        with open(spec["log_path"], "w") as log:
            job = subprocess.Popen(spec["argv"], cwd=spec["repo"], env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        groups[job.pid] = starttime(job.pid)
        timed_out, rc = False, None
        try:
            rc = job.wait(timeout=work)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(job.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                rc = job.wait(timeout=spec["kill_grace_seconds"])
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(job.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    rc = job.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    rc = "unreaped"
        st.update(rc=rc, timed_out=timed_out)
        if pg_file.exists():
            for line in pg_file.read_text().split("\n"):
                if line.strip():
                    pid, s0 = line.split()
                    groups.setdefault(int(pid), int(s0) if s0.isdigit() else None)
    except Exception as e:  # noqa: BLE001 - the status file must always be written
        st["envelope_error"] = f"{type(e).__name__}: {e}"
        st["traceback"] = traceback.format_exc()[-2000:]
        if st["rc"] is None:
            st["rc"] = "envelope_error"
    # bounded cleanup: ownership-verified group TERM, then KILL token survivors
    deadline = min(time.monotonic() + spec.get("cleanup_seconds", 60), hard_end_mono - 2)
    killed = []
    for pid, s0 in groups.items():
        if owned(pid, s0, spec["token"]):
            try:
                os.killpg(pid, signal.SIGTERM); killed.append(pid)
            except ProcessLookupError:
                pass
    while survivors(spec["token"]) and time.monotonic() < deadline - 5:
        time.sleep(0.5)
    for pid in survivors(spec["token"]):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(1)
    left = survivors(spec["token"])
    st["cleanup"] = {"groups": sorted(groups), "term_sent": killed, "survivors": left, "verified_gone": not left}
    finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
