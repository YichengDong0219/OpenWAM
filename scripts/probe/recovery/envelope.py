"""Host-side envelope for one recovery invocation (runs detached on the GPU host; the controller only polls status).

spec.json: {token, argv, env, work_seconds, kill_grace_seconds, cleanup_seconds, repo, expect: {git_head,
            tracked_digest}, log_path, status_path}
1. verify the host worktree (HEAD, clean tracked tree, digest of `git ls-files -s`) before anything starts;
2. start the job in a new session with RECOVERY_JOB_TOKEN=<token>; record (pgid, uid, starttime) ownership;
3. wait <= work_seconds, then SIGTERM the job group, KILL after kill_grace_seconds;
4. clean every process group recorded by the job (JOB_PGIDS file "pgid starttime" lines) after re-verifying that its
   leader is ours (uid), unchanged (starttime) and carries the token; finally scan /proc for any survivor carrying the
   token; all of it within cleanup_seconds;
5. write status.json atomically. Total wall <= work + grace + cleanup (+ small bookkeeping).
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
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
    return uid == os.getuid() and starttime(pid) == st0 and proc_token(pid) == token


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


def tracked_digest(repo: str) -> tuple[str, str, bool]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    ls = subprocess.run(["git", "ls-files", "-s"], cwd=repo, capture_output=True, text=True).stdout
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo,
                                capture_output=True, text=True).stdout.strip())
    return head, hashlib.sha256(ls.encode()).hexdigest(), dirty


def main() -> int:
    spec = json.loads(Path(sys.argv[1]).read_text())
    status_path = Path(spec["status_path"])
    t0, t0u = time.monotonic(), time.strftime("%Y-%m-%dT%H:%M:%S%z")
    st = {"token": spec["token"], "start_utc": t0u, "rc": None}

    def finish(**kw):
        st.update(kw, end_utc=time.strftime("%Y-%m-%dT%H:%M:%S%z"), wall_s=round(time.monotonic() - t0, 2))
        tmp = status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=1)); os.replace(tmp, status_path)

    head, digest, dirty = tracked_digest(spec["repo"])
    st["host_code"] = {"git_head": head, "tracked_digest": digest, "dirty": dirty}
    exp = spec["expect"]
    if dirty or head != exp["git_head"] or digest != exp["tracked_digest"]:
        finish(rc="code_mismatch", cleanup={"groups": [], "verified_gone": True})
        return 2
    pg_file = Path(spec["log_path"]).with_suffix(".pgids")
    env = dict(os.environ, **{k: str(v) for k, v in spec["env"].items()}, RECOVERY_JOB_TOKEN=spec["token"],
               JOB_PGIDS=str(pg_file))
    with open(spec["log_path"], "w") as log:
        job = subprocess.Popen(spec["argv"], cwd=spec["repo"], env=env, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    groups = {job.pid: starttime(job.pid)}
    timed_out = False
    try:
        rc = job.wait(timeout=spec["work_seconds"])
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(job.pid, signal.SIGTERM)
        try:
            rc = job.wait(timeout=spec["kill_grace_seconds"])
        except subprocess.TimeoutExpired:
            os.killpg(job.pid, signal.SIGKILL)
            rc = job.wait()
    # cleanup within cleanup_seconds
    deadline = time.monotonic() + spec["cleanup_seconds"]
    if pg_file.exists():
        for line in pg_file.read_text().split("\n"):
            if line.strip():
                pid, s0 = line.split()
                groups.setdefault(int(pid), int(s0) if s0.isdigit() else None)
    killed = []
    for pid, s0 in groups.items():
        if s0 is not None and owned(pid, s0, spec["token"]):
            try:
                os.killpg(pid, signal.SIGTERM); killed.append(pid)
            except ProcessLookupError:
                pass
    while survivors(spec["token"]) and time.monotonic() < deadline - 3:
        time.sleep(0.5)
    for pid in survivors(spec["token"]):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(1)
    left = survivors(spec["token"])
    finish(rc=rc, timed_out=timed_out, cleanup={"groups": sorted(groups), "term_sent": killed, "survivors": left,
                                                "verified_gone": not left})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
