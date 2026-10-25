"""GPU-host side of one recovery invocation (launched by executor.py under its timeout; CUDA_VISIBLE_DEVICES set).

  qualify --out DIR --seed-param P --start-seed S       one official expert check of candidate scene seed S
  episode --out DIR --arm A --manifest M --episode I    probe server + one manifest episode for arm A

Every job writes DIR/job.json (exit code, timings, GPU memory samples summary) and never touches another
process than the ones it started. Environment: ROBOTWIN_PYTHON, OPENWAM_PYTHON, CUDA_HOME_BIN.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
CKPT = ROOT / "assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboTwin-Full"
CONFIG = ROOT / "configs/deploy_probe.yaml"


def base_env(out: Path) -> dict:
    robot_python = os.environ["ROBOTWIN_PYTHON"]
    env = dict(os.environ,
               PATH=f"{Path(robot_python).parent}:{os.environ.get('CUDA_HOME_BIN', '/usr/local/cuda-12.4/bin')}:{os.environ.get('PATH', '')}",
               PYTHONPATH=f"{ROOT}:{ROOT / 'benchmarks/robotwin'}:{os.environ.get('PYTHONPATH', '')}",
               PYTHONUNBUFFERED="1", ROBOTWIN_RUNTIME_ROOT=str(out / "runtime"))
    for k in ("ROBOTWIN_PROBE_STEP_LIMIT", "ROBOTWIN_ENABLE_PLANNER_FALLBACK", "ROBOTWIN_RECOVERY_ARM",
              "ROBOTWIN_START_SEED", "ROBOTWIN_MAX_CANDIDATES", "ROBOTWIN_SEED_PARAM"):
        env.pop(k, None)
    egl = Path(robot_python).parent.parent / "lib/python3.10/site-packages/sapien/vulkan_library/10_nvidia.json"
    if egl.exists():
        env.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", str(egl))
    return env


def render_device() -> str:
    pci = subprocess.check_output(["nvidia-smi", "--id=" + os.environ["CUDA_VISIBLE_DEVICES"],
                                   "--query-gpu=pci.bus_id", "--format=csv,noheader"], text=True, timeout=20).strip()
    return "pci:" + pci[-12:]


class MemSampler(threading.Thread):
    """Samples the allocated GPU every 5 s (discrete samples, not a continuous peak proof)."""

    def __init__(self, path: Path):
        super().__init__(daemon=True)
        self.path, self.stop_flag, self.min_free, self.max_used, self.n = path, threading.Event(), None, None, 0

    def run(self):
        idx = os.environ["CUDA_VISIBLE_DEVICES"]
        with open(self.path, "w") as f:
            while not self.stop_flag.is_set():
                try:
                    out = subprocess.run(["nvidia-smi", f"--id={idx}", "--query-gpu=memory.used,memory.free",
                                          "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                         timeout=20).stdout.strip()
                    used, free = [int(x) for x in out.split(",")]
                    f.write(f"{time.time():.1f},{used},{free}\n"); f.flush()
                    self.n += 1
                    self.min_free = free if self.min_free is None else min(self.min_free, free)
                    self.max_used = used if self.max_used is None else max(self.max_used, used)
                except Exception as e:  # noqa: BLE001 - sampling must not kill the job
                    f.write(f"{time.time():.1f},error,{type(e).__name__}\n"); f.flush()
                self.stop_flag.wait(5)


def wrapper_cmd(task_args: list[str]) -> list[str]:
    return [os.environ["ROBOTWIN_PYTHON"], str(ROOT / "benchmarks/robotwin/eval_policy_wrapper.py"), "labtasker",
            "--task", "hanging_mug", "--mode", "demo_randomized",
            "--policy-config", str(ROOT / "benchmarks/robotwin/policy_config.yml"), "--render-device", render_device(),
            *task_args]


CHILDREN: list[subprocess.Popen] = []
PIDS_FILE: Path | None = None


def spawn(cmd, env, stream, **kw) -> subprocess.Popen:
    """Start a child in its own process group and record the group id for the executor's cleanup."""
    p = subprocess.Popen(cmd, env=env, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, **kw)
    CHILDREN.append(p)
    if PIDS_FILE is not None:
        with open(PIDS_FILE, "a") as f:
            f.write(f"{p.pid}\n")
    return p


def kill_children(*_):
    for p in CHILDREN:
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.time() + 30
    for p in CHILDREN:
        try:
            p.wait(max(0.1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def run(cmd, env, log: Path) -> int:
    with open(log, "w") as stream:
        return spawn(cmd, env, stream).wait()


def qualify(a) -> int:
    out = a.out
    env = dict(base_env(out), ROBOTWIN_PROBE_ROOT=str(out), ROBOTWIN_SEED_PARAM=str(a.seed_param),
               ROBOTWIN_START_SEED=str(a.start_seed), ROBOTWIN_MAX_CANDIDATES="1")
    rc = run(wrapper_cmd(["--operation", "build_manifest", "--total-episodes", "1", "--result-file",
                          str(out / "manifest.json"), "--progress-file", str(out / "progress.json")]),
             env, out / "qualify.log")
    return rc


def episode(a) -> int:
    out = a.out
    env = base_env(out)
    manifest = json.loads(Path(a.manifest).read_text())
    total = len(manifest["entries"])
    sampler = MemSampler(out / "gpu_mem.csv")
    sampler.start()
    server_log = open(out / "server.log", "w")
    server = spawn([os.environ["OPENWAM_PYTHON"], str(ROOT / "scripts/deploy.py"), "--ckpt-dir", str(CKPT),
                    "--config", str(CONFIG), "--device", "cuda:0", "--host", "127.0.0.1", "--port", str(a.port)],
                   dict(env, OPENWAM_PROBE_ROOT=str(out)), server_log)
    info = {"server_ready": False}
    rc = None
    try:
        from benchmarks.utils.transport import WSPolicyClient
        t0 = time.monotonic()
        while True:
            if server.poll() is not None:
                raise RuntimeError("server exited during startup")
            try:
                with WSPolicyClient(f"ws://127.0.0.1:{a.port}", timeout=30) as c:
                    if c.ping().get("type") == "pong":
                        break
            except Exception:  # noqa: BLE001
                pass
            if time.monotonic() - t0 > 900:
                raise TimeoutError("server readiness timeout")
            time.sleep(5)
        info.update(server_ready=True, server_ready_s=round(time.monotonic() - t0, 1))
        arm_dir = out / a.arm
        arm_dir.mkdir()
        aenv = dict(env, ROBOTWIN_PROBE_ROOT=str(arm_dir), ROBOTWIN_RECOVERY_ARM=a.arm,
                    ROBOTWIN_RUNTIME_ROOT=str(arm_dir / "runtime"))
        rc = run(wrapper_cmd(["--operation", "run_eval", "--host", "127.0.0.1", "--port", str(a.port),
                              "--total-episodes", str(total), "--manifest-file", str(a.manifest),
                              "--manifest-hash", manifest["manifest_hash"], "--episode-start", str(a.episode),
                              "--num-episodes", "1", "--instruction-type", "unseen",
                              "--result-file", str(arm_dir / "results.json"),
                              "--progress-file", str(arm_dir / "progress.json")]), aenv, arm_dir / "eval.log")
    except Exception as e:  # noqa: BLE001
        info["error"] = f"{type(e).__name__}: {e}"
        rc = 1 if rc is None else rc
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
        server_log.close()
        sampler.stop_flag.set()
        sampler.join(30)
        info.update(gpu_samples=sampler.n, min_free_mib=sampler.min_free, max_used_mib=sampler.max_used)
        (out / "episode_info.json").write_text(json.dumps(info, indent=1))
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("qualify"); q.add_argument("--out", type=Path, required=True)
    q.add_argument("--seed-param", type=int, required=True); q.add_argument("--start-seed", type=int, required=True)
    e = sub.add_parser("episode"); e.add_argument("--out", type=Path, required=True)
    e.add_argument("--arm", choices=("B0", "B1", "B2"), required=True)
    e.add_argument("--manifest", type=Path, required=True); e.add_argument("--episode", type=int, required=True)
    e.add_argument("--port", type=int, default=8871)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    global PIDS_FILE
    PIDS_FILE = a.out / "pgids.txt"
    signal.signal(signal.SIGTERM, lambda *x: (kill_children(), sys.exit(143)))
    t0 = time.time()
    rc = qualify(a) if a.cmd == "qualify" else episode(a)
    (a.out / "job.json").write_text(json.dumps({"argv": sys.argv, "rc": rc, "wall_s": round(time.time() - t0, 1),
                                                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}))
    return int(rc != 0)


if __name__ == "__main__":
    raise SystemExit(main())
