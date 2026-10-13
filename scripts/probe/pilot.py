"""G0 only: serial expert manifests and rollouts, confined to physical GPU3."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.utils.transport import WSPolicyClient
from benchmarks.robotwin.probe_capture import atomic_json


def run_logged(command, env, path):
    with path.open("w") as stream:
        subprocess.run(command, env=env, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def inventory(root):
    rows, errors = [], []
    for path in sorted(root.glob("*/ep*/chunk*/record.json")):
        d = json.loads(path.read_text())
        trace = Path(d["probe_receipt"]["trace_path"])
        if not trace.is_file():
            errors.append(f"missing trace {trace}")
        elif hashlib.sha256(trace.read_bytes()).hexdigest() != d["probe_receipt"]["sha256"]:
            errors.append(f"trace hash mismatch {trace}")
        for entry in [d, *d["post_obs"]]:
            for image in entry["pngs"].values():
                png = Path(image["path"])
                if not png.is_file() or hashlib.sha256(png.read_bytes()).hexdigest() != image["sha256"]:
                    errors.append(f"PNG missing or hash mismatch {png}")
        if not d["complete"]:
            errors.append(f"incomplete record {path}")
        count = d["executed_count"]
        if count != sum(a["count_delta"] for a in d["actions_executed"]):
            errors.append(f"execution count mismatch {path}")
        actual = {p["offset"] for p in d["post_obs"]}
        expected = set(range(4, count + 1, 4))
        if actual != expected:
            errors.append(f"post observation alignment mismatch {path}: {actual} != {expected}")
        if d["termination"] == "none" and count != d["inference_horizon"]:
            errors.append(f"nonterminal short chunk {path}")
        rows.append({"record": str(path), "task": d["task"], "episode": d["episode"],
                     "chunk": d["chunk"], "executed_count": count, "termination": d["termination"],
                     "trace_bytes": trace.stat().st_size if trace.exists() else None})
    return {"chunks": rows, "errors": errors, "passed": bool(rows) and not errors}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--tasks", nargs="+", default=["place_empty_cup", "hanging_mug", "turn_switch"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--port", type=int, default=8853)
    parser.add_argument("--step-limit", type=int, help="smoke only, e.g.64 for two chunks")
    parser.add_argument("--existing-server", action="store_true")
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    if args.inventory_only:
        result = inventory(args.run.resolve())
        atomic_json(args.run / "inventory.json", result)
        return 0 if result["passed"] else 1
    if args.episodes <= 0 or any(t not in ("place_empty_cup", "hanging_mug", "turn_switch") for t in args.tasks):
        parser.error("G0 supports only the three approved tasks and positive episode counts")
    run = args.run.resolve()
    if not run.is_relative_to((ROOT / "outputs/probe").resolve()):
        parser.error("run must be below outputs/probe")
    run.mkdir(parents=True, exist_ok=False)
    robot_python = os.environ["ROBOTWIN_PYTHON"]
    openwam_python = os.environ["OPENWAM_PYTHON"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="3", ROBOTWIN_PROBE_ROOT=str(run),
               ROBOTWIN_PATH=os.environ["ROBOTWIN_PATH"],
               PYTHONPATH=f"{ROOT}:{ROOT / 'benchmarks/robotwin'}:{os.environ.get('PYTHONPATH', '')}",
               PYTHONUNBUFFERED="1")
    if args.step_limit is not None:
        if args.step_limit <= 0:
            parser.error("step-limit must be positive")
        env["ROBOTWIN_PROBE_STEP_LIMIT"] = str(args.step_limit)
    egl = Path(robot_python).parent.parent / "lib/python3.10/site-packages/sapien/vulkan_library/10_nvidia.json"
    if egl.exists():
        env.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", str(egl))
    gpu = subprocess.check_output(["nvidia-smi", "--id=3", "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"], text=True).strip()
    if int(gpu.rsplit(",", 1)[1]) < 26624:
        raise RuntimeError(f"GPU3 free memory below26GiB: {gpu}")
    pci = subprocess.check_output(["nvidia-smi", "--id=3", "--query-gpu=pci.bus_id", "--format=csv,noheader"], text=True).strip()
    render_device = "pci:" + pci[-12:]
    config = ROOT / "configs/deploy_probe.yaml"
    if not config.exists():
        raise FileNotFoundError("Claude M5 config not yet available")
    import yaml
    deploy = yaml.safe_load(config.read_text())
    if deploy["optimization"]["dit_cache"]["enabled"] or deploy["optimization"]["compile"]["enabled"]:
        raise ValueError("G0 requires cache and compile disabled")
    atomic_json(run / "run.json", {"gpu": gpu, "gpu_physical_index": 3, "episodes": args.episodes,
        "tasks": args.tasks, "step_limit": args.step_limit,
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})
    server = None
    server_stream = None
    try:
        if not args.existing_server:
            server_stream = (run / "server.log").open("w")
            server = subprocess.Popen([openwam_python, str(ROOT / "scripts/deploy.py"),
                "--ckpt-dir", str(ROOT / "assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboTwin-Full"),
                "--config", str(config), "--device", "cuda:0", "--host", "127.0.0.1", "--port", str(args.port)],
                env=env, cwd=ROOT, stdout=server_stream, stderr=subprocess.STDOUT, start_new_session=True)
        with WSPolicyClient(f"ws://127.0.0.1:{args.port}", timeout=600) as client:
            deadline = time.monotonic() + 600
            while True:
                if server is not None and server.poll() is not None:
                    raise RuntimeError("probe server exited; see server.log")
                try:
                    pong = client.ping()
                    if pong.get("type") == "pong":
                        break
                except Exception:
                    client.close()
                if time.monotonic() >= deadline:
                    raise TimeoutError("probe server readiness timeout")
                time.sleep(2)
        for task in args.tasks:
            directory = run / task
            directory.mkdir(exist_ok=True)
            task_env = dict(env, ROBOTWIN_RUNTIME_ROOT=str(directory / "runtime"))
            base = [robot_python, str(ROOT / "benchmarks/robotwin/eval_policy_wrapper.py"), "labtasker",
                "--task", task, "--mode", "demo_randomized", "--policy-config", str(ROOT / "benchmarks/robotwin/policy_config.yml"),
                "--total-episodes", str(args.episodes), "--render-device", render_device]
            manifest = directory / "manifest.json"
            run_logged(base + ["--operation", "build_manifest", "--result-file", str(manifest),
                "--progress-file", str(directory / "manifest_progress.json")], task_env, directory / "manifest.log")
            digest = json.loads(manifest.read_text())["manifest_hash"]
            run_logged(base + ["--operation", "run_eval", "--host", "127.0.0.1", "--port", str(args.port),
                "--manifest-file", str(manifest), "--manifest-hash", digest, "--episode-start", "0",
                "--num-episodes", str(args.episodes), "--instruction-type", "unseen",
                "--result-file", str(directory / "results.json"), "--progress-file", str(directory / "eval_progress.json")],
                task_env, directory / "eval.log")
        result = inventory(run)
        result["requested_episodes"] = len(args.tasks) * args.episodes
        result["recorded_episodes"] = len({(r["task"], r["episode"]) for r in result["chunks"]})
        result["passed"] &= result["recorded_episodes"] == result["requested_episodes"]
        atomic_json(run / "inventory.json", result)
        return 0 if result["passed"] else 1
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
        if server_stream is not None:
            server_stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
