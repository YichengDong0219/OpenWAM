"""V5 environment check: compare a capture-OFF rollout to an exact captured manifest entry.

Use an already-ready, otherwise idle server. Never run alongside another client:
resetting the server mid-rollout would invalidate both runs. The original client
debug logger supplies capture-OFF evidence without enabling G0 probe hooks.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import hashlib
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.robotwin.probe_capture import atomic_json


def compare(captured, debug_dir):
    records = [json.loads(p.read_text()) for p in sorted(captured.glob("chunk*/record.json"))]
    metas = [(p, json.loads(p.read_text())) for p in sorted(debug_dir.glob("ep*/step*/meta.json"))]
    if not records or not metas:
        raise ValueError("missing capture-ON or capture-OFF records")
    on_actions = [action for r in records for action in r["actions"][:r["executed_count"]]]
    off_actions = [r["action"] for _, r in metas]
    errors = []
    if len(on_actions) != len(off_actions):
        errors.append(f"rollout lengths differ: ON={len(on_actions)}, OFF={len(off_actions)}")
    count = min(len(on_actions), len(off_actions))
    diff = float(np.max(np.abs(np.asarray(on_actions[:count]) - np.asarray(off_actions[:count]))))
    if diff != 0:
        errors.append(f"action max_abs={diff}")
    images_checked = 0
    states_max = 0.
    for r in records:
        rows = [{"offset": 0, "pngs": r["pngs"], "proprio": r["proprio"]}, *r["post_obs"]]
        for entry in rows:
            index = r["take_action_cnt"] + entry["offset"]
            if index >= len(metas):
                continue  # final state has no next policy request in capture-OFF debug data
            meta_path, off = metas[index]
            if off["prompt"] != r["prompt"]:
                errors.append(f"prompt mismatch at step{index}")
            if entry["offset"] == 0 and entry.get("proprio") is not None:
                states_max = max(states_max, float(np.max(np.abs(np.asarray(off["state"]) - entry["proprio"]))))
            for camera in ("head", "left", "right"):
                image = meta_path.parent / f"{camera}.png"
                if hashlib.sha256(image.read_bytes()).hexdigest() != entry["pngs"][camera]["sha256"]:
                    errors.append(f"image mismatch at step{index}/{camera}")
                images_checked += 1
    if states_max != 0:
        errors.append(f"chunk-start proprio difference {states_max}")
    return {"passed": not errors, "errors": errors, "actions_on": len(on_actions), "actions_off": len(off_actions),
            "action_max_abs": diff, "chunk_start_proprio_max_abs": states_max, "images_checked": images_checked}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pilot-run", required=True, type=Path)
    p.add_argument("--task", default="place_empty_cup")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--port", type=int, default=8853)
    p.add_argument("--compare-only", action="store_true")
    args = p.parse_args()
    out = args.output_dir.resolve()
    if args.compare_only:
        result = compare(args.pilot_run / args.task / f"ep{args.episode:04d}", out / "debug")
        atomic_json(out / "client_capture_comparison.json", result)
        return 0 if result["passed"] else 1
    if not out.is_relative_to((ROOT / "outputs/probe").resolve()):
        p.error("output-dir must be below outputs/probe")
    out.mkdir(parents=True, exist_ok=False)
    manifest_path = (args.pilot_run / args.task / "manifest.json").resolve()
    manifest = json.loads(manifest_path.read_text())
    policy = yaml.safe_load((ROOT / "benchmarks/robotwin/policy_config.yml").read_text())
    policy.update(debug=True, debug_dir=str(out / "debug"))
    policy.pop("probe_dir", None)
    policy_path = out / "policy_off.yml"
    policy_path.write_text(yaml.safe_dump(policy))
    python = os.environ["ROBOTWIN_PYTHON"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="3", ROBOTWIN_RUNTIME_ROOT=str(out / "runtime"),
        PYTHONPATH=f"{ROOT}:{ROOT / 'benchmarks/robotwin'}:{os.environ.get('PYTHONPATH', '')}",
        PATH=f"{Path(python).parent}:{Path(python).parent.parent.parent.parent / 'bin'}:/usr/local/cuda-12.4/bin:{os.environ.get('PATH', '')}",
        PYTHONUNBUFFERED="1")
    for key in ("ROBOTWIN_PROBE_ROOT", "ROBOTWIN_PROBE_STEP_LIMIT", "ROBOTWIN_ENABLE_PLANNER_FALLBACK"):
        env.pop(key, None)
    egl = Path(python).parent.parent / "lib/python3.10/site-packages/sapien/vulkan_library/10_nvidia.json"
    if egl.exists():
        env.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", str(egl))
    bus = subprocess.check_output(["nvidia-smi", "--id=3", "--query-gpu=pci.bus_id", "--format=csv,noheader"], text=True).strip()
    command = [python, str(ROOT / "benchmarks/robotwin/eval_policy_wrapper.py"), "labtasker", "--task", args.task,
        "--mode", "demo_randomized", "--policy-config", str(policy_path), "--total-episodes", str(len(manifest["entries"])),
        "--render-device", "pci:" + bus[-12:], "--operation", "run_eval", "--host", "127.0.0.1", "--port", str(args.port),
        "--manifest-file", str(manifest_path), "--manifest-hash", manifest["manifest_hash"],
        "--episode-start", str(args.episode), "--num-episodes", "1", "--instruction-type", "unseen",
        "--result-file", str(out / "results.json"), "--progress-file", str(out / "progress.json")]
    with (out / "eval_off.log").open("w") as stream:
        subprocess.run(command, env=env, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    result = compare(args.pilot_run / args.task / f"ep{args.episode:04d}", out / "debug")
    atomic_json(out / "client_capture_comparison.json", result)
    print(json.dumps(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
