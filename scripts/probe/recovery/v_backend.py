"""V: backend validation against the immutable P1A C0 reference (Codex R86 policy, tolerance 0).

Reruns hanging_mug_ep0000_chunk0002 with video_seed 1001 / action_seed 2001 from the same G0 source trace P1A used,
twice, and compares every numeric field with the reference trace (seal 22ab20fe, s3 shard): initial video/action
noise, first-frame latents, proprio, mask, schedule, all step inputs/velocities, final latents and normalized /
physical actions. Bitwise equality and finiteness are required; residuals are always saved.

Only the ordinary full GPU residency is supported (remote 48 GB); the local offload recipe was withdrawn (R87).
The result is bound to --run-id and records host, device, code and input SHAs; every field of the REFERENCE
(not of the candidate) is required. Writes OUT/v_result.json.
Usage: v_backend.py --out OUT --run-id ID --source-trace T --source-trace-sha S --reference R --reference-sha S
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/probe"))
CKPT = ROOT / "assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboTwin-Full"
CONFIG = ROOT / "configs/deploy_probe.yaml"


def sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def compare(name, a, b, out):
    if a is None and b is None:
        out[name] = {"equal": True, "none": True}
        return
    if not (torch.is_tensor(b) or hasattr(b, "shape") or isinstance(b, (list, tuple))):
        out[name] = {"equal": a == b}
        return
    if (a is None) != (b is None):
        out[name] = {"equal": False, "reason": "one side None"}
        return
    ta, tb = torch.as_tensor(a), torch.as_tensor(b)
    if ta.shape != tb.shape or ta.dtype != tb.dtype:
        out[name] = {"equal": False, "reason": f"shape/dtype {tuple(ta.shape)}/{ta.dtype} vs {tuple(tb.shape)}/{tb.dtype}"}
        return
    fa = ta.float() if ta.is_floating_point() else ta
    fb = tb.float() if tb.is_floating_point() else tb
    finite = bool(torch.isfinite(fa).all()) if ta.is_floating_point() else True
    eq = bool(torch.equal(ta, tb))
    diff = (fa - fb).abs() if ta.is_floating_point() else (fa != fb).float()
    out[name] = {"equal": eq, "finite": finite, "max_abs": float(diff.max()) if diff.numel() else 0.0,
                 "n_diff": int((diff != 0).sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source-trace", type=Path, required=True)
    ap.add_argument("--source-trace-sha", required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--reference-sha", required=True)
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    import socket, subprocess
    res = {"format": "recovery-v-result/2", "run_id": a.run_id, "residency": "full", "host": socket.gethostname(),
           "start": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "code_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
           "code_dirty": bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                             capture_output=True, text=True).stdout.strip()),
           "self_sha256": sha_file(__file__), "source_trace_sha256": a.source_trace_sha, "reference_sha256": a.reference_sha}
    for label, p, h in (("source", a.source_trace, a.source_trace_sha), ("reference", a.reference, a.reference_sha)):
        if sha_file(p) != h:
            res.update(engineering_pass=False, error=f"{label} sha mismatch")
            (a.out / "v_result.json").write_text(json.dumps(res, indent=1)); return 2
    import openwam
    res["openwam_import"] = str(Path(openwam.__file__).resolve())
    if not Path(openwam.__file__).resolve().is_relative_to(ROOT):
        res.update(engineering_pass=False, error=f"openwam imported from {openwam.__file__}, not {ROOT}")
        (a.out / "v_result.json").write_text(json.dumps(res, indent=1)); return 2
    from offline_runner import OfflineRunner, load_trace
    src = load_trace(a.source_trace)
    ref = torch.load(a.reference, map_location="cpu", weights_only=False)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    runner = OfflineRunner(str(CKPT), str(CONFIG))
    runner.precompute_prompts([src["cond"]["prompt"]], offload_text_encoder=False)
    res["load_s"] = round(time.time() - t0, 1)
    res["env"] = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0),
                  "capability": list(torch.cuda.get_device_capability(0))}
    runs = []
    for rep in range(2):
        with torch.no_grad():
            out = runner.rerun(src, video_seed=1001, action_seed=2001, record=True)
        torch.cuda.synchronize()
        pt = out["probe_trace"]
        cmp = {}
        # every REQUIRED field comes from the reference; a missing candidate field is a failure
        compare("init.video_noise", pt["init"].get("video_noise"), ref["init"]["video_noise"], cmp)
        compare("init.action_noise", pt["init"].get("action_noise"), ref["init"]["action_noise"], cmp)
        compare("cond.first_frame_latents", pt["cond"].get("first_frame_latents"), ref["cond"]["first_frame_latents"], cmp)
        compare("cond.proprio", pt["cond"].get("proprio"), ref["cond"]["proprio"], cmp)
        compare("active_action_mask", pt.get("active_action_mask"), ref["meta"]["active_action_mask"], cmp)
        sched_ref = [tuple(map(float, x)) for x in ref["meta"]["schedule"]]
        cmp["schedule"] = {"equal": [tuple(map(float, x)) for x in pt["schedule"]] == sched_ref}
        executed = [i for i in range(len(sched_ref) - 1) if sched_ref[i] != sched_ref[i + 1]]
        ref_idx = [int(r["i"]) for r in ref["steps"]]
        cand_idx = [int(x["i"]) for x in pt["steps"]]
        cmp["steps.index_set"] = {"equal": ref_idx == cand_idx == executed and len(set(cand_idx)) == len(cand_idx),
                                  "ref": ref_idx, "cand": cand_idx, "executed": executed}
        by_i = {int(x["i"]): x for x in pt["steps"]}
        for r in ref["steps"]:
            i = int(r["i"]); s_ = by_i.get(i, {})
            for k in r:
                if k == "i":
                    continue
                if k in ("t_v", "t_a"):
                    cmp[f"step{i}.{k}"] = {"equal": k in s_ and float(s_[k]) == float(r[k])}
                else:
                    compare(f"step{i}.{k}", s_.get(k), r[k], cmp)
        for k in ref["final"]:
            compare(f"final.{k}", pt["final"].get(k), ref["final"][k], cmp)
        ok = all(v.get("equal") and v.get("finite", True) for v in cmp.values())
        runs.append({"rep": rep, "bitwise_equal_all": ok, "n_fields": len(cmp),
                     "failed_fields": sorted(k for k, v in cmp.items() if not (v.get("equal") and v.get("finite", True))),
                     "fields": cmp})
    res["runs"] = runs
    res["peak_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    res["peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
    res["engineering_pass"] = all(r["bitwise_equal_all"] for r in runs) and not res["code_dirty"]
    res["end"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (a.out / "v_result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"engineering_pass": res["engineering_pass"], "peak_reserved_mib": res["peak_reserved_mib"],
                      "failed": [r["failed_fields"][:5] for r in runs]}))
    return 0 if res["engineering_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
