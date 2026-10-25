"""V: backend validation against the immutable P1A C0 reference (Codex R86 policy, tolerance 0).

Reruns hanging_mug_ep0000_chunk0002 with video_seed 1001 / action_seed 2001 from the same G0 source trace P1A used,
twice, and compares every numeric field with the reference trace (seal 22ab20fe, s3 shard): initial video/action
noise, first-frame latents, proprio, mask, schedule, all step inputs/velocities, final latents and normalized /
physical actions. Bitwise equality and finiteness are required; residuals are always saved.

--residency full     : ordinary load on the GPU (remote 48 GB);
--residency offload  : load on CPU, move everything except the text encoder to the GPU, encode the prompt on the
                       GPU with the DiT parked on CPU (OfflineRunner.precompute_prompts), then park the text encoder
                       on CPU. Residency only; no sampler/model logic changes.
Writes OUT/v_result.json. Usage: v_backend.py --out OUT --source-trace T --reference R --reference-sha S --residency M
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


def offload_load(device: str):
    from offline_runner import OfflineRunner
    runner = OfflineRunner(str(CKPT), str(CONFIG), device="cpu")
    arch = runner.arch
    dev = torch.device(device)
    arch._device = dev
    if getattr(arch, "proprio_encoder", None) is not None:
        arch.proprio_encoder.to(dev)
    for bb in arch.backbones.values():
        bb._device = dev
        for name, mod in bb.named_children():
            if name != "text_encoder":
                mod.to(dev)
    vb = arch.video_backbone
    vb.dit.to("cpu"); torch.cuda.empty_cache()
    vb.text_encoder.to(dev)
    vb.dit.to(dev)  # precompute_prompts parks the DiT itself while encoding
    return runner


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source-trace", type=Path, required=True)
    ap.add_argument("--source-trace-sha", required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--reference-sha", required=True)
    ap.add_argument("--residency", choices=("full", "offload"), required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    res = {"format": "recovery-v-result/1", "residency": a.residency, "start": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    for label, p, h in (("source", a.source_trace, a.source_trace_sha), ("reference", a.reference, a.reference_sha)):
        if sha_file(p) != h:
            res.update(engineering_pass=False, error=f"{label} sha mismatch")
            (a.out / "v_result.json").write_text(json.dumps(res, indent=1)); return 2
    from offline_runner import OfflineRunner, load_trace
    src = load_trace(a.source_trace)
    ref = torch.load(a.reference, map_location="cpu", weights_only=False)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    runner = offload_load("cuda") if a.residency == "offload" else OfflineRunner(str(CKPT), str(CONFIG))
    runner.precompute_prompts([src["cond"]["prompt"]], offload_text_encoder=(a.residency == "offload"))
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
        compare("init.video_noise", pt["init"]["video_noise"], ref["init"]["video_noise"], cmp)
        compare("init.action_noise", pt["init"]["action_noise"], ref["init"]["action_noise"], cmp)
        compare("cond.first_frame_latents", pt["cond"]["first_frame_latents"], ref["cond"]["first_frame_latents"], cmp)
        compare("cond.proprio", pt["cond"]["proprio"], ref["cond"]["proprio"], cmp)
        compare("active_action_mask", pt["active_action_mask"], ref["meta"]["active_action_mask"] if "meta" in ref else ref["active_action_mask"], cmp)
        sched_ref = [tuple(map(float, x)) for x in (ref["meta"]["schedule"] if "meta" in ref else ref["schedule"])]
        cmp["schedule"] = {"equal": [tuple(map(float, x)) for x in pt["schedule"]] == sched_ref}
        if len(pt["steps"]) != len(ref["steps"]):
            cmp["steps"] = {"equal": False, "reason": f"{len(pt['steps'])} vs {len(ref['steps'])} steps"}
        else:
            for s, r in zip(pt["steps"], ref["steps"]):
                for k in s:
                    if k in ("i", "t_v", "t_a"):
                        cmp[f"step{s['i']}.{k}"] = {"equal": s[k] == r.get(k)}
                    elif torch.is_tensor(s[k]) or k in r:
                        compare(f"step{s['i']}.{k}", s[k], r.get(k), cmp)
        for k in ("video_latents", "action_latents", "actions_normalized", "actions_physical"):
            compare(f"final.{k}", pt["final"][k], ref["final"][k], cmp)
        ok = all(v.get("equal") and v.get("finite", True) for v in cmp.values())
        runs.append({"rep": rep, "bitwise_equal_all": ok, "n_fields": len(cmp),
                     "failed_fields": sorted(k for k, v in cmp.items() if not (v.get("equal") and v.get("finite", True))),
                     "fields": cmp})
    res["runs"] = runs
    res["peak_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    res["peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
    res["engineering_pass"] = all(r["bitwise_equal_all"] for r in runs)
    res["end"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (a.out / "v_result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"engineering_pass": res["engineering_pass"], "peak_reserved_mib": res["peak_reserved_mib"],
                      "failed": [r["failed_fields"][:5] for r in runs]}))
    return 0 if res["engineering_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
