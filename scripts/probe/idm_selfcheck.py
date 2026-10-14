"""K6/K7 IDM runtime self-check on the G2 prereg verification observations.

Implementer smoke test only (Codex V2 is the independent acceptance). Reports bit-exactness
checks and shapes; computes no Q6 effect metric (no action-vs-command errors, no swap
magnitudes beyond a nonzero flag).

    CUDA_VISIBLE_DEVICES=2 python scripts/probe/idm_selfcheck.py \
        --ckpt-dir assets/openwam_ckpt/openwam_study/robotwin_dual_system_idm \
        --prereg outputs/probe/g2_protocol_20261014/G2_PREREG.json --out outputs/probe/g2_idm_selfcheck_<date>
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from offline_runner import OfflineRunner, sha256_file  # noqa: E402


def same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    return np.array_equal(np.asarray(a), np.asarray(b))


def same_steps(x, y, keys) -> bool:
    return len(x) == len(y) and all(int(s["i"]) == int(t["i"]) and all(same(s[k], t[k]) for k in keys) for s, t in zip(x, y))


def raises(fn) -> bool:
    try:
        fn()
    except ValueError:
        return True
    return False


def check_obs(runner, v, out_dir):
    rec_path = Path(v["record"])
    rec = json.loads(rec_path.read_text())
    if sha256_file(rec_path) != v["record_sha256"]:
        raise ValueError("record sha mismatch: " + str(rec_path))
    (vr, ar), (vd, _) = v["receiver"], v["donor"]
    a2 = [a for a in v["stage1_independence_action_seeds"] if a != ar][0]
    obs = runner.canvas_from_pngs(rec["pngs"], prompt=rec["prompt"], state=rec.get("proprio"))
    trace_path = out_dir / f"{v['obs_id']}_v{vr}_a{ar}.pt"
    conditions = {"first_frame_image": [obs["image"]], "prompt": obs["prompt"], "proprio": obs.get("state"),
                  "probe": {"trace_path": str(trace_path), "video_seed": vr, "action_seed": ar}}
    online = runner.engine.generate(conditions)
    trace = torch.load(trace_path, map_location="cpu", weights_only=False)
    c = {}

    r1 = runner.rerun(trace)["probe_trace"]
    c["online_offline_actions_equal"] = same(online["actions"], r1["final"]["actions_physical"])
    c["online_offline_stage2_steps_equal"] = same_steps(trace["steps"], r1["steps"], ("t_a", "action_latents_in", "action_velocity"))
    r1b = runner.rerun(trace)["probe_trace"]
    c["repeat_stage1_equal"] = same_steps(r1["stage1_steps"], r1b["stage1_steps"], ("video_latents_in", "video_velocity"))
    c["repeat_actions_equal"] = same(r1["final"]["actions_physical"], r1b["final"]["actions_physical"])

    ra2 = runner.rerun(trace, action_seed=a2)["probe_trace"]
    c["stage1_independent_of_action_seed"] = (
        same_steps(r1["stage1_steps"], ra2["stage1_steps"], ("t_v", "video_latents_in", "video_velocity"))
        and same(r1["final"]["stage1_future_latents"], ra2["final"]["stage1_future_latents"]))
    c["action_seed_changes_action_noise"] = not same(r1["init"]["action_noise"], ra2["init"]["action_noise"])

    own = r1["final"]["stage1_future_latents"]
    ffl = r1["cond"]["first_frame_latents"]
    nf = ffl.shape[2]
    rd = runner.rerun(trace, video_seed=vd)["probe_trace"]
    donor = rd["final"]["stage1_future_latents"]
    rsw = runner.rerun(trace, cond_future_override=donor)["probe_trace"]
    used = rsw["final"]["cond_video_latents_used"]
    c["donor_future_differs"] = not same(own, donor)
    c["swap_keeps_receiver_stage1"] = same(rsw["final"]["stage1_future_latents"], own)
    c["swap_future_is_donor"] = same(used[:, :, nf:], donor[:, :, nf:])
    c["swap_first_frame_is_receiver"] = same(used[:, :, :nf], ffl.to(used.dtype))
    c["swap_changes_actions_nonzero"] = not same(rsw["final"]["actions_physical"], r1["final"]["actions_physical"])

    # same-source identity AFTER a different condition ran: fresh KV / no stale state
    rid = runner.rerun(trace, cond_future_override=own)["probe_trace"]
    c["identity_override_actions_equal"] = same(rid["final"]["actions_physical"], r1["final"]["actions_physical"])
    c["identity_override_stage2_steps_equal"] = same_steps(r1["steps"], rid["steps"], ("action_latents_in", "action_velocity"))

    rnd = torch.randn(own.shape, generator=torch.Generator().manual_seed(0)).to(own.dtype)
    rr = runner.rerun(trace, cond_future_override=rnd)["probe_trace"]
    c["random_override_first_frame_protected"] = same(rr["final"]["cond_video_latents_used"][:, :, :nf], ffl.to(own.dtype))
    c["dtype_mismatch_raises"] = raises(lambda: runner.rerun(trace, cond_future_override=own.float()))
    c["shape_mismatch_raises"] = raises(lambda: runner.rerun(trace, cond_future_override=own[:, :, :-1]))

    frames = runner.gt_future_frames(rec)
    c["gt_frame0_equals_trace_image"] = same(np.asarray(frames[0].convert("RGB")), trace["cond"]["image"])
    gt = runner.encode_frames(frames).cpu()
    c["gt_latent_shape_dtype_match"] = gt.shape == own.shape and gt.dtype == own.dtype
    ff_diff = float((gt[:, :, :nf].float() - ffl.float()).abs().max())
    c["gt_first_latent_equals_receiver"] = ff_diff == 0.0
    rgt = runner.rerun(trace, cond_future_override=gt)["probe_trace"]
    c["gt_override_first_frame_is_receiver"] = same(rgt["final"]["cond_video_latents_used"][:, :, :nf], ffl.to(own.dtype))
    c["gt_override_future_is_gt"] = same(rgt["final"]["cond_video_latents_used"][:, :, nf:], gt[:, :, nf:])

    # the trace keeps proprio after the model's dtype cast, so compare against the same cast
    expected_proprio = runner.arch.normalize_deploy_proprio(rec.get("proprio")).reshape(-1)
    traced_proprio = trace["cond"]["proprio"].reshape(-1)
    c["proprio_is_idm_normalized_record_state"] = same(traced_proprio, expected_proprio.to(traced_proprio.dtype))
    proprio_cast_err = float((traced_proprio.float() - expected_proprio).abs().max())
    info = {"obs_id": v["obs_id"], "checks": c,
            "gt_first_latent_max_abs_diff": ff_diff, "proprio_bf16_cast_max_abs_err": proprio_cast_err,
            "shapes": {"video_latents": list(own.shape), "first_frame_latent_frames": int(nf),
                       "idm_actions": list(np.asarray(r1["final"]["actions_physical"]).shape),
                       "record_actions": list(np.asarray(rec["actions"]).shape)},
            "dtype": str(own.dtype), "schedule": trace["meta"]["schedule"],
            "num_frames": trace["meta"]["num_frames"], "action_num_frames": trace["meta"]["action_num_frames"],
            "active_action_mask": trace["meta"]["active_action_mask"].tolist(),
            "active_action_mask_source": trace["meta"]["active_action_mask_source"],
            "trace": {"path": str(trace_path), "sha256": sha256_file(trace_path)}}
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to reuse output dir " + str(out))
    out.mkdir(parents=True)
    prereg = json.loads(Path(args.prereg).read_text())
    by_id = {o["obs_id"]: o for o in prereg["observations"]}
    ver = [{**by_id[v["obs_id"]], **v} for v in prereg["verification"]]
    t0 = time.time()
    runner = OfflineRunner(args.ckpt_dir, args.config)
    runner.precompute_prompts([json.loads(Path(v["record"]).read_text())["prompt"] for v in ver])
    results = [check_obs(runner, v, out) for v in ver]
    summary = {"scope": "K6/K7 implementer self-check; not independent acceptance; no Q6 effect metrics",
               "ckpt_dir": args.ckpt_dir, "prereg_sha256": sha256_file(args.prereg),
               "all_passed": all(all(r["checks"].values()) for r in results),
               "failed": {r["obs_id"]: [k for k, ok in r["checks"].items() if not ok] for r in results},
               "results": results, "seconds": round(time.time() - t0, 1),
               "code_sha256": {n: sha256_file(Path(__file__).resolve().parent / n)
                               for n in ("idm_selfcheck.py", "offline_runner.py")},
               "idm_py_sha256": sha256_file(Path(__file__).resolve().parents[2] / "openwam/model/architectures/dual_system/idm.py")}
    tmp = out / "summary.json.tmp"
    tmp.write_text(json.dumps(summary, indent=1, default=str))
    os.replace(tmp, out / "summary.json")
    print(json.dumps({"all_passed": summary["all_passed"], "failed": summary["failed"], "seconds": summary["seconds"]}))


if __name__ == "__main__":
    main()
