"""Fake GPU jobs for the end-to-end synthetic pipeline test (CPU only, production-schema outputs).

  v        --out D --run-id R                      -> D/v_result.json (two bitwise-equal runs)
  qualify  --out D --start-seed S                  -> official-looking seed_audit (+ manifest when accepted)
  episode  --out D --arm A --manifest M --expected-manifest-sha H --episode I --seed S [--fail-rc N]
           -> records / recovery events / results and full-schema R + candidate traces at triggers
Acceptance rule for qualification: seed % 3 != 0 (reason expert_plan_fail otherwise). Triggers: ordinal 0 only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE / "trace_plan.json").read_text())
SEEDS = {0: {"ref": (1001, 2001), "cand": (3001, 4001)}}


def rnd(sig, seed):
    g = torch.Generator().manual_seed(int(seed))
    return torch.randn(*sig["shape"], generator=g).to(getattr(torch, sig["dtype"].split(".")[-1]))


def trace(path: Path, vseed, aseed, ff, prop, target=None, clamp_ref_sha=None):
    eps = rnd(PLAN["init"]["video_noise"], vseed)
    act = rnd(PLAN["init"]["action_noise"], aseed)
    steps = []
    for i in PLAN["executed_steps"]:
        tv, ta = PLAN["schedule"][i]
        if target is None:
            z = rnd(PLAN["step_tensors"]["video_latents_in"], vseed * 100 + i)
        else:
            s = tv / PLAN["num_train_timesteps"]
            z = eps.clone()
            z[:, :, 1:] = ((1 - s) * target[:, :, 1:].float() + s * eps[:, :, 1:].float()).to(eps.dtype)
        z[:, :, :1] = ff
        steps.append({"i": i, "t_v": tv, "t_a": ta, "sigma_v": tv / 1000, "sigma_a": ta / 1000, "video_stepping": True,
                      "action_stepping": True, "cache_hit": False, "video_latents_in": z,
                      "action_latents_in": rnd(PLAN["step_tensors"]["action_latents_in"], aseed * 100 + i),
                      "video_velocity": rnd(PLAN["step_tensors"]["video_velocity"], vseed * 1000 + i),
                      "action_velocity": rnd(PLAN["step_tensors"]["action_velocity"], aseed * 1000 + i)})
    final = target.clone() if target is not None else rnd(PLAN["final_tensors"]["video_latents"], vseed + 7)
    final[:, :, :1] = ff
    torch.save({"meta": {"schedule": [tuple(x) for x in PLAN["schedule"]], "num_frames": PLAN["num_frames"],
                         "action_num_frames": PLAN["action_num_frames"],
                         "active_action_mask": torch.tensor(PLAN["active_action_mask"]), "seed": 42,
                         "video_seed": vseed, "action_seed": aseed, "replay_mode": None,
                         "clamp_source_path": "R" if clamp_ref_sha else None, "clamp_source_sha256": clamp_ref_sha},
                "cond": {"first_frame_latents": ff, "proprio": prop}, "init": {"video_noise": eps, "action_noise": act},
                "steps": steps,
                "final": {"video_latents": final, "action_latents": rnd(PLAN["final_tensors"]["action_latents"], aseed + 7),
                          "actions_normalized": np.zeros(PLAN["final_arrays"]["actions_normalized"], np.float32),
                          "actions_physical": np.zeros(PLAN["final_arrays"]["actions_physical"], np.float32)}}, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind"); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run-id"); ap.add_argument("--start-seed", type=int); ap.add_argument("--arm")
    ap.add_argument("--manifest"); ap.add_argument("--expected-manifest-sha"); ap.add_argument("--episode", type=int)
    ap.add_argument("--seed", type=int); ap.add_argument("--fail-rc", type=int, default=0)
    a, _ = ap.parse_known_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.kind == "v":
        (a.out / "v_result.json").write_text(json.dumps({
            "run_id": a.run_id, "reference_sha256": "8e124efdf1dc1210c8bdbd81cec2678015d19bff5168c211aa1a82d57780aa38",
            "source_trace_sha256": "b101730f60261707ba6b5c95b8dddf9fa71f1d81d06bd091de08ff31620e2e20",
            "code_dirty": False, "engineering_pass": True, "runs": [{"bitwise_equal_all": True}] * 2}))
        return 0
    if a.kind == "qualify":
        acc = a.start_seed % 3 != 0
        d = a.out / "hanging_mug"; d.mkdir(parents=True)
        (d / "seed_audit.jsonl").write_text(json.dumps({"seed": a.start_seed, "accepted": acc,
                                                        "reason": "accepted" if acc else "expert_plan_fail"}) + "\n")
        if acc:
            (a.out / "manifest.json").write_text(json.dumps({"benchmark": "robotwin", "task": "hanging_mug",
                "mode": "demo_randomized", "entries": [{"episode": 0, "seed": a.start_seed,
                "instructions": {"unseen": [f"hang the mug {a.start_seed}"]}}], "manifest_hash": "sha256:x"}))
            return 0
        return 1
    # episode
    raw = Path(a.manifest).read_bytes()
    if hashlib.sha256(raw).hexdigest() != a.expected_manifest_sha:
        return 3
    if json.loads(raw)["entries"][a.episode]["seed"] != a.seed:
        return 3
    if a.fail_rc or os.environ.get("FAKE_EPISODE_FAIL_RC"):
        return a.fail_rc or int(os.environ["FAKE_EPISODE_FAIL_RC"])
    arm, o = a.arm, a.episode
    d = a.out / arm / "hanging_mug" / f"ep{o:04d}"
    d.mkdir(parents=True)
    trig = 2 if o == 0 else None
    n = 4
    ff = rnd(PLAN["cond"]["first_frame_latents"], 50 + o)
    prop = rnd(PLAN["cond"]["proprio"], 60 + o)
    events = []
    for c in range(n):
        cd = d / f"chunk{c:04d}"; cd.mkdir()
        diverged = trig is not None and c > trig and arm != "B0"
        rec = {"chunk": c, "env_seed": a.seed, "actions": [[c, {"B0": 0.0, "B1": 1.0, "B2": 2.0}[arm] if (trig is not None and c >= trig) else 0.0]],
               "task_fields": {"x": c + (100 if diverged else 0)}, "proprio": [o], "endpose": [c],
               "pngs": {"head": {"sha256": f"h{o}{c}{'d' if diverged else ''}"}},
               "post_obs": [{"offset": k, "task_fields": {"x": c + (100 if diverged else 0)}} for k in range(4, 33, 4)],
               "complete": True, "termination": "none"}
        if trig is not None and c >= trig and arm == "B0":
            rec["actions"] = [[c, 0.0]]
        ev = {"chunk": c, "arm": arm, "eligible_trigger": c == trig, "trigger": c == trig}
        if c == trig:
            ev["correction_index"] = 0
            if arm != "B0":
                rsha = trace(cd.with_name(cd.name + "_ref.pt"), *SEEDS[0]["ref"], ff, prop)
                tgt = torch.load(cd.with_name(cd.name + "_ref.pt"), weights_only=False)["final"]["video_latents"]
                csha = trace(cd.with_suffix(".pt"), *SEEDS[0]["cand"], ff, prop,
                             target=tgt if arm == "B2" else None, clamp_ref_sha=rsha if arm == "B2" else None)
                rec["recovery_receipt"] = {"arm": arm, "n_generations": 2,
                                           "ref": {"trace_path": str(cd.with_name(cd.name + "_ref.pt")), "sha256": rsha},
                                           "cand": {"trace_path": str(cd.with_suffix(".pt")), "sha256": csha}}
        events.append(ev)
        if c == n - 1:
            rec.update(termination="step_lim", terminal_task_fields={"x": 9}, terminal_eval_success=False)
        (cd / "record.json").write_text(json.dumps(rec))
    (d / "recovery_events.json").write_text(json.dumps({"arm": arm, "events": events}))
    (a.out / arm / "results.json").write_text(json.dumps({"episodes": [{"episode": o, "seed": a.seed, "success": False}]}))
    (a.out / "episode_info.json").write_text(json.dumps({"server_ready": True, "min_free_mib": 30000}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
