"""Derive the frozen trace plan (schedule, executed forwards, mask, required keys, shapes/dtypes) from the immutable
P1A C0 reference trace (sha 8e124efd). Validators check every reference/candidate trace against this plan, never
against what the candidate trace itself declares. Usage: make_trace_plan.py REFERENCE.pt OUT.json"""
import hashlib, json, sys
import torch
ref_path, out = sys.argv[1], sys.argv[2]
raw = open(ref_path, "rb").read()
assert hashlib.sha256(raw).hexdigest() == "8e124efdf1dc1210c8bdbd81cec2678015d19bff5168c211aa1a82d57780aa38"
r = torch.load(ref_path, map_location="cpu", weights_only=False)
sig = lambda t: {"shape": list(t.shape), "dtype": str(t.dtype)}
sched = [[float(a), float(b)] for a, b in r["meta"]["schedule"]]
executed = [i for i in range(len(sched) - 1) if sched[i] != sched[i + 1]]
step0 = r["steps"][0]
plan = {"format": "recovery-trace-plan/1", "source_reference_sha256": hashlib.sha256(raw).hexdigest(),
        "num_train_timesteps": 1000.0, "schedule": sched, "executed_steps": executed,
        "num_frames": r["meta"]["num_frames"], "action_num_frames": r["meta"]["action_num_frames"],
        "active_action_mask": [bool(x) for x in r["meta"]["active_action_mask"].tolist()],
        "cond": {k: sig(r["cond"][k]) for k in ("first_frame_latents", "proprio")},
        "init": {k: sig(r["init"][k]) for k in ("video_noise", "action_noise")},
        "step_scalar_keys": sorted(k for k, v in step0.items() if not torch.is_tensor(v)),
        "step_tensors": {k: sig(v) for k, v in step0.items() if torch.is_tensor(v)},
        "final_tensors": {k: sig(r["final"][k]) for k in ("video_latents", "action_latents")},
        "final_arrays": {k: list(r["final"][k].shape) for k in ("actions_normalized", "actions_physical")}}
json.dump(plan, open(out, "w"), indent=1)
print(json.dumps({"executed": executed, "step_tensors": list(plan["step_tensors"]), "scalar": plan["step_scalar_keys"]}))
