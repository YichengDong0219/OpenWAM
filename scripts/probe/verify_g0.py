"""Independent G0 checks on one frozen client input (no simulator state restore).

Run in the OpenWAM env on the server host so trace files can be loaded directly.
Checks fail closed; V5 here verifies the same-input capture switch. Whole-rollout
environment alignment remains the separate C3/client cross-review requirement.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.utils import client
from benchmarks.utils.transport import WSPolicyClient


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float().numpy().astype(np.float64)
    return np.asarray(value, dtype=np.float64)


def difference(a, b):
    a, b = array(a), array(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f"shape/nonfinite mismatch {a.shape}, {b.shape}")
    absolute = np.abs(a - b)
    relative = absolute / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-12)
    return {"max_abs": float(absolute.max(initial=0)), "max_relative": float(relative.max(initial=0)),
            "rms": float(np.sqrt(np.mean(absolute ** 2)))}


def require_keys(obj, keys, context):
    missing = set(keys) - set(obj)
    if missing:
        raise ValueError(f"{context} missing fields: {sorted(missing)}")


def validate_trace(trace):
    require_keys(trace, ("meta", "cond", "init", "steps", "final"), "trace")
    meta = trace["meta"]
    require_keys(meta, ("code", "deploy_config_sha256", "checkpoint_path", "seed", "video_seed", "action_seed",
        "denoise_steps", "schedule", "dit_cache_enabled", "compile_enabled", "num_frames", "action_num_frames",
        "active_action_mask", "replay_mode", "replay_source_sha256", "timestamp"), "meta")
    if meta["dit_cache_enabled"] or meta["compile_enabled"]:
        raise ValueError("verification requires eager/no-cache trace")
    if meta["num_frames"] != 9 or meta["action_num_frames"] != 33:
        raise ValueError("unexpected RoboTwin-Full frame lengths")
    require_keys(trace["cond"], ("prompt", "image", "proprio", "first_frame_latents"), "cond")
    require_keys(trace["init"], ("video_noise", "action_noise"), "init")
    require_keys(trace["final"], ("video_latents", "action_latents", "actions_physical", "actions_returned", "actions_normalized"), "final")
    first = trace["cond"]["first_frame_latents"]
    raw_noise = trace["init"]["video_noise"]
    if first is not None and isinstance(raw_noise, torch.Tensor) and torch.equal(raw_noise[:, :, :first.shape[2]], first):
        raise ValueError("init.video_noise contains conditioned first frame, not original noise")
    active = meta["active_action_mask"]
    if active is None or int(array(active).sum()) != 20:
        raise ValueError("trace must record actual 20-active-dimension mask")
    if array(trace["final"]["actions_physical"]).shape != (32, 20):
        raise ValueError("physical action chunk is not (32,20)")
    steps = trace["steps"]
    schedule = meta["schedule"]
    expected = [i for i in range(len(schedule) - 1) if schedule[i] != schedule[i+1]]
    if [s["i"] for s in steps] != expected:
        raise ValueError("trace steps do not cover executed schedule")
    for step in steps:
        require_keys(step, ("i", "t_v", "t_a", "sigma_v", "sigma_a", "video_latents_in", "action_latents_in",
                            "video_velocity", "action_velocity"), "step")
        tv, ta = schedule[step["i"]]
        if step["t_v"] != tv or step["t_a"] != ta:
            raise ValueError("recorded timestep does not match schedule")
        for key in ("video_latents_in", "action_latents_in", "video_velocity", "action_velocity"):
            tensor = step[key]
            if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or not torch.isfinite(tensor).all():
                raise ValueError(f"{key} must be a finite CPU tensor")
        if first is not None and not torch.equal(step["video_latents_in"][:, :, :first.shape[2]], first):
            raise ValueError("conditioning first frame changed inside trace")
    return {"steps": len(steps), "video_shape": list(steps[0]["video_latents_in"].shape),
            "action_shape": list(steps[0]["action_latents_in"].shape),
            "video_dtype": str(steps[0]["video_latents_in"].dtype),
            "action_dtype": str(steps[0]["action_latents_in"].dtype)}


def trace_values(trace):
    values = {"final.actions_physical": trace["final"]["actions_physical"],
              "final.action_latents": trace["final"]["action_latents"]}
    for step in trace["steps"]:
        values[f"step{step['i']}.action_velocity"] = step["action_velocity"]
    return values


def tolerance_from_repeats(traces):
    maps = [trace_values(t) for t in traces]
    tolerance = {}
    for key in maps[0]:
        metrics = [difference(a[key], b[key]) for a, b in itertools.combinations(maps, 2)]
        tolerance[key] = {"observed_max_abs": max(m["max_abs"] for m in metrics),
                          "observed_max_relative": max(m["max_relative"] for m in metrics)}
        tolerance[key]["atol"] = tolerance[key]["observed_max_abs"] * 1.1
    return tolerance


def compare_traces(reference, other, tolerance):
    a, b = trace_values(reference), trace_values(other)
    if set(a) != set(b):
        raise ValueError("trace action prediction steps differ")
    rows = {key: {**difference(a[key], b[key]), "atol": tolerance[key]["atol"]} for key in a}
    for value in rows.values():
        value["passed"] = value["max_abs"] <= value["atol"]
    return {"passed": all(v["passed"] for v in rows.values()), "tensors": rows}


def payload_from_record(record):
    imgs = {}
    for label in ("head", "left", "right"):
        image = record["pngs"][label]
        path = Path(image["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != image["sha256"]:
            raise ValueError(f"input PNG hash mismatch: {path}")
        imgs[label] = client.encode_numpy_b64(np.asarray(Image.open(path).convert("RGB")))
    return client.build_payload(head=imgs["head"], left_wrist=imgs["left"], right_wrist=imgs["right"],
                                prompt=record["prompt"], state=record["proprio"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--url", default="ws://127.0.0.1:8853")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to((ROOT / "outputs/probe").resolve()):
        parser.error("output-dir must be below outputs/probe")
    out.mkdir(parents=True, exist_ok=False)
    report = {"scope": "same-frozen-input; whole-rollout alignment is separate", "record": str(args.record), "checks": {}}
    try:
        record = json.loads(args.record.read_text())
        base = payload_from_record(record)
        with WSPolicyClient(args.url, timeout=600) as ws:
            def generate(name, *, probe=True, seeds=False, replay=None, video_seed=None, action_seed=None):
                ws.reset()
                payload = copy.deepcopy(base)
                path = out / f"{name}.pt"
                if probe:
                    payload["probe"] = {"trace_path": str(path), "video_seed": 42 if seeds else video_seed,
                        "action_seed": 42 if seeds else action_seed, "replay_trace_path": str(replay) if replay else None,
                        "replay_mode": "self" if replay else None}
                else:
                    payload.update(capture_generation=True, decode_generation=False)
                response = ws.predict_once(payload)
                atomic_json(out / f"{name}_response.json", response)
                if "generation" not in response:
                    raise ValueError("reset request failed to produce a generation")
                if not probe:
                    return None, response
                receipt = response.get("probe")
                if not receipt or receipt["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
                    raise ValueError("missing or corrupt trace receipt")
                trace = torch.load(path, map_location="cpu", weights_only=False)
                structure = validate_trace(trace)
                return trace, {"response": response, "structure": structure, "bytes": path.stat().st_size}
            repeated = [generate(f"repeat{i}") for i in range(5)]
            traces = [x[0] for x in repeated]
            tolerance = tolerance_from_repeats(traces)
            report["checks"]["V0"] = {"passed": True, "repeat_count": 5, "tolerance": tolerance,
                "note": "empirical max error ×1.1, zero stays zero; does not prove universal determinism"}
            report["checks"]["V2"] = {"passed": True, "structure": repeated[0][1]["structure"],
                "trace_bytes": repeated[0][1]["bytes"]}
            explicit, _ = generate("explicit_defaults", seeds=True)
            report["checks"]["V1_explicit_defaults"] = compare_traces(traces[0], explicit, tolerance)
            video_changed, _ = generate("video_seed_only", video_seed=7)
            action_changed, _ = generate("action_seed_only", action_seed=7)
            independent = {
                "video_seed_keeps_action_noise": torch.equal(traces[0]["init"]["action_noise"], video_changed["init"]["action_noise"]),
                "video_seed_changes_video_noise": not torch.equal(traces[0]["init"]["video_noise"], video_changed["init"]["video_noise"]),
                "action_seed_keeps_video_noise": torch.equal(traces[0]["init"]["video_noise"], action_changed["init"]["video_noise"]),
                "action_seed_changes_action_noise": not torch.equal(traces[0]["init"]["action_noise"], action_changed["init"]["action_noise"]),
            }
            report["checks"]["V1_seed_independence"] = {**independent, "passed": all(independent.values())}
            replayed, _ = generate("self_replay", replay=out / "repeat0.pt")
            report["checks"]["V3_self_replay"] = compare_traces(traces[0], replayed, tolerance)
            _, off = generate("capture_off", probe=False)
            measured = difference(traces[0]["final"]["actions_returned"], off["generation"]["actions"])
            allowed = tolerance["final.actions_physical"]["atol"]
            report["checks"]["V1_legacy_interface"] = {**measured, "atol": allowed, "passed": measured["max_abs"] <= allowed}
            measured_on = difference(record["actions"], traces[0]["final"]["actions_returned"])
            report["checks"]["V5_same_input_capture_on_off"] = {**measured, "atol": allowed,
                "passed": measured["max_abs"] <= allowed and measured_on["max_abs"] <= allowed,
                "online_record_reproduction": {**measured_on, "passed": measured_on["max_abs"] <= allowed}}
            negative = {}
            for name, alter in (
                ("action_timestep", lambda t: t["meta"]["schedule"].__setitem__(0, (t["meta"]["schedule"][0][0], t["meta"]["schedule"][0][1] + 1))),
                ("video_dtype", lambda t: t["steps"][0].__setitem__("video_latents_in", t["steps"][0]["video_latents_in"].float())),
            ):
                invalid = copy.deepcopy(traces[0])
                alter(invalid)
                source = out / f"invalid_{name}.pt"
                torch.save(invalid, source)
                try:
                    generate(f"reject_{name}", replay=source)
                except client.ServerError as exc:
                    negative[name] = {"passed": True, "server_error": str(exc)}
                else:
                    negative[name] = {"passed": False, "error": "incompatible replay was accepted"}
            report["checks"]["V3_replay_rejects_invalid_inputs"] = {"passed": all(x["passed"] for x in negative.values()), "cases": negative}
        report["passed"] = all(v["passed"] for v in report["checks"].values())
    except Exception as exc:
        report.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        atomic_json(out / "verification.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
