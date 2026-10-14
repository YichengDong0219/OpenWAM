"""Result-blind deterministic G1 registration from the complete G0 metadata inventory.

This script uses only chunk indices, episode outcomes, schedules and input hashes;
it never loads G1 outputs or model weights. Joint approval is a separate freeze receipt
that references the exact SHA256 of this JSON (the JSON is not edited on approval).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random

TASKS = ("hanging_mug", "place_empty_cup", "turn_switch")
STAGES = (("early", "middle"), ("middle", "late"), ("early", "late"),
          ("early", "middle"), ("middle", "late"))
CENTERS = {"early": 1/6, "middle": 1/2, "late": 5/6}
VIDEO = [1001, 1002, 1003, 1004]
ACTION = [2001, 2002, 2003, 2004]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(metadata, *, metadata_sha, protocol_sha, source_commit, sampler_sha):
    rows = sorted(metadata, key=lambda x: (x["task"], x["episode"], x["chunk"]))
    if len(rows) != 160 or {r["task"] for r in rows} != set(TASKS):
        raise ValueError("expected complete G0 inventory: 160 chunks, three tasks")
    if len({r["trace"] for r in rows}) != len(rows):
        raise ValueError("duplicate trace")
    rng = random.Random(20261013)
    observations, checks, episodes = [], [], []
    for task in TASKS:
        if {r["episode"] for r in rows if r["task"] == task} != set(range(5)):
            raise ValueError(f"{task}: expected episodes 0..4")
        for ep in range(5):
            group = [r for r in rows if r["task"] == task and r["episode"] == ep]
            n = len(group)
            if [r["chunk"] for r in group] != list(range(n)) or n < 2:
                raise ValueError(f"{task}/{ep}: insufficient or noncontiguous chunks")
            outcome = group[-1]["termination"]
            if outcome not in ("success", "step_lim"):
                raise ValueError(f"{task}/{ep}: unrecognized final outcome {outcome}")
            split = "explore" if ep < 3 else "holdout"
            episodes.append(dict(task=task, episode=ep, split=split, n_chunks=n, outcome=outcome))
            selected = set()
            for slot, stage in enumerate(STAGES[ep]):
                def progress(r):
                    return (r["chunk"] + 0.5) / n
                def stage_of(r):
                    p = progress(r)
                    return "early" if p < 1/3 else "middle" if p < 2/3 else "late"
                candidates = [r for r in group if stage_of(r) == stage and r["chunk"] not in selected]
                fallback = not candidates
                r = (rng.choice(candidates) if candidates else
                     min((x for x in group if x["chunk"] not in selected),
                         key=lambda x: (abs(progress(x) - CENTERS[stage]), x["chunk"])))
                selected.add(r["chunk"])
                for field in ("trace_sha256", "record_sha256"):
                    digest = r[field]
                    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                        raise ValueError(f"invalid {field}")
                steps = r["steps"]
                if len(steps) != 10 or [s["i"] for s in steps] != list(range(10)):
                    raise ValueError("expected ten ordered forward steps")
                if any(not math.isfinite(s[k]) for s in steps for k in ("sigma_v", "sigma_a", "t_v", "t_a")):
                    raise ValueError("nonfinite schedule")
                bands = {band: min(steps, key=lambda s: (abs(s["sigma_v"]-target), s["i"]))
                         for band, target in (("high", .9), ("mid", .5), ("low", .1))}
                if len({s["i"] for s in bands.values()}) != 3:
                    raise ValueError("sigma bands collide; resolve before registration")
                obs_id = f"{task}_ep{ep:04d}_chunk{r['chunk']:04d}"
                observations.append(dict(obs_id=obs_id, task=task, episode=ep, split=split,
                    trace=r["trace"], trace_sha256=r["trace_sha256"], record=r["record"],
                    record_sha256=r["record_sha256"], chunk=r["chunk"], n_chunks=n,
                    selection_slot=slot, requested_stage=stage, relative_stage=stage_of(r),
                    relative_progress=progress(r), selection_fallback=fallback, episode_outcome=outcome,
                    sigma_steps={b:s["i"] for b,s in bands.items()}, sigma_details=bands,
                    minimum_forward_sigma=min(s["sigma_v"] for s in steps)))
                if ep == 0 and slot == 0:
                    checks.append({"obs_id": obs_id})
    pairs = []
    for kind in ("primary_same_action", "secondary_shifted_action"):
        for v in VIDEO:
            for j, a in enumerate(ACTION):
                for d in VIDEO:
                    if d != v:
                        pairs.append(dict(receiver=[v,a], donor=[d,a if kind == "primary_same_action"
                            else ACTION[(j+1) % len(ACTION)]], pair_kind=kind))
    rating_rng = random.Random(20261015)
    rating_clips = []
    cells = [(v, a) for v in VIDEO for a in ACTION]
    for obs in sorted(observations, key=lambda o: o["obs_id"]):
        for v, a in sorted(rating_rng.sample(cells, 2)):
            rating_clips.append({"obs_id": obs["obs_id"], "split": obs["split"],
                                 "video_seed": v, "action_seed": a})
    return {
        "format": "openwam-g1-prereg/1",
        "approval": "requires separate joint freeze receipt referencing this file SHA256",
        "provenance": {"g0_metadata_sha256": metadata_sha, "protocol_sha256": protocol_sha,
                       "sampler_sha256": sampler_sha, "reviewed_model_code_commit": source_commit},
        "scope": ["Q1_seed_grid", "Q2_fixed_state_local_response", "Q4_decode_and_rating_pack_only"],
        "excluded": ["G2_full_trajectory_swap", "IDM_download", "training", "closed_loop_intervention",
                     "uncalibrated_active_arm_labels", "unappointed_semantic_raters"],
        "selection": {"rng": "Python stdlib random.Random", "seed": 20261013,
                      "input_sort": ["task", "episode", "chunk"],
                      "stage_formula": "(chunk_index+0.5)/n; early<1/3, middle<2/3, else late",
                      "episode_stage_pairs": STAGES, "fallback": "nearest unused chunk to stage center; lower index breaks ties",
                      "stratification": "all five episodes per task, two observations each; outcomes annotated without balancing"},
        "video_seeds": VIDEO, "action_seeds": ACTION,
        "observations": observations, "episodes": episodes, "pairs": pairs, "isolated_checks": checks,
        "metrics": {
            "Q1_primary": "two-way centered SS over 32x6 physical xyz; raw SS and fraction; zero total => null",
            "Q1_scatter": "all six unordered seed pairs within each fixed other seed; D_xyz mean metres primary, max secondary",
            "Q1_secondary": "rotation geodesic mean/max degrees with degenerate counts; continuous gripper mean/max abs",
            "Q2_primary": "RMS velocity swap-minus-recomputed receiver over 32xactive20 normalized coordinates",
            "Q2_secondary": ["l2", "max_abs", "base_l2", "base_rms", "rel_l2 (zero denominator => null)"],
            "active_dimensions": list(range(10))+list(range(34,44)),
            "no_op": "same receiver and step recomputed versus recorded velocity; retain all errors",
            "isolated": "first selection slot in ep0 per task; first video/action receiver, second video same action donor, three bands",
            "response_interpretation": "numerical sensitivity relative to measured no-op; no semantic or practical success threshold",
            "rotation": "first two columns, Gram-Schmidt, conversion normalization floor 1e-8; degeneracy threshold 1e-6",
            "degenerate_policy": "report counts; geodesic angles for invalid rotations are not interpretable and must be flagged",
            "decode_proxy": "mean absolute adjacent-frame RGB change in head/left-wrist/right-wrist canvas slots, scale 0..255; no semantic labels"},
        "statistics": {"aggregation": "pair mean per observation/band/kind, observation mean per episode, episode mean per task",
                       "Q1_aggregation": "decomposition per observation then equal observation/episode weights; no pooling raw SS across observations",
                       "bootstrap_unit": "whole episode within task and split", "bootstrap_repetitions": 10000,
                       "bootstrap_seed": 20261014, "interval": "95% percentile; numpy quantile linear interpolation",
                       "split_policy": "explore ep0-2 only first; freeze exploration analysis code and interpretation before computing holdout ep3-4",
                       "missing_data": "never silently drop missing/nonfinite values; mark incomplete and report denominators",
                       "no_p_values": True, "no_cross_task_pooling": True},
        "verification": {"selection": "per task ep0 first selected observation, receiver first video/action, donor second video same action; all three sigma bands",
                         "independent": "GPU3 full baseline generation and direct single-step rerun; compare tensors and metrics against saved results",
                         "rating_pack": "60 clips: two per observation, stdlib random.Random(20261015), sorted obs_id, sample from video-major/action-minor grid; holdout materials after release only",
                         "rating_clips": rating_clips},
        "limitations": ["local forward input sigma never below approximately 0.357; low means lowest observed noisy input, not clean future",
                        "same initial action noise does not eliminate donor action feedback under mutual attention",
                        "per-task split sizes three/two episodes; bootstrap intervals descriptive only",
                        "relative trajectory progress is not semantic task phase",
                        "all holdout episodes succeeded; turn_switch exploration episodes all exhausted step limit; split differences confound outcome and phase coverage",
                        "G0 data reused; holdout is reserved replication, not independently collected confirmation",
                        "pixel changes and normalized velocity changes do not establish semantic following or execution utility"],
        "expected_counts": {"observations": 30, "explore_observations": 18, "holdout_observations": 12,
                            "grid_generations": 480, "pairs_per_kind_per_observation": 48,
                            "step_rows": 8640, "explore_step_rows": 5184, "holdout_step_rows": 3456,
                            "isolated_rows": 9}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = build(json.loads(args.metadata.read_text()), metadata_sha=sha(args.metadata),
                   protocol_sha=sha(args.protocol), source_commit=args.source_commit, sampler_sha=sha(__file__))
    encoded = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists() and args.out.read_bytes() != encoded:
        raise SystemExit("refusing to overwrite a different registration; choose a new candidate path")
    tmp = args.out.with_suffix(args.out.suffix+".tmp")
    tmp.write_bytes(encoded)
    tmp.replace(args.out)
    print(json.dumps({"sha256": sha(args.out), "counts": result["expected_counts"],
                      "fallbacks": sum(o["selection_fallback"] for o in result["observations"])}))


if __name__ == "__main__":
    main()
