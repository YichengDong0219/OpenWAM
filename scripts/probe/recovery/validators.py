"""Stage validators: turn raw job outputs into pinned receipts. They never read success/rescue counts to promote.

Each receipt carries validator_sha256 (this file), backend_manifest_sha256, and hashes of the inputs it read.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
SEED_PARAM = {"Q_eng": 1, "Q_eval": 2}
TARGET = {"Q_eng": 3, "Q_eval": 20}
START = {"Q_eng": 200000, "Q_eval": 300000}
ROTATION = (("B0", "B1", "B2"), ("B1", "B2", "B0"), ("B2", "B0", "B1"))


def sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def validator_sha() -> str:
    return sha_file(HERE)


def selection_files() -> list[Path]:
    return [ROOT / "benchmarks/robotwin/eval_policy_wrapper.py", ROOT / "scripts/probe/recovery/job.py", HERE]


def selection_algorithm_sha() -> str:
    h = hashlib.sha256()
    for p in selection_files():
        h.update(p.name.encode()); h.update(sha_file(p).encode())
    return h.hexdigest()


def write_receipt(path: Path, body: dict) -> dict:
    body = {**body, "validator_sha256": validator_sha()}
    path.write_text(json.dumps(body, indent=1, sort_keys=True))
    return {"path": str(path), "sha256": sha_file(path), **({"manifest_sha256": body["manifest_sha256"]}
                                                           if "manifest_sha256" in body else {})}


# ---------------------------------------------------------------- V
def validate_V(result_path: Path, backend_manifest_sha: str, receipt_path: Path) -> dict:
    r = json.loads(Path(result_path).read_text())
    ok = (r.get("engineering_pass") is True and len(r.get("runs", [])) == 2
          and all(x["bitwise_equal_all"] for x in r["runs"]))
    return write_receipt(receipt_path, {"stage": "V", "engineering_pass": ok, "backend_manifest_sha256": backend_manifest_sha,
                                        "v_result_sha256": sha_file(result_path), "residency": r.get("residency"),
                                        "peak_reserved_mib": r.get("peak_reserved_mib")})


# ---------------------------------------------------------------- Q
def classify_candidate(cdir: Path, rc) -> dict:
    """accepted / rejected (expert says not feasible) / infra (job failed without an expert verdict)."""
    audit = cdir / "hanging_mug" / "seed_audit.jsonl"
    rows = [json.loads(l) for l in audit.read_text().splitlines()] if audit.exists() else []
    manifest = cdir / "manifest.json"
    if rc == 0 and manifest.exists() and rows and rows[-1]["accepted"]:
        return {"verdict": "accepted", "audit": rows[-1], "manifest": json.loads(manifest.read_text())}
    if rows and not rows[-1]["accepted"]:
        return {"verdict": "rejected", "audit": rows[-1]}
    return {"verdict": "infra", "rc": rc, "audit": rows[-1] if rows else None}


def validate_Q(stage: str, candidates: list[dict], backend_manifest_sha: str, out_dir: Path, receipt_path: Path) -> dict:
    """candidates: ordered [{seed, dir, rc}] actually attempted. Seals the combined manifest of the first TARGET accepted."""
    from benchmarks.utils.eval_manifest import seal_manifest, write_manifest
    seeds = [c["seed"] for c in candidates]
    contiguous = seeds == list(range(START[stage], START[stage] + len(seeds)))
    accepted, rows, benchmark = [], [], None
    for c in candidates:
        v = classify_candidate(Path(c["dir"]), c["rc"])
        rows.append({"seed": c["seed"], "verdict": v["verdict"], "reason": (v.get("audit") or {}).get("reason")})
        if v["verdict"] == "accepted":
            e = v["manifest"]["entries"]
            if len(e) != 1 or e[0]["seed"] != c["seed"]:
                raise ValueError(f"candidate {c['seed']} manifest does not hold exactly its own seed")
            accepted.append(e[0])
            benchmark = v["manifest"]["benchmark"]
    infra = [r for r in rows if r["verdict"] == "infra"]
    ok = contiguous and not infra and len(accepted) == TARGET[stage] and len(candidates) <= {"Q_eng": 30, "Q_eval": 200}[stage]
    body = {"stage": stage, "engineering_pass": ok, "backend_manifest_sha256": backend_manifest_sha,
            "selection_algorithm_sha256": selection_algorithm_sha(), "candidate_count": len(candidates),
            "candidates": rows, "contiguous": contiguous, "infra_failures": len(infra)}
    if ok:
        entries = [{**e, "episode": i} for i, e in enumerate(accepted[:TARGET[stage]])]
        sealed = seal_manifest({"benchmark": benchmark, "task": "hanging_mug", "mode": "demo_randomized",
                                "entries": entries})
        mpath = write_manifest(out_dir / "manifest.json", sealed)
        order = [list(ROTATION[i % 3]) for i in range(len(entries))]
        (out_dir / "arm_order.json").write_text(json.dumps(order))
        body.update(accepted_scene_seeds=[e["seed"] for e in entries], manifest_path=str(mpath),
                    manifest_sha256=sha_file(mpath), manifest_hash=sealed["manifest_hash"],
                    arm_order_sha256=sha_file(out_dir / "arm_order.json"))
    return write_receipt(receipt_path, body)


# ---------------------------------------------------------------- M2 / M3
def _records(arm_dir: Path) -> list[dict]:
    recs = [json.loads(p.read_text()) for p in sorted(arm_dir.glob("hanging_mug/ep*/chunk*/record.json"))]
    return sorted(recs, key=lambda d: d["chunk"])


def _events(arm_dir: Path) -> dict | None:
    p = next(iter(arm_dir.glob("hanging_mug/ep*/recovery_events.json")), None)
    return json.loads(p.read_text()) if p else None


def _check_clamp(ref_trace: Path, cand_trace: Path, ntt: float = 1000.0) -> dict:
    ref = torch.load(ref_trace, map_location="cpu", weights_only=False)
    cand = torch.load(cand_trace, map_location="cpu", weights_only=False)
    tgt, eps = ref["final"]["video_latents"], cand["init"]["video_noise"]
    bad = []
    for s in cand["steps"]:
        sigma = float(s["t_v"]) / ntt
        exp = ((1.0 - sigma) * tgt[:, :, 1:].float() + sigma * eps[:, :, 1:].float()).to(s["video_latents_in"].dtype)
        if not torch.equal(s["video_latents_in"][:, :, 1:], exp):
            bad.append(int(s["i"]))
        if not torch.equal(s["video_latents_in"][:, :, :1], cand["cond"]["first_frame_latents"].to(s["video_latents_in"].dtype)):
            bad.append(f"frame0@{s['i']}")
    same_cond = torch.equal(ref["cond"]["first_frame_latents"], cand["cond"]["first_frame_latents"])
    return {"steps_bad": bad, "same_first_frame_as_ref": bool(same_cond),
            "clamp_meta_ok": cand["meta"].get("clamp_source_sha256") == sha_file(ref_trace)}


def validate_episodes(stage: str, q_receipt: dict, episodes: list[dict], backend_manifest_sha: str,
                      scientific_bundle_sha: str, local_backend: bool, receipt_path: Path) -> dict:
    """episodes: [{ordinal, seed, arm, dir, rc}] actually run. M2: engineering gate; M3: completeness report."""
    n_seeds = TARGET["Q_eng" if stage == "M2" else "Q_eval"]
    problems, per_seed, exercised = [], {}, {"B1": 0, "B2": 0}
    for e in episodes:
        per_seed.setdefault(e["ordinal"], {})[e["arm"]] = e
    for ordn in range(n_seeds):
        arms = per_seed.get(ordn, {})
        if set(arms) != {"B0", "B1", "B2"}:
            problems.append(f"ordinal {ordn}: arms run {sorted(arms)}")
            continue
        data = {}
        for arm, e in arms.items():
            d = Path(e["dir"]) / arm
            info = json.loads((Path(e["dir"]) / "episode_info.json").read_text()) if (Path(e["dir"]) / "episode_info.json").exists() else {}
            recs, ev = _records(d), _events(d)
            if e["rc"] != 0 or not (d / "results.json").exists() or not recs or ev is None:
                problems.append(f"ordinal {ordn} {arm}: incomplete (rc={e['rc']})"); continue
            if not recs[-1].get("complete") or "terminal_task_fields" not in recs[-1]:
                problems.append(f"ordinal {ordn} {arm}: last chunk lacks termination state")
            if local_backend and (info.get("min_free_mib") is None or info["min_free_mib"] < 2048):
                problems.append(f"ordinal {ordn} {arm}: local free-memory floor violated ({info.get('min_free_mib')})")
            data[arm] = (recs, ev, d)
        if len(data) != 3:
            continue
        trig = {arm: next((x["chunk"] for x in data[arm][1]["events"] if x["eligible_trigger"]), None) for arm in data}
        first = [t for t in trig.values() if t is not None]
        cut = min(first) if first else None
        if len(set(trig.values())) != 1:
            problems.append(f"ordinal {ordn}: first trigger differs across arms {trig}")
        lim = cut if cut is not None else min(len(v[0]) for v in data.values())
        for c in range(lim):
            acts = [data[a][0][c]["actions"] for a in ("B0", "B1", "B2")]
            tfs = [data[a][0][c]["task_fields"] for a in ("B0", "B1", "B2")]
            if not (acts[0] == acts[1] == acts[2] and tfs[0] == tfs[1] == tfs[2]):
                problems.append(f"ordinal {ordn}: common prefix broken at chunk {c}"); break
        if cut is None and len({len(v[0]) for v in data.values()}) != 1:
            problems.append(f"ordinal {ordn}: no trigger yet arms differ in length")
        refs = {}
        for arm in ("B1", "B2"):
            recs, ev, d = data[arm]
            for x in ev["events"]:
                if not x["eligible_trigger"]:
                    continue
                rec = next((r for r in recs if r["chunk"] == x["chunk"]), None)
                rr = (rec or {}).get("recovery_receipt")
                if not rr or rr.get("n_generations") != 2 or rr.get("arm") != arm:
                    problems.append(f"ordinal {ordn} {arm}: trigger chunk {x['chunk']} lacks a recovery receipt"); continue
                ref_p, cand_p = Path(rr["ref"]["trace_path"]), Path(rr["cand"]["trace_path"])
                if not ref_p.exists() or sha_file(ref_p) != rr["ref"]["sha256"] or sha_file(cand_p) != rr["cand"]["sha256"]:
                    problems.append(f"ordinal {ordn} {arm}: recovery trace hash mismatch"); continue
                exercised[arm] += 1
                refs[(arm, x["correction_index"])] = torch.load(ref_p, map_location="cpu", weights_only=False)["final"]["video_latents"]
                cand_meta = torch.load(cand_p, map_location="cpu", weights_only=False)["meta"]
                if arm == "B2":
                    chk = _check_clamp(ref_p, cand_p)
                    if chk["steps_bad"] or not chk["same_first_frame_as_ref"] or not chk["clamp_meta_ok"]:
                        problems.append(f"ordinal {ordn} B2: clamp invariant failed {chk}")
                elif cand_meta.get("clamp_source_path") is not None:
                    problems.append(f"ordinal {ordn} B1: candidate unexpectedly clamped")
        if ("B1", 0) in refs and ("B2", 0) in refs and not torch.equal(refs[("B1", 0)], refs[("B2", 0)]):
            problems.append(f"ordinal {ordn}: first reference proposal R differs between B1 and B2")
    complete = sum(1 for o in per_seed.values() for a, e in o.items() if e["rc"] == 0)
    body = {"stage": stage, "backend_manifest_sha256": backend_manifest_sha, "scientific_bundle_sha256": scientific_bundle_sha,
            "q_manifest_sha256": q_receipt.get("manifest_sha256"), "complete_episodes": complete,
            "recovery_invocations_exercised": exercised, "problems": problems}
    if stage == "M2":
        body["common_prefix_pass"] = not any("prefix" in p or "trigger differs" in p for p in problems)
        body["invariants_pass"] = not problems
        body["recovery_path_exercised"] = exercised["B1"] > 0 and exercised["B2"] > 0
        body["engineering_pass"] = (complete == 9 and not problems and body["recovery_path_exercised"])
    else:
        body["engineering_pass"] = None  # M3 is a descriptive run; this is its completeness/integrity record
    return write_receipt(receipt_path, body)
