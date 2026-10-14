"""Claude's independent C4 audit (holdout source collection and selection). Reads the raw capture tree,
the frozen prereg and the selection file; imports neither g2_prereg.py nor Codex's collection scripts.

Checks: expert seeds (ascending from the registered starts, first three passing, rejections logged, no
reuse of exploration seeds), episode completeness (contiguous chunks, terminal success/step_lim only on
the last chunk, step limits), the holdout selection re-derived with my own sampler, every selected file
hash (record, trace, all listed PNGs), and GT eligibility re-derived from the raw record predicates.

    python c4_audit_claude.py --prereg P --capture ROOT --selection S --out audit.json
"""
import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

TASKS = ["hanging_mug", "place_empty_cup", "turn_switch"]
STAGES = [("early", "middle"), ("middle", "late"), ("early", "late")]
CENT = {"early": 1 / 6, "middle": 1 / 2, "late": 5 / 6}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def stage(c, n):
    p = (c + 0.5) / n
    return "early" if p < 1 / 3 else ("middle" if p < 2 / 3 else "late")


def my_sampler(rows):
    """Prereg holdout rule: Random(20261017); task, episode, slot order; per episode stage pair by episode
    index; random choice among unused chunks of the requested stage, else nearest unused to the stage
    centre with lower index tie break."""
    rng, out = random.Random(20261017), []
    for t in TASKS:
        for ep in range(3):
            g = sorted([r for r in rows if r["task"] == t and r["episode"] == ep], key=lambda r: r["chunk"])
            n = len(g)
            if n < 2 or [r["chunk"] for r in g] != list(range(n)):
                raise ValueError(f"insufficient/noncontiguous {t} ep{ep}")
            used = set()
            for slot, stg in enumerate(STAGES[ep]):
                cand = [r for r in g if r["chunk"] not in used and stage(r["chunk"], n) == stg]
                if cand:
                    r, fb = rng.choice(cand), False
                else:
                    r = min((r for r in g if r["chunk"] not in used),
                            key=lambda r: (abs((r["chunk"] + 0.5) / n - CENT[stg]), r["chunk"]))
                    fb = True
                used.add(r["chunk"])
                out.append({"task": t, "episode": ep, "slot": slot, "chunk": r["chunk"], "requested": stg,
                            "stage": stage(r["chunk"], n), "progress": (r["chunk"] + 0.5) / n, "fallback": fb, "n": n})
    return out


def gt_eligible(rec, rec_path):
    reasons = []
    if rec.get("complete") is not True:
        reasons.append("complete")
    if rec.get("inference_horizon") != 32:
        reasons.append("horizon")
    if rec.get("executed_count") != 32:
        reasons.append("executed_count")
    rows = rec.get("actions_executed") or []
    if [r.get("offset") for r in rows] != list(range(1, 33)) or any(r.get("count_delta") != 1 or r.get("exception") is not None for r in rows):
        reasons.append("execution_rows")
    if [p.get("offset") for p in rec.get("post_obs") or []] != list(range(4, 33, 4)):
        reasons.append("post_offsets")
    cams_ok = True
    for entry in [rec.get("pngs") or {}] + [p.get("pngs") or {} for p in rec.get("post_obs") or []]:
        if set(entry) != {"head", "left", "right"}:
            cams_ok = False
            continue
        for v in entry.values():
            if not Path(v["path"]).exists() or sha(v["path"]) != v["sha256"]:
                cams_ok = False
    if not cams_ok:
        reasons.append("cameras_or_hashes")
    a = rec.get("actions")
    if not (isinstance(a, list) and len(a) == 32 and all(len(x) == 20 and all(isinstance(y, (int, float)) and math.isfinite(y) for y in x) for x in a)):
        reasons.append("commands")
    return not reasons, reasons


def main():
    ap = argparse.ArgumentParser()
    for n in ("prereg", "capture", "selection", "out"):
        ap.add_argument("--" + n, required=True)
    a = ap.parse_args()
    pre = json.loads(Path(a.prereg).read_text())
    H = pre["holdout"]
    root = Path(a.capture)
    issues, notes = [], {}

    # -- seeds -------------------------------------------------------------------------------------
    explore_seeds = defaultdict(set)
    for o in pre["observations"]:
        explore_seeds[o["task"]].add(json.loads(Path(o["record"]).read_text())["env_seed"])
    rows = []
    for t in TASKS:
        audit = [json.loads(l) for l in (root / t / "seed_audit.jsonl").read_text().splitlines()]
        seeds = [r["seed"] for r in audit]
        if seeds != list(range(H["expert_seed_starts"][t], H["expert_seed_starts"][t] + len(seeds))):
            issues.append(f"{t}: seed audit not a contiguous ascending scan from the registered start")
        acc = [r for r in audit if r["accepted"]]
        if [r["episode"] for r in acc] != [0, 1, 2] or len(acc) != H["new_episodes_per_task"]:
            issues.append(f"{t}: accepted episodes {[r['episode'] for r in acc]}")
        if audit and not audit[-1]["accepted"]:
            issues.append(f"{t}: scan continued after the third acceptance")
        if any((r["reason"] == "accepted") != r["accepted"] for r in audit):
            issues.append(f"{t}: inconsistent accept reasons")
        if {r["seed"] for r in acc} & explore_seeds[t]:
            issues.append(f"{t}: holdout seed reuses an exploration seed")
        notes[t] = {"scan": [(r["seed"], r["reason"]) for r in audit], "explore_max_seed": max(explore_seeds[t])}
        for ep in range(3):
            recs = sorted((root / t / f"ep{ep:04d}").glob("chunk*/record.json"))
            for rp in recs:
                r = json.loads(rp.read_text())
                rows.append({"task": t, "episode": ep, "chunk": r["chunk"], "env_seed": r["env_seed"], "termination": r["termination"],
                             "take_action_cnt": r["take_action_cnt"], "executed_count": r["executed_count"],
                             "record": str(rp.resolve()), "record_sha256": sha(rp),
                             "trace": r["probe_receipt"]["trace_path"], "trace_sha256": r["probe_receipt"]["sha256"],
                             "folder_chunk": int(rp.parent.name[5:])})
            if acc and len(acc) > ep:
                want = acc[ep]["seed"]
                got = {r["env_seed"] for r in rows if r["task"] == t and r["episode"] == ep}
                if got != {want}:
                    issues.append(f"{t} ep{ep}: env_seed {got} != accepted {want}")

    # -- completeness ---------------------------------------------------------------------------------
    for t in TASKS:
        for ep in range(3):
            g = sorted([r for r in rows if r["task"] == t and r["episode"] == ep], key=lambda r: r["chunk"])
            if [r["chunk"] for r in g] != list(range(len(g))) or any(r["chunk"] != r["folder_chunk"] for r in g):
                issues.append(f"{t} ep{ep}: chunks not contiguous / folder mismatch")
            if len(g) < 2:
                issues.append(f"{t} ep{ep}: fewer than 2 chunks")
            if any(r["termination"] != "none" for r in g[:-1]) or g[-1]["termination"] not in ("success", "step_lim"):
                issues.append(f"{t} ep{ep}: termination pattern {[r['termination'] for r in g]}")
            # take_action_cnt is the count when the chunk was requested; the limit is reached inside the chunk
            end = g[-1]["take_action_cnt"] + g[-1]["executed_count"]
            if g[-1]["termination"] == "step_lim" and end < H["native_step_limits"][t]:
                issues.append(f"{t} ep{ep}: step_lim before the native limit ({end})")
            if g[-1]["termination"] == "success" and end > H["native_step_limits"][t]:
                issues.append(f"{t} ep{ep}: success after the native limit ({end})")
            if any(sha(r["trace"]) != r["trace_sha256"] for r in g):
                issues.append(f"{t} ep{ep}: trace hash mismatch")
            notes[f"{t}_ep{ep}"] = {"n_chunks": len(g), "outcome": g[-1]["termination"], "actions_at_end": end}

    # -- selection ------------------------------------------------------------------------------------
    sel = json.loads(Path(a.selection).read_text())
    mine = my_sampler(rows)
    got = sorted(sel["observations"], key=lambda o: (TASKS.index(o["task"]), o["episode"], o["selection_slot"]))
    if len(got) != len(mine) or len(got) != 18:
        issues.append(f"selection size {len(got)} vs mine {len(mine)}")
    for m, o in zip(mine, got):
        exp = (m["task"], m["episode"], m["slot"], m["chunk"], m["requested"], m["stage"], m["fallback"], m["n"])
        obs = (o["task"], o["episode"], o["selection_slot"], o["chunk"], o["requested_stage"], o["relative_stage"],
               o["selection_fallback"], o["n_chunks"])
        if exp != obs or abs(o["relative_progress"] - m["progress"]) > 1e-12:
            issues.append(f"selection differs: mine {exp} vs selection {obs}")
        if o["obs_id"] != f"{o['task']}_ep{o['episode']:04d}_chunk{o['chunk']:04d}" or o.get("split") != "holdout":
            issues.append(f"selection identity/split {o['obs_id']}")
        r = next(x for x in rows if (x["task"], x["episode"], x["chunk"]) == (o["task"], o["episode"], o["chunk"]))
        if (o["record"], o["record_sha256"], o["trace"], o["trace_sha256"], o["env_seed"]) != (r["record"], r["record_sha256"], r["trace"], r["trace_sha256"], r["env_seed"]):
            issues.append(f"selection file refs {o['obs_id']}")
        last = max(x["chunk"] for x in rows if (x["task"], x["episode"]) == (o["task"], o["episode"]))
        outcome = next(x["termination"] for x in rows if (x["task"], x["episode"], x["chunk"]) == (o["task"], o["episode"], last))
        if o["episode_outcome"] != outcome:
            issues.append(f"episode outcome {o['obs_id']}")
        for p in o["pngs"]:
            if sha(p["path"]) != p["sha256"]:
                issues.append(f"png hash {p['path']}")
        rec = json.loads(Path(o["record"]).read_text())
        el, reasons = gt_eligible(rec, o["record"])
        if el != o["gt"]["eligible"]:
            issues.append(f"GT eligibility {o['obs_id']}: mine {el} {reasons} vs selection {o['gt']}")
        if el and len(o["pngs"]) != 27:
            issues.append(f"GT-eligible {o['obs_id']} lists {len(o['pngs'])} PNGs")
    notes["selection"] = [(o["obs_id"], o["requested_stage"], o["relative_stage"], o["selection_fallback"], o["gt"]["eligible"]) for o in got]
    notes["gt_eligible_by_task"] = {t: sum(o["gt"]["eligible"] for o in got if o["task"] == t) for t in TASKS}
    if sel.get("prereg_sha256") != sha(a.prereg):
        issues.append("selection prereg sha")
    out = {"format": "claude-c4-audit/1", "script_sha256": sha(__file__), "prereg_sha256": sha(a.prereg),
           "selection_sha256": sha(a.selection), "n_records": len(rows), "issues": issues, "notes": notes}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({"issues": issues, "n_records": len(rows), "gt_eligible_by_task": notes["gt_eligible_by_task"],
                      "episodes": {k: v for k, v in notes.items() if "_ep" in k}}, indent=1))


if __name__ == "__main__":
    main()
