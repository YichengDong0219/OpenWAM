"""Claude's independent recomputation of Codex's G2 holdout A3 (R30/R31).

Uses only my own exploration-audit functions (g2_audit_claude.py: physical metrics, direction,
equal-weight hierarchy, whole-episode bootstrap) on the formal holdout traces; imports neither Codex's
analysis code nor its holdout adapter, nor the G2 wrappers. Observations come from the frozen holdout
manifest's selection. Also checks the exact key sets, the Q5 source-check journal, donor equivalence
and the GT subset, then compares every task-level summary with Codex's analysis.json (the coverage key
alias eligible -> eligible_fraction is mapped explicitly, not skipped).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import g2_audit_claude as A


def main():
    ap = argparse.ArgumentParser()
    for n in ("prereg", "manifest", "q5", "q6", "codex", "out"):
        ap.add_argument("--" + n, required=True)
    a = ap.parse_args()
    pre = json.loads(Path(a.prereg).read_text())
    man = json.loads(Path(a.manifest).read_text())
    sel_path = man["selection"]["path"]
    if A.sha(sel_path) != man["selection"]["sha256"]:
        raise SystemExit("selection hash")
    obs_all = json.loads(Path(sel_path).read_text())["observations"]
    by_id = {o["obs_id"]: o for o in obs_all}
    seeds = [(v, x) for v in pre["video_seeds"] for x in pre["action_seeds"]]
    G, checks, issues = A.Groups(), defaultdict(int), []

    def rows(d):
        r = [json.loads(l) for l in open(Path(d) / "index.jsonl")]
        for x in r:
            if A.sha(x["path"]) != x["sha256"]:
                issues.append(("sha", x["key"]))
        return {x["key"]: x for x in r}

    q5, q6 = rows(a.q5), rows(a.q6)
    pk = lambda p: f"{p['receiver'][0]}-{p['receiver'][1]}|{p['donor'][0]}-{p['donor'][1]}"
    exp5, exp6 = set(), set()
    for o in obs_all:
        oid = o["obs_id"]
        for v, x in seeds:
            exp5 |= {f"baseline|{oid}|{v}-{x}", f"identity|{oid}|{v}-{x}"}
            exp6 |= {f"baseline|{oid}|{v}-{x}", f"identity|{oid}|{v}-{x}"}
            if o["gt"]["eligible"]:
                exp6.add(f"gt|{oid}|{v}-{x}")
        exp5 |= {f"swap|{oid}|{pk(p)}" for p in pre["pairs"]}
        exp6 |= {f"alt|{oid}|{pk(p)}" for p in pre["pairs"]} | {f"source|{oid}"}
        if o["gt"]["eligible"]:
            exp6.add(f"gt_future|{oid}")
    checks["q5_keyset_exact"] = int(set(q5) == exp5)
    checks["q6_keyset_exact"] = int(set(q6) == exp6)
    checks["q5_units"], checks["q6_units"] = len(q5), len(q6)

    # source-check journal: one passing row per observation, in selection order
    sj = [json.loads(l) for l in open(Path(a.q5) / "source_checks.jsonl")]
    stamp = json.loads((Path(a.q5) / "RUN_STAMP.json").read_text())
    checks["source_checks_rows"] = len(sj)
    checks["source_checks_ok"] = int([r["obs_id"] for r in sj] == [o["obs_id"] for o in obs_all] and all(
        r["result"] == "steps_and_final_bitwise_equal" and r["source_trace_sha256"] == by_id[r["obs_id"]]["trace_sha256"]
        and r["runtime_binding_sha256"] == stamp["runtime_binding_sha256"] for r in sj))
    checks["no_failures_files"] = int(not (Path(a.q5) / "failures.jsonl").exists() and not (Path(a.q6) / "failures.jsonl").exists())

    # ---- Q5 ----
    cache = {}

    def load(path):
        if path not in cache:
            cache[path] = A.final(path)
        return cache[path]

    for o in obs_all:
        cache.clear()
        oid = o["obs_id"]
        for p in pre["pairs"]:
            r = q5[f"swap|{oid}|{pk(p)}"]
            tr = load(r["path"])
            R = load(q5[f"baseline|{oid}|{p['receiver'][0]}-{p['receiver'][1]}"]["path"])
            D = load(q5[f"baseline|{oid}|{p['donor'][0]}-{p['donor'][1]}"]["path"])
            if tr["meta"]["receiver_baseline"]["path"] != q5[f"baseline|{oid}|{p['receiver'][0]}-{p['receiver'][1]}"]["path"]:
                issues.append(("q5_ref", r["key"]))
            Ar, Ad, As = (np.asarray(x["final"]["actions_physical"]) for x in (R, D, tr))
            checks["q5_swap"] += 1
            checks["q5_final_equals_donor"] += int(np.array_equal(As, Ad))
            checks["q5_steps_equal_donor"] += int(A.step_equal(tr["steps"], D["steps"], ("action_latents_in", "action_velocity")))
            G.add("Q5_effect", o, A.pair_metrics(As, Ar))
            G.add("Q5_natural", o, A.pair_metrics(Ad, Ar))
            dm, _ = A.direction(Ar, Ad, As)
            G.add("Q5_direction_coverage", o, {"eligible": float(dm is not None)})
            if dm is not None:
                G.add("Q5_direction", o, dm)
        for v, x in seeds:
            b = load(q5[f"baseline|{oid}|{v}-{x}"]["path"])
            i = load(q5[f"identity|{oid}|{v}-{x}"]["path"])
            checks["q5_identity_equal"] += int(np.array_equal(np.asarray(b["final"]["actions_physical"]), np.asarray(i["final"]["actions_physical"]))
                                               and A.step_equal(b["steps"], i["steps"], ("action_latents_in", "action_velocity", "video_latents_in", "video_velocity")))

    # ---- Q6 ----
    for o in obs_all:
        oid = o["obs_id"]
        base = {(v, x): A.final(q6[f"baseline|{oid}|{v}-{x}"]["path"]) for v, x in seeds}
        for v, x in seeds:
            i = A.final(q6[f"identity|{oid}|{v}-{x}"]["path"])
            checks["q6_identity_equal"] += int(np.array_equal(np.asarray(base[(v, x)]["final"]["actions_physical"]), np.asarray(i["final"]["actions_physical"])))
        for p in pre["pairs"]:
            tr = A.final(q6[f"alt|{oid}|{pk(p)}"]["path"])
            R, D = base[tuple(p["receiver"])], base[tuple(p["donor"])]
            Ar, Ad, Aa = (np.asarray(x["final"]["actions_physical"]) for x in (R, D, tr))
            checks["q6_alt"] += 1
            checks["q6_final_equals_donor"] += int(np.array_equal(Aa, Ad))
            checks["q6_steps_equal_donor"] += int(A.step_equal(tr["steps"], D["steps"], ("action_latents_in", "action_velocity")))
            G.add("Q6_effect", o, A.pair_metrics(Aa, Ar))
            G.add("Q6_natural", o, A.pair_metrics(Ad, Ar))
            dm, _ = A.direction(Ar, Ad, Aa)
            G.add("Q6_direction_coverage", o, {"eligible": float(dm is not None)})
            if dm is not None:
                G.add("Q6_direction", o, dm)
        for sp in pre["idm_action_scatter_pairs"]:
            v, (x1, x2) = sp["video_seed"], sp["action_seed_pair"]
            G.add("Q6_action_scatter", o, A.pair_metrics(np.asarray(base[(v, x1)]["final"]["actions_physical"]),
                                                          np.asarray(base[(v, x2)]["final"]["actions_physical"])))
            checks["q6_scatter_pairs"] += 1
        if o["gt"]["eligible"]:
            cmds = np.asarray(json.loads(Path(o["record"]).read_text())["actions"], dtype=np.float64)
            for v, x in seeds:
                tr = A.final(q6[f"gt|{oid}|{v}-{x}"]["path"])
                As_, Ag = np.asarray(base[(v, x)]["final"]["actions_physical"]), np.asarray(tr["final"]["actions_physical"])
                m = A.pair_metrics(Ag, As_)
                eg, es = A.pair_metrics(Ag, cmds)["D_xyz_m"], A.pair_metrics(As_, cmds)["D_xyz_m"]
                m.update(E_gt_m=eg, E_self_m=es, delta_E_m=eg - es)
                G.add("GT_command_consistency", o, m)
                checks["gt_rows"] += 1
    gt_by_ep = defaultdict(int)
    for o in obs_all:
        if o["gt"]["eligible"]:
            gt_by_ep[f"{o['task']}_ep{o['episode']}"] += 1
    mine = {g: A.summarize(r, obs_all) for g, r in G.rows.items()}

    # ---- compare ----
    alias = {"eligible": "eligible_fraction"}
    codex = json.loads(Path(a.codex).read_text())["groups"]
    cmp = {"compared": 0, "aliased": 0, "max_abs_diff_mean": 0.0, "max_abs_diff_ci": 0.0, "mismatches": [], "codex_only": []}
    for g, tasks in mine.items():
        for t, res in tasks.items():
            ct = codex[g]["tasks"][t]
            if (ct["n_rows"], ct["n_observations_with_rows"]) != (res["n_rows"], res["n_observations_with_rows"]):
                cmp["mismatches"].append((g, t, "counts"))
            for k, v in res["metrics"].items():
                ck = alias.get(k, k)
                cmp["aliased"] += int(ck != k)
                cv = ct["metrics"].get(ck)
                if v is None or cv is None:
                    cmp["mismatches"].append((g, t, k, "null"))
                    continue
                cmp["compared"] += 1
                dm = abs(v["mean"] - cv["mean"])
                dc = max(abs(v["ci95"][0] - cv["ci95"][0]), abs(v["ci95"][1] - cv["ci95"][1]))
                cmp["max_abs_diff_mean"], cmp["max_abs_diff_ci"] = max(cmp["max_abs_diff_mean"], dm), max(cmp["max_abs_diff_ci"], dc)
                if dm > 1e-12 * max(1, abs(v["mean"])) or dc > 1e-12 * max(1, abs(v["mean"])) or \
                        (v["n_eligible_episodes"], v["n_total_episodes"]) != (cv["n_eligible_episodes"], cv["n_total_episodes"]):
                    cmp["mismatches"].append((g, t, k, v["mean"], cv["mean"], v["ci95"], cv["ci95"]))
            mine_keys = {alias.get(k, k) for k in res["metrics"]}
            cmp["codex_only"] += [(g, t, k) for k in ct["metrics"] if k not in mine_keys]
    out = {"format": "claude-g2-holdout-audit/1", "script_sha256": A.sha(__file__), "functions_sha256": A.sha(A.__file__),
           "codex_analysis_sha256": A.sha(a.codex), "manifest_sha256": A.sha(a.manifest), "issues": issues,
           "checks": dict(checks), "gt_eligible_by_episode": dict(gt_by_ep), "comparison": cmp, "mine": mine}
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"issues": len(issues), "checks": dict(checks), "gt_by_episode": dict(gt_by_ep),
                      **{k: v for k, v in cmp.items() if k != "mismatches"}, "mismatches": cmp["mismatches"][:10]}, indent=1))


if __name__ == "__main__":
    main()
