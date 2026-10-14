"""Claude's independent audit of Codex's G2 A3 (R26/R27). Written from the frozen prereg text only;
imports neither Codex's analysis code nor the G2 wrappers or g1_metrics.

Recomputes from the raw formal traces: every pairwise physical metric, direction quantities, exact
denominators, observation -> episode -> task equal weighting, the 10000x whole-episode bootstrap,
the donor equivalence (final actions and every action step tensor), the shared-action-noise premise
of the degeneracy argument, and the GT pairing. Then compares against Codex's analysis.json.

    python g2_audit_claude.py --prereg P --q5 DIR --q6 DIR --codex analysis.json --out audit.json
"""
import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

XYZ = ((0, 3), (10, 13))
ROT = ((3, 9), (13, 19))
GRIP = (9, 19)
FLOOR, DEGEN = 1e-8, 1e-6


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def xyz(a):
    a = np.asarray(a, dtype=np.float64)
    return np.stack([a[:, s:e] for s, e in XYZ], axis=1)  # 32 x 2 x 3


def rotmats(a):
    a = np.asarray(a, dtype=np.float64)
    out, bad = [], []
    for s, e in ROT:
        r6 = a[:, s:e]
        c1, c2 = r6[:, 0:3], r6[:, 3:6]
        n1 = np.linalg.norm(c1, axis=1)
        b1 = c1 / np.maximum(n1, FLOOR)[:, None]
        u2 = c2 - (b1 * c2).sum(1, keepdims=True) * b1
        n2 = np.linalg.norm(u2, axis=1)
        b2 = u2 / np.maximum(n2, FLOOR)[:, None]
        b3 = np.cross(b1, b2)
        out.append(np.stack([b1, b2, b3], axis=2))  # columns
        bad.append((n1 < DEGEN) | (n2 < DEGEN))
    return np.stack(out, axis=1), np.stack(bad, axis=1)  # 32x2x3x3, 32x2


def pair_metrics(A, B):
    dx = np.linalg.norm(xyz(A) - xyz(B), axis=2).reshape(-1)
    Ra, ia = rotmats(A)
    Rb, ib = rotmats(B)
    ok = ~(ia | ib)
    tr = np.einsum("tkij,tkij->tk", Ra, Rb)  # trace(Ra^T Rb)
    ang = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1.0, 1.0)))[ok]
    ga = np.asarray(A, dtype=np.float64)[:, list(GRIP)]
    gb = np.asarray(B, dtype=np.float64)[:, list(GRIP)]
    g = np.abs(ga - gb).reshape(-1)
    return {"D_xyz_m": dx.mean(), "max_xyz_m": dx.max(),
            "rotation_mean_deg": ang.mean() if ang.size else None, "rotation_max_deg": ang.max() if ang.size else None,
            "rotation_invalid_A": float(ia.sum()), "rotation_invalid_B": float(ib.sum()),
            "rotation_valid_comparisons": float(ok.sum()),
            "gripper_mean_abs": g.mean(), "gripper_max_abs": g.max()}


def direction(A_recv, A_donor, A_int):
    d = (xyz(A_donor) - xyz(A_recv)).reshape(-1)
    delta = (xyz(A_int) - xyz(A_recv)).reshape(-1)
    nd = np.linalg.norm(d)
    r_d = nd / 8.0
    if r_d < 0.001:
        return None, r_d
    p = float(delta @ d / (d @ d))
    nde = np.linalg.norm(delta)
    return {"r_d_m": r_d, "p": p, "s_m": float(delta @ d / nd / 8.0), "r_perp_m": float(np.linalg.norm(delta - p * d) / 8.0),
            "cosine": None if nde == 0 else float(delta @ d / (nde * nd)),
            "positive_p": float(p > 0), "negative_p": float(p < 0), "zero_p": float(p == 0)}, r_d


def final(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def step_equal(a, b, keys):
    return len(a) == len(b) and all(torch.equal(s[k], t[k]) for s, t in zip(a, b) for k in keys)


class Groups:
    def __init__(self):
        self.rows = defaultdict(list)  # group -> list of (task, episode, obs, metrics)

    def add(self, group, obs, m):
        self.rows[group].append((obs["task"], int(obs["episode"]), obs["obs_id"], m))


def summarize(rows, obs_all, reps=10000, seed=20261018):
    """pair mean per obs -> equal obs per episode -> equal episode per task; bootstrap whole episodes."""
    out = {}
    tasks = sorted({o["task"] for o in obs_all})
    metrics = sorted({k for _, _, _, m in rows for k in m})
    by_obs = defaultdict(list)
    for t, e, o, m in rows:
        by_obs[(t, e, o)].append(m)
    for task in tasks:
        res = {"n_rows": sum(1 for t, *_ in rows if t == task),
               "n_observations_with_rows": len({o for (t, e, o) in by_obs if t == task}), "metrics": {}}
        for k in metrics:
            obs_means = defaultdict(list)
            for (t, e, o), ms in by_obs.items():
                vals = [m[k] for m in ms if m.get(k) is not None]
                if t == task and vals:
                    obs_means[e].append(np.mean(vals))
            eps = sorted(e for e in {o["episode"] for o in obs_all if o["task"] == task})
            ep_means = [np.mean(obs_means[e]) for e in eps if obs_means[e]]
            if not ep_means:
                res["metrics"][k] = None
                continue
            x = np.asarray(ep_means)
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, len(x), size=(reps, len(x)))
            boot = x[idx].mean(axis=1)
            res["metrics"][k] = {"mean": float(x.mean()), "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                                 "n_eligible_episodes": len(ep_means), "n_total_episodes": len(eps)}
        out[task] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--q5", required=True)
    ap.add_argument("--q6", required=True)
    ap.add_argument("--codex", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pre = json.loads(Path(a.prereg).read_text())
    obs_all = pre["observations"]
    by_id = {o["obs_id"]: o for o in obs_all}
    G = Groups()
    checks = defaultdict(int)
    issues = []

    # ---- Q5 ----
    q5 = [json.loads(l) for l in open(Path(a.q5) / "index.jsonl")]
    base_cache = {}

    def base(path):
        if path not in base_cache:
            base_cache[path] = final(path)
        return base_cache[path]

    for r in q5:
        if sha(r["path"]) != r["sha256"]:
            issues.append(("sha", r["key"]))
        if r["kind"] != "swap":
            continue
        tr = final(r["path"])
        obs = by_id[r["obs_id"]]
        rb, db = tr["meta"]["receiver_baseline"], tr["meta"]["donor_baseline"]
        R, D = base(rb["path"]), base(db["path"])
        if sha(rb["path"]) != rb["sha256"] or sha(db["path"]) != db["sha256"]:
            issues.append(("baseline_sha", r["key"]))
        Ar, Ad, As = (np.asarray(x["final"]["actions_physical"]) for x in (R, D, tr))
        checks["q5_swap"] += 1
        checks["q5_final_equals_donor"] += int(np.array_equal(As, Ad))
        checks["q5_steps_equal_donor"] += int(step_equal(tr["steps"], D["steps"], ("action_latents_in", "action_velocity")))
        checks["q5_receiver_donor_share_action_noise"] += int(torch.equal(R["init"]["action_noise"], D["init"]["action_noise"]))
        checks["q5_swap_noise_is_receiver"] += int(torch.equal(tr["init"]["action_noise"], R["init"]["action_noise"]))
        G.add("Q5_effect", obs, pair_metrics(As, Ar))
        G.add("Q5_natural", obs, pair_metrics(Ad, Ar))
        dm, r_d = direction(Ar, Ad, As)
        G.add("Q5_direction_coverage", obs, {"eligible": float(dm is not None)})
        if dm is not None:
            G.add("Q5_direction", obs, dm)

    # ---- Q6 ----
    q6 = [json.loads(l) for l in open(Path(a.q6) / "index.jsonl")]
    rows6 = defaultdict(dict)
    for r in q6:
        if sha(r["path"]) != r["sha256"]:
            issues.append(("sha", r["key"]))
        rows6[r["kind"]][r["key"]] = r
    baselines = {}
    for k, r in rows6["baseline"].items():
        baselines[(r["obs_id"], tuple(r["receiver"]))] = final(r["path"])
    for k, r in rows6["alt"].items():
        tr = final(r["path"])
        obs = by_id[r["obs_id"]]
        R, D = baselines[(r["obs_id"], tuple(r["receiver"]))], baselines[(r["obs_id"], tuple(r["donor"]))]
        Ar, Ad, Aa = (np.asarray(x["final"]["actions_physical"]) for x in (R, D, tr))
        checks["q6_alt"] += 1
        checks["q6_final_equals_donor"] += int(np.array_equal(Aa, Ad))
        checks["q6_steps_equal_donor"] += int(step_equal(tr["steps"], D["steps"], ("action_latents_in", "action_velocity")))
        checks["q6_receiver_donor_share_action_noise"] += int(torch.equal(R["init"]["action_noise"], D["init"]["action_noise"]))
        checks["q6_cond_is_donor_future"] += int(torch.equal(tr["final"]["cond_video_latents_used"], D["final"]["cond_video_latents_used"]))
        G.add("Q6_effect", obs, pair_metrics(Aa, Ar))
        G.add("Q6_natural", obs, pair_metrics(Ad, Ar))
        dm, _ = direction(Ar, Ad, Aa)
        G.add("Q6_direction_coverage", obs, {"eligible": float(dm is not None)})
        if dm is not None:
            G.add("Q6_direction", obs, dm)
    for o in obs_all:
        for v in pre["video_seeds"]:
            seeds = pre["action_seeds"]
            for i in range(len(seeds)):
                for j in range(i + 1, len(seeds)):
                    A1 = np.asarray(baselines[(o["obs_id"], (v, seeds[i]))]["final"]["actions_physical"])
                    A2 = np.asarray(baselines[(o["obs_id"], (v, seeds[j]))]["final"]["actions_physical"])
                    G.add("Q6_action_scatter", o, pair_metrics(A1, A2))
                    checks["q6_scatter_pairs"] += 1
    gt_obs = set()
    for k, r in rows6["gt"].items():
        tr = final(r["path"])
        obs = by_id[r["obs_id"]]
        gt_obs.add(r["obs_id"])
        cmds = np.asarray(json.loads(Path(obs["record"]).read_text())["actions"], dtype=np.float64)
        Aself = np.asarray(baselines[(r["obs_id"], tuple(r["receiver"]))]["final"]["actions_physical"])
        Agt = np.asarray(tr["final"]["actions_physical"])
        m = pair_metrics(Agt, Aself)
        e_gt, e_self = pair_metrics(Agt, cmds)["D_xyz_m"], pair_metrics(Aself, cmds)["D_xyz_m"]
        m.update(E_gt_m=e_gt, E_self_m=e_self, delta_E_m=e_gt - e_self)
        G.add("GT_command_consistency", obs, m)
        checks["gt_rows"] += 1
    checks["gt_observations"] = len(gt_obs)
    checks["gt_obs_equal_prereg"] = int(gt_obs == set(pre["gt_eligibility"]["eligible_explore_obs_ids"]))

    mine = {g: summarize(rows, obs_all) for g, rows in G.rows.items()}

    # ---- compare with Codex ----
    codex = json.loads(Path(a.codex).read_text())["groups"]
    cmp = {"compared": 0, "max_abs_diff_mean": 0.0, "max_abs_diff_ci": 0.0, "mismatches": []}
    for g, tasks in mine.items():
        for t, res in tasks.items():
            ct = codex.get(g, {}).get("tasks", {}).get(t)
            if ct is None:
                cmp["mismatches"].append((g, t, "missing in codex"))
                continue
            if ct["n_rows"] != res["n_rows"] or ct["n_observations_with_rows"] != res["n_observations_with_rows"]:
                cmp["mismatches"].append((g, t, "counts", res["n_rows"], ct["n_rows"], res["n_observations_with_rows"], ct["n_observations_with_rows"]))
            for k, v in res["metrics"].items():
                cv = ct["metrics"].get(k)
                if v is None or cv is None:
                    if (v is None) != (cv is None or cv.get("mean") is None):
                        cmp["mismatches"].append((g, t, k, "null mismatch"))
                    continue
                cmp["compared"] += 1
                dm = abs(v["mean"] - cv["mean"])
                dc = max(abs(v["ci95"][0] - cv["ci95"][0]), abs(v["ci95"][1] - cv["ci95"][1]))
                cmp["max_abs_diff_mean"] = max(cmp["max_abs_diff_mean"], dm)
                cmp["max_abs_diff_ci"] = max(cmp["max_abs_diff_ci"], dc)
                if dm > 1e-12 * max(1, abs(v["mean"])) or v["n_eligible_episodes"] != cv["n_eligible_episodes"]:
                    cmp["mismatches"].append((g, t, k, "mean/episodes", v["mean"], cv["mean"]))
                if dc > 1e-12 * max(1, abs(v["mean"])):
                    cmp["mismatches"].append((g, t, k, "ci", v["ci95"], cv["ci95"]))
    out = {"format": "claude-g2-a3-audit/1", "script_sha256": sha(__file__), "prereg_sha256": sha(a.prereg),
           "q5_index_sha256": sha(Path(a.q5) / "index.jsonl"), "q6_index_sha256": sha(Path(a.q6) / "index.jsonl"),
           "codex_analysis_sha256": sha(a.codex), "issues": issues, "checks": dict(checks), "comparison": cmp,
           "mine": mine}
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"issues": len(issues), "checks": dict(checks),
                      "compared": cmp["compared"], "mismatches": len(cmp["mismatches"]),
                      "max_abs_diff_mean": cmp["max_abs_diff_mean"], "max_abs_diff_ci": cmp["max_abs_diff_ci"]}, indent=1))


if __name__ == "__main__":
    main()
