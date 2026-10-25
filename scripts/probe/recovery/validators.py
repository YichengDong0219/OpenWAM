"""Stage validators (controller side): raw job outputs -> pinned receipts. They NEVER read success/rescue counts to
promote a stage. Every receipt carries validator_sha256, backend_manifest_sha256 and hashes of what it read.
Trace paths inside receipts are host-absolute; they are mapped to the fetched local copy of the SAME cell directory
and SHA-checked against the receipt (receipts themselves are never rewritten).
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
SEED_PARAM = {"Q_eng": 1, "Q_eval": 2}
TARGET = {"Q_eng": 3, "Q_eval": 20}
START = {"Q_eng": 200000, "Q_eval": 300000}
CAND_CAP = {"Q_eng": 30, "Q_eval": 200}
ROTATION = (("B0", "B1", "B2"), ("B1", "B2", "B0"), ("B2", "B0", "B1"))
V_REFERENCE_SHA = "8e124efdf1dc1210c8bdbd81cec2678015d19bff5168c211aa1a82d57780aa38"
V_SOURCE_SHA = "b101730f60261707ba6b5c95b8dddf9fa71f1d81d06bd091de08ff31620e2e20"
REJECT_REASONS = {"expert_plan_fail", "expert_success_false", "unstable"}  # registered before any qualification run
RECOVERY_SEEDS = {0: {"ref": (1001, 2001), "cand": (3001, 4001)}, 1: {"ref": (1002, 2002), "cand": (3002, 4002)}}
MAX_CORRECTIONS, COOLDOWN = 2, 2


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
    if Path(path).exists():
        raise FileExistsError(f"receipt already exists: {path}")
    body = {**body, "validator_sha256": validator_sha()}
    Path(path).write_text(json.dumps(body, indent=1, sort_keys=True, default=str))
    ref = {"path": str(path), "sha256": sha_file(path)}
    if "manifest_sha256" in body:
        ref["manifest_sha256"] = body["manifest_sha256"]
    return ref


# ---------------------------------------------------------------- V
def validate_V(result_path: Path, run_id: str, rc, backend_manifest_sha: str, receipt_path: Path, *, fetch_ok: bool) -> dict:
    problems = []
    r = {}
    if not fetch_ok:
        problems.append("fetch not verified")
    if rc != 0:
        problems.append(f"job rc={rc}")
    if not Path(result_path).exists():
        problems.append("no v_result.json")
    else:
        r = json.loads(Path(result_path).read_text())
        if r.get("run_id") != run_id:
            problems.append(f"result run_id {r.get('run_id')} != {run_id}")
        if r.get("reference_sha256") != V_REFERENCE_SHA or r.get("source_trace_sha256") != V_SOURCE_SHA:
            problems.append("reference/source SHA mismatch")
        if r.get("code_dirty") is not False:
            problems.append("code dirty or unknown")
        runs = r.get("runs", [])
        if len(runs) != 2 or not all(x.get("bitwise_equal_all") is True for x in runs):
            problems.append("not two bitwise-equal runs")
        if r.get("engineering_pass") is not True:
            problems.append("engineering_pass false")
    return write_receipt(receipt_path, {"stage": "V", "engineering_pass": not problems, "problems": problems,
                                        "run_id": run_id, "backend_manifest_sha256": backend_manifest_sha,
                                        "v_result_sha256": sha_file(result_path) if Path(result_path).exists() else None,
                                        "code_commit": r.get("code_commit"), "env": r.get("env"),
                                        "peak_reserved_mib": r.get("peak_reserved_mib")})


# ---------------------------------------------------------------- Q
def classify_candidate(cdir: Path, rc, seed: int) -> dict:
    """accepted / rejected (registered expert verdict) / infra (anything else: exceptions, timeouts, missing rows)."""
    audit = Path(cdir) / "hanging_mug" / "seed_audit.jsonl"
    rows = [json.loads(l) for l in audit.read_text().splitlines() if l.strip()] if audit.exists() else []
    if len(rows) != 1 or rows[0].get("seed") != seed:
        return {"verdict": "infra", "why": f"audit rows {len(rows)} / seed mismatch", "audit": rows}
    row = rows[0]
    manifest = Path(cdir) / "manifest.json"
    if row["accepted"] is True and rc == 0 and manifest.exists():
        m = json.loads(manifest.read_text())
        if len(m.get("entries", [])) == 1 and m["entries"][0]["seed"] == seed:
            return {"verdict": "accepted", "audit": row, "manifest": m}
        return {"verdict": "infra", "why": "manifest does not hold exactly this seed", "audit": row}
    if row["accepted"] is False and row.get("reason") in REJECT_REASONS and rc != 0 and not manifest.exists():
        return {"verdict": "rejected", "audit": row}
    return {"verdict": "infra", "why": f"unregistered outcome rc={rc} reason={row.get('reason')}", "audit": row}


def validate_Q(stage: str, candidates: list[dict], backend_manifest_sha: str, out_dir: Path, receipt_path: Path) -> dict:
    from benchmarks.utils.eval_manifest import seal_manifest, write_manifest
    seeds = [c["seed"] for c in candidates]
    contiguous = seeds == list(range(START[stage], START[stage] + len(seeds)))
    accepted, rows, benchmark = [], [], None
    for c in candidates:
        v = classify_candidate(Path(c["dir"]), c["rc"], c["seed"]) if c.get("fetch_ok") else {"verdict": "infra", "why": "fetch"}
        rows.append({"seed": c["seed"], "run_id": c.get("run_id"), "verdict": v["verdict"],
                     "reason": (v.get("audit") or {}).get("reason") if isinstance(v.get("audit"), dict) else None,
                     "why": v.get("why")})
        if v["verdict"] == "accepted":
            accepted.append(v["manifest"]["entries"][0]); benchmark = v["manifest"]["benchmark"]
    infra = [r for r in rows if r["verdict"] == "infra"]
    ok = contiguous and not infra and len(accepted) == TARGET[stage] and len(candidates) <= CAND_CAP[stage]
    body = {"stage": stage, "engineering_pass": ok, "backend_manifest_sha256": backend_manifest_sha,
            "selection_algorithm_sha256": selection_algorithm_sha(), "candidate_count": len(candidates),
            "candidates": rows, "contiguous": contiguous, "infra_failures": len(infra)}
    if ok:
        entries = [{**e, "episode": i} for i, e in enumerate(accepted)]
        sealed = seal_manifest({"benchmark": benchmark, "task": "hanging_mug", "mode": "demo_randomized", "entries": entries})
        mpath = write_manifest(Path(out_dir) / "manifest.json", sealed)
        order = [list(ROTATION[i % 3]) for i in range(len(entries))]
        (Path(out_dir) / "arm_order.json").write_text(json.dumps(order))
        body.update(accepted_scene_seeds=[e["seed"] for e in entries], manifest_path=str(mpath),
                    manifest_sha256=sha_file(mpath), manifest_hash=sealed["manifest_hash"],
                    arm_order_sha256=sha_file(Path(out_dir) / "arm_order.json"))
    return write_receipt(receipt_path, body)


# ---------------------------------------------------------------- traces
def _finite(t) -> bool:
    return (not torch.is_tensor(t)) or (not t.is_floating_point()) or bool(torch.isfinite(t.float()).all())


PLAN = json.loads((HERE.parent / "trace_plan.json").read_text())


def _sig_ok(t, sig) -> bool:
    return torch.is_tensor(t) and list(t.shape) == sig["shape"] and str(t.dtype) == sig["dtype"] and _finite(t)


def check_trace(tr: dict, seeds: tuple[int, int], label: str) -> list[str]:
    """Full schema check of an online probe trace against the FROZEN plan (not against the trace's own claims)."""
    bad, m = [], tr.get("meta", {})
    if [[float(a), float(b)] for a, b in m.get("schedule", [])] != PLAN["schedule"]:
        bad.append(f"{label}: schedule != frozen plan")
    if (m.get("num_frames"), m.get("action_num_frames")) != (PLAN["num_frames"], PLAN["action_num_frames"]):
        bad.append(f"{label}: frame counts")
    mask = m.get("active_action_mask")
    if mask is None or [bool(x) for x in torch.as_tensor(mask).tolist()] != PLAN["active_action_mask"]:
        bad.append(f"{label}: action mask")
    if (m.get("video_seed"), m.get("action_seed")) != tuple(seeds):
        bad.append(f"{label}: seeds {m.get('video_seed'), m.get('action_seed')} != {seeds}")
    for k, sig in PLAN["cond"].items():
        if not _sig_ok(tr.get("cond", {}).get(k), sig):
            bad.append(f"{label}: cond.{k}")
    for k, sig in PLAN["init"].items():
        if not _sig_ok(tr.get("init", {}).get(k), sig):
            bad.append(f"{label}: init.{k}")
    steps = tr.get("steps") or []
    if [int(x.get("i", -1)) for x in steps] != PLAN["executed_steps"]:
        bad.append(f"{label}: forward steps {[x.get('i') for x in steps]} != {PLAN['executed_steps']}")
    for x in steps:
        i = int(x.get("i", -1))
        if set(PLAN["step_scalar_keys"]) - set(x) or set(PLAN["step_tensors"]) - set(x):
            bad.append(f"{label}: step {i} keys"); continue
        if 0 <= i < len(PLAN["schedule"]) and (float(x["t_v"]), float(x["t_a"])) != tuple(PLAN["schedule"][i]):
            bad.append(f"{label}: step {i} t_v/t_a")
        for k, sig in PLAN["step_tensors"].items():
            if not _sig_ok(x[k], sig):
                bad.append(f"{label}: step {i}.{k}")
    for k, sig in PLAN["final_tensors"].items():
        if not _sig_ok(tr.get("final", {}).get(k), sig):
            bad.append(f"{label}: final.{k}")
    for k, shape in PLAN["final_arrays"].items():
        a = tr.get("final", {}).get(k)
        if a is None or list(torch.as_tensor(a).shape) != shape or not bool(torch.isfinite(torch.as_tensor(a).float()).all()):
            bad.append(f"{label}: final.{k}")
    return bad


def check_clamp(ref: dict, cand: dict, ref_sha: str) -> list[str]:
    """B2 candidate (schema already checked): every planned forward's future = (1-s)*R_final + s*own initial noise,
    frame 0 = own first frame; final future = R future; meta names R's sha."""
    bad = []
    ntt = PLAN["num_train_timesteps"]
    if [int(x["i"]) for x in cand.get("steps", [])] != PLAN["executed_steps"]:
        return ["clamp: candidate forwards do not cover the frozen plan"]
    tgt, eps = ref["final"]["video_latents"], cand["init"]["video_noise"]
    ff = cand["cond"]["first_frame_latents"]
    for x in cand["steps"]:
        z = x["video_latents_in"]
        sigma = float(PLAN["schedule"][int(x["i"])][0]) / ntt
        exp = ((1.0 - sigma) * tgt[:, :, 1:].float() + sigma * eps[:, :, 1:].float()).to(z.dtype)
        if not torch.equal(z[:, :, 1:], exp):
            bad.append(f"future@{x['i']}")
        if not torch.equal(z[:, :, : ff.shape[2]], ff.to(z.dtype)):
            bad.append(f"frame0@{x['i']}")
    fin = cand["final"]["video_latents"]
    if not torch.equal(fin[:, :, 1:], tgt[:, :, 1:].to(fin.dtype)):
        bad.append("final future != target")
    if cand["meta"].get("clamp_source_sha256") != ref_sha:
        bad.append("clamp meta sha")
    return bad


def _load_trace(receipt_part: dict, remote_dir: str, local_dir: str):
    p = str(receipt_part["trace_path"])
    if not p.startswith(remote_dir.rstrip("/") + "/"):
        return None, f"trace path outside cell dir: {p}"
    lp = Path(local_dir) / p[len(remote_dir.rstrip("/")) + 1:]
    if not lp.exists():
        return None, f"missing local copy {lp}"
    if sha_file(lp) != receipt_part["sha256"]:
        return None, f"sha mismatch {lp}"
    return torch.load(lp, map_location="cpu", weights_only=False), None


# ---------------------------------------------------------------- M2 / M3
def _records(arm_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(arm_dir.glob("hanging_mug/ep*/chunk*/record.json"))]


def _events(arm_dir: Path):
    ps = list(arm_dir.glob("hanging_mug/ep*/recovery_events.json"))
    return json.loads(ps[0].read_text()) if len(ps) == 1 else None


def _state(rec: dict) -> dict:
    return {"task_fields": rec.get("task_fields"), "proprio": rec.get("proprio"), "endpose": rec.get("endpose"),
            "pngs": {k: (v or {}).get("sha256") for k, v in (rec.get("pngs") or {}).items()}}


def validate_episodes(stage: str, q_receipt: dict, grid: list[dict], backend_manifest_sha: str,
                      scientific_bundle_sha: str | None, local_backend: bool, receipt_path: Path,
                      stage_stop_reason: str | None = None) -> dict:
    n_seeds = TARGET["Q_eng" if stage == "M2" else "Q_eval"]
    seeds = q_receipt.get("accepted_scene_seeds", [])
    problems, exercised = [], {"B1": 0, "B2": 0}
    keys = [(c["ordinal"], c["arm"]) for c in grid]
    expected = {(o, a) for o in range(n_seeds) for a in ("B0", "B1", "B2")}
    if len(keys) != len(set(keys)) or set(keys) - expected:
        problems.append("grid has duplicate or unexpected cells")
    cells = {(c["ordinal"], c["arm"]): c for c in grid}
    rows = []
    for o in range(n_seeds):
        data = {}
        for arm in ("B0", "B1", "B2"):
            c = cells.get((o, arm))
            row = {"ordinal": o, "arm": arm, "seed": seeds[o] if o < len(seeds) else None,
                   "status": (c or {}).get("status", "not_run"), "run_id": (c or {}).get("run_id")}
            rows.append(row)
            if c is None or c.get("status") != "ran" or c.get("rc") != 0 or not c.get("fetch_ok") or not c.get("dir"):
                problems.append(f"ordinal {o} {arm}: {row['status']} rc={(c or {}).get('rc')} fetch={(c or {}).get('fetch_ok')}")
                continue
            d = Path(c["dir"]) / arm
            recs, ev = _records(d), _events(d)
            res_p = d / "results.json"
            info_p = Path(c["dir"]) / "episode_info.json"
            info = json.loads(info_p.read_text()) if info_p.exists() else {}
            if not recs or ev is None or not res_p.exists():
                problems.append(f"ordinal {o} {arm}: missing records/events/results"); continue
            chunks = [r["chunk"] for r in recs]
            if chunks != list(range(len(recs))):
                problems.append(f"ordinal {o} {arm}: chunk indices {chunks[:5]}.. not contiguous")
            if any(r.get("env_seed") != seeds[o] for r in recs):
                problems.append(f"ordinal {o} {arm}: env_seed != qualified seed {seeds[o]}")
            last = recs[-1]
            term = last.get("termination")
            results = json.loads(res_p.read_text())
            succ = [e.get("success") for e in results.get("episodes", [])]
            if term not in ("success", "step_lim", "plan_fail") or "terminal_task_fields" not in last \
                    or succ != [term == "success"] or last.get("terminal_eval_success") is not (term == "success"):
                problems.append(f"ordinal {o} {arm}: termination/result inconsistent ({term}, {succ})")
            if any(r.get("complete") is not True for r in recs[:-1]):
                problems.append(f"ordinal {o} {arm}: incomplete intermediate chunk")
            if local_backend and (info.get("min_free_mib") is None or info["min_free_mib"] < 2048):
                problems.append(f"ordinal {o} {arm}: local free-memory floor violated ({info.get('min_free_mib')})")
            elig = [x for x in ev["events"] if x.get("eligible_trigger")]
            ch = [x["chunk"] for x in elig]
            if len(elig) > MAX_CORRECTIONS or any(b - a - 1 < COOLDOWN for a, b in zip(ch, ch[1:])) \
                    or [x.get("correction_index") for x in elig] != list(range(len(elig))):
                problems.append(f"ordinal {o} {arm}: correction cap/cooldown/index violated {ch}")
            data[arm] = (recs, ev, c)
        if len(data) != 3:
            continue
        first = {a: next((x["chunk"] for x in data[a][1]["events"] if x.get("eligible_trigger")), None) for a in data}
        if len(set(first.values())) != 1:
            problems.append(f"ordinal {o}: first trigger differs {first}"); continue
        cut = first["B0"]
        lens = {a: len(data[a][0]) for a in data}
        lim = cut if cut is not None else lens["B0"]
        if cut is None and len(set(lens.values())) != 1:
            problems.append(f"ordinal {o}: no trigger but lengths differ {lens}")
        for k in range(min(lim, *lens.values())):
            rs = [data[a][0][k] for a in ("B0", "B1", "B2")]
            if not (rs[0]["actions"] == rs[1]["actions"] == rs[2]["actions"]
                    and _state(rs[0]) == _state(rs[1]) == _state(rs[2])
                    and [p.get("task_fields") for p in rs[0]["post_obs"]] == [p.get("task_fields") for p in rs[1]["post_obs"]]
                    == [p.get("task_fields") for p in rs[2]["post_obs"]]):
                problems.append(f"ordinal {o}: common prefix broken at chunk {k}"); break
        if cut is None:
            terms = {data[a][0][-1].get("termination") for a in data}
            if len(terms) != 1:
                problems.append(f"ordinal {o}: no trigger but terminations differ {terms}")
        else:
            starts = [_state(data[a][0][cut]) if cut < lens[a] else None for a in ("B0", "B1", "B2")]
            evs = [next(x for x in data[a][1]["events"] if x["chunk"] == cut) for a in ("B0", "B1", "B2")]
            strip = lambda e: {k: v for k, v in e.items() if k not in ("arm",)}
            if None in starts or not (starts[0] == starts[1] == starts[2]) or not (strip(evs[0]) == strip(evs[1]) == strip(evs[2])):
                problems.append(f"ordinal {o}: state/decision at trigger chunk {cut} differs across arms")
        refs = {}
        for arm in ("B1", "B2"):
            recs, ev, c = data[arm]
            for x in (e for e in ev["events"] if e.get("eligible_trigger")):
                rec = next((r for r in recs if r["chunk"] == x["chunk"]), None)
                rr = (rec or {}).get("recovery_receipt")
                if not rr or rr.get("arm") != arm or rr.get("n_generations") != 2:
                    problems.append(f"ordinal {o} {arm}: chunk {x['chunk']} lacks a recovery receipt"); continue
                ref, e1 = _load_trace(rr["ref"], c["remote_dir"] + "/" + arm, c["dir"] + "/" + arm)
                cand, e2 = _load_trace(rr["cand"], c["remote_dir"] + "/" + arm, c["dir"] + "/" + arm)
                if e1 or e2:
                    problems.append(f"ordinal {o} {arm}: {e1 or e2}"); continue
                sd = RECOVERY_SEEDS[x["correction_index"]]
                schema = check_trace(ref, sd["ref"], f"{arm} R") + check_trace(cand, sd["cand"], f"{arm} cand")
                if schema:
                    problems.append(f"ordinal {o} {arm}: trace schema {schema[:4]}"); continue
                if not (torch.equal(ref["cond"]["first_frame_latents"], cand["cond"]["first_frame_latents"])
                        and torch.equal(torch.as_tensor(ref["cond"]["proprio"]), torch.as_tensor(cand["cond"]["proprio"]))):
                    problems.append(f"ordinal {o} {arm}: R and candidate conditions differ (state advanced)")
                if not all(_finite(t) for t in (ref["final"]["video_latents"], cand["final"]["video_latents"])):
                    problems.append(f"ordinal {o} {arm}: non-finite latents")
                if arm == "B2":
                    bad = check_clamp(ref, cand, rr["ref"]["sha256"])
                    if bad:
                        problems.append(f"ordinal {o} B2: clamp invariant failed {bad[:4]}")
                elif cand["meta"].get("clamp_source_path") is not None or cand["meta"].get("replay_mode") is not None:
                    problems.append(f"ordinal {o} B1: candidate unexpectedly clamped/replayed")
                exercised[arm] += 1
                if x["correction_index"] == 0:
                    refs[arm] = (ref["final"]["video_latents"], cand["init"]["video_noise"], cand["init"]["action_noise"])
        if "B1" in refs and "B2" in refs:
            if not torch.equal(refs["B1"][0], refs["B2"][0]):
                problems.append(f"ordinal {o}: first reference proposal R differs between B1 and B2")
            if not (torch.equal(refs["B1"][1], refs["B2"][1]) and torch.equal(refs["B1"][2], refs["B2"][2])):
                problems.append(f"ordinal {o}: B1/B2 candidates do not share the frozen candidate noise")
    complete = sum(1 for c in grid if c.get("status") == "ran" and c.get("rc") == 0 and c.get("fetch_ok"))
    body = {"stage": stage, "backend_manifest_sha256": backend_manifest_sha, "scientific_bundle_sha256": scientific_bundle_sha,
            "q_manifest_sha256": q_receipt.get("manifest_sha256"), "complete_episodes": complete, "grid": rows,
            "recovery_invocations_exercised": exercised, "problems": problems,
            "stage_status": "complete" if (stage_stop_reason is None and complete == len(expected)) else "incomplete",
            "stage_stop_reason": stage_stop_reason}
    if stage == "M2":
        body["common_prefix_pass"] = not any(("prefix" in p) or ("trigger" in p and "differs" in p) for p in problems)
        body["invariants_pass"] = not problems
        body["recovery_path_exercised"] = exercised["B1"] > 0 and exercised["B2"] > 0
        body["engineering_pass"] = (complete == 9 and not problems and body["recovery_path_exercised"]
                                    and stage_stop_reason is None)
    else:
        body["engineering_pass"] = None  # M3: descriptive run; this is its all-N completeness/integrity record
    return write_receipt(receipt_path, body)
