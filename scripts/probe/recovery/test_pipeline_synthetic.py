"""Synthetic CPU tests for the recovery pipeline (fake nvidia-smi, throwaway git repo; no GPU, model or simulator).

Includes regressions for the seven Codex R87 counterexamples (cx1..cx7)."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
import validators as V  # noqa: E402
from executor import Executor  # noqa: E402
from ledger import Ledger  # noqa: E402

RESULTS = {}


def check(name, cond):
    RESULTS[name] = bool(cond)
    print(("PASS " if cond else "FAIL ") + name, flush=True)


def fake_smi(d: Path, gpus, apps, fail_gpus=False, fail_apps=False):
    (d / "gpus.txt").write_text("\n".join(f"{i}, {u}, {f}" for i, u, f in gpus) + "\n")
    (d / "apps.txt").write_text("".join(f"{u}, 1234\n" for u in apps))
    smi = d / "nvidia-smi"
    smi.write_text(f"""#!/bin/bash
if [[ "$*" == *query-compute-apps* ]]; then cat {d}/apps.txt; {'exit 9' if fail_apps else 'exit 0'}
else cat {d}/gpus.txt; {'exit 9' if fail_gpus else 'exit 0'}; fi
""")
    smi.chmod(smi.stat().st_mode | stat.S_IEXEC)


def make_repo(tmp: Path) -> tuple[Path, dict]:
    repo = tmp / "repo"
    (repo / "scripts/probe/recovery").mkdir(parents=True)
    shutil.copy(HERE / "envelope.py", repo / "scripts/probe/recovery/envelope.py")
    for c in (["git", "init", "-q"], ["git", "add", "-A"],
              ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "t"]):
        subprocess.run(c, cwd=repo, check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    import hashlib
    ls = subprocess.run(["git", "ls-files", "-s"], cwd=repo, capture_output=True, text=True).stdout
    return repo, {"git_head": head, "tracked_digest": hashlib.sha256(ls.encode()).hexdigest()}


def make_executor(tmp: Path, repo: Path, expect: dict):
    host = {"name": "fake", "ssh": None, "repo": str(repo), "eligible_gpus": [0, 1], "min_free_mib": 26624,
            "policy_backend": "remote_48GB", "probe_root": str(tmp / "probe"), "expect": expect,
            "env": {"OPENWAM_PYTHON": sys.executable, "PATH": os.environ["PATH"]}}
    logs = []
    ex = Executor(tmp / "root", {"remote_reference_once": host}, None, logs.append)
    return ex, host, logs


V_STATE = {"validation_candidate": "remote_reference_once", "candidate_already_attempted": False}


def test_ledger_and_lock():
    with tempfile.TemporaryDirectory() as t:
        L = Ledger(Path(t))
        L.reserve("V", 1800, {"x": 1})
        try:
            L.reserve("V", 10, {}); ok = False
        except RuntimeError:
            ok = True
        check("ledger_refuses_second_allocation", ok)
        check("ledger_restart_with_active_halts", Ledger(Path(t)).state()["halted"])
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        repo, exp = make_repo(tmp)
        fake_smi(tmp, [(0, "GPU-A", 30000)], [])
        old = os.environ["PATH"]; os.environ["PATH"] = f"{tmp}:{old}"
        try:
            ex, host, _ = make_executor(tmp, repo, exp)
            code = (f"import sys; sys.path.insert(0, {str(HERE)!r}); from executor import Executor; "
                    f"Executor(__import__('pathlib').Path({str(tmp / 'root')!r}), {{}}, None, print)")
            r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
            check("cx4_second_controller_process_blocked", r.returncode != 0 and "lock" in (r.stderr + r.stdout))
        finally:
            os.environ["PATH"] = old


def test_invoke_envelope():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        repo, exp = make_repo(tmp)
        fake_smi(tmp, [(0, "GPU-A", 30000), (1, "GPU-B", 47000)], ["GPU-B"])
        old = os.environ["PATH"]; os.environ["PATH"] = f"{tmp}:{old}"
        try:
            ex, host, logs = make_executor(tmp, repo, exp)
            adm = ex.admission(host)
            check("admission_skips_busy_gpu", adm["ok"] and adm["chosen"]["uuid"] == "GPU-A")
            res = ex.invoke("V", host, [sys.executable, "-c", "print('hello')"], 1800, {"t": "ok"}, V_STATE)
            st = ex.ledger.state()
            inv = st["invocations"][-1]
            check("invoke_envelope_runs_settles_verifies", res["status"] == "ran" and res["rc"] == 0
                  and st["active_jobs"] == 0 and inv["cleanup_verified"] and 0 < st["used_card_seconds"]["V"] < 120)
            # orphan child in its own session: killed by ownership-verified cleanup, inside the reserved hard limit
            child = ("import subprocess,sys,time,os;"
                     "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(900)'],start_new_session=True);"
                     "st=open(f'/proc/{p.pid}/stat').read().rsplit(')',1)[1].split()[19];"
                     "open(os.environ['JOB_PGIDS'],'a').write(f'{p.pid} {st}\\n'); time.sleep(900)")
            t0 = time.monotonic()
            res = ex.invoke("V", host, [sys.executable, "-c", child], 230, {"t": "timeout"}, V_STATE)
            took = time.monotonic() - t0
            js = json.loads(Path(res["jobdir"], "status.json").read_text()) if res.get("jobdir") else {}
            gone = all(subprocess.run(["pgrep", "-g", str(g)], capture_output=True).returncode != 0
                       for g in js.get("cleanup", {}).get("groups", []))
            check("timeout_and_orphan_cleanup_within_hard", res["status"] == "ran" and js.get("timed_out") is True
                  and gone and took <= 230 and js["cleanup"]["verified_gone"])
            # code drift: the envelope refuses to start a job on an unexpected tree
            (repo / "scripts/probe/recovery/envelope.py").write_text((repo / "scripts/probe/recovery/envelope.py").read_text() + "\n# drift\n")
            res = ex.invoke("V", host, [sys.executable, "-c", "print(1)"], 1800, {"t": "third"}, V_STATE)
            check("V_invocation_cap_2_enforced", res["status"] == "gate_refused"
                  and "invocation cap reached" in res["verdict"]["reasons"])
            exd = Executor(tmp / "root_drift", {"remote_reference_once": host}, None, logs.append)
            res = exd.invoke("V", host, [sys.executable, "-c", "print(1)"], 1800, {"t": "drift"}, V_STATE)
            js = json.loads(Path(res["jobdir"], "status.json").read_text()) if res.get("jobdir") else {}
            check("envelope_refuses_code_drift", res.get("rc") == "code_mismatch" and js.get("rc") == "code_mismatch")
            subprocess.run(["git", "checkout", "-q", "--", "."], cwd=repo, check=True)
            res = ex.invoke("V", host, [sys.executable, "-c", "print(1)"], 1800, {"t": "bad"},
                            {"validation_candidate": "nope", "candidate_already_attempted": False})
            check("gate_refuses_unapproved_candidate", res["status"] == "gate_refused")
            res = ex.invoke("Q_eng", host, [sys.executable, "-c", "print(1)"], 240, {"t": "nobundle"})
            check("gate_refuses_without_joint_bundle", res["status"] == "gate_refused")
            # cx6: first query fails (exit 9) while the second succeeds -> unknown, no admission
            fake_smi(tmp, [(0, "GPU-A", 30000)], [], fail_gpus=True)
            a = ex.admission(host)
            check("cx6_first_query_failure_is_unknown", a["ok"] is False and a.get("chosen") is None)
            fake_smi(tmp, [(0, "GPU-A", 30000)], [], fail_apps=True)
            check("second_query_failure_is_unknown", ex.admission(host)["ok"] is False)
            fake_smi(tmp, [(0, "GPU-A", 30000), (1, "GPU-B", 47000)], [])
            ex.ledger.update(bound_gpu_uuid="GPU-A")
            a = ex.admission(host)
            check("bound_gpu_never_switched", a["chosen"]["uuid"] == "GPU-A" and all(g["uuid"] == "GPU-A" for g in a["gpus"]))
            fake_smi(tmp, [(0, "GPU-A", 30000), (1, "GPU-B", 47000)], ["GPU-A"])
            check("bound_gpu_busy_means_wait", ex.admission(host)["chosen"] is None)
            ex.ledger.halt("test halt")
            n_before = len(logs)
            res = ex.invoke("V", host, [sys.executable, "-c", "print(1)"], 1800, {"t": "halted"}, V_STATE)
            check("halt_checked_before_any_query", res["status"] == "refused" and len(logs) == n_before)
        finally:
            os.environ["PATH"] = old


def test_wait_slots_persist():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        repo, exp = make_repo(tmp)
        fake_smi(tmp, [(0, "GPU-A", 1000)], [])
        old = os.environ["PATH"]; os.environ["PATH"] = f"{tmp}:{old}"
        try:
            ex, host, logs = make_executor(tmp, repo, exp)
            ex.policy = dict(ex.policy, wait=dict(ex.policy["wait"], maximum_slots=2))
            import executor as E
            real_sleep = E.time.sleep
            E.time.sleep = lambda s: None
            try:
                adm = ex.wait_for_gpu(host)
            finally:
                E.time.sleep = real_sleep
            w = ex.ledger.state()["wait"]
            ex2_slots = Ledger(tmp / "root" / "ledger").state()["wait"]
            check("wait_slots_persist_and_cap", adm is None and w["slots_used"] == 2 and ex2_slots == w
                  and w["last_slot_index"] >= 1)
        finally:
            os.environ["PATH"] = old


def test_transfer_strict():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        repo, exp = make_repo(tmp)
        old = os.environ["PATH"]; fake_smi(tmp, [(0, "GPU-A", 30000)], []); os.environ["PATH"] = f"{tmp}:{old}"
        try:
            ex, host, _ = make_executor(tmp, repo, exp)
            src = tmp / "remote_cell"; (src / "a").mkdir(parents=True)
            (src / "a" / "record.json").write_text("{}")
            ok, info = ex.fetch_verified(host, str(src), tmp / "local_cell")
            check("fetch_verified_ok", ok and info["n_files"] == 1)
            ok2, info2 = ex.fetch_verified(host, str(src), tmp / "local_cell")
            check("fetch_refuses_existing_target", not ok2 and "exists" in info2["error"])
            m = tmp / "m.json"; m.write_text("{\"a\":1}")
            dst = tmp / "remote_manifest.json"; dst.write_text("{\"old\":1}")
            check("push_refuses_existing_remote", ex.push_new(host, m, str(dst)) is False and json.loads(dst.read_text()) == {"old": 1})
            check("push_new_verifies_sha", ex.push_new(host, m, str(tmp / "fresh/manifest.json")))
        finally:
            os.environ["PATH"] = old


def test_validate_V():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        good = {"run_id": "V-001-x", "reference_sha256": V.V_REFERENCE_SHA, "source_trace_sha256": V.V_SOURCE_SHA,
                "code_dirty": False, "engineering_pass": True,
                "runs": [{"bitwise_equal_all": True}, {"bitwise_equal_all": True}]}
        p = tmp / "v.json"; p.write_text(json.dumps(good))
        b = json.loads(Path(V.validate_V(p, "V-001-x", 0, "bm", tmp / "r1.json", fetch_ok=True)["path"]).read_text())
        check("V_fresh_pass_accepted", b["engineering_pass"])
        b = json.loads(Path(V.validate_V(p, "V-001-x", 1, "bm", tmp / "r2.json", fetch_ok=True)["path"]).read_text())
        check("cx5_V_rc1_with_old_pass_rejected", not b["engineering_pass"])
        b = json.loads(Path(V.validate_V(p, "V-002-y", 0, "bm", tmp / "r3.json", fetch_ok=True)["path"]).read_text())
        check("V_stale_run_id_rejected", not b["engineering_pass"])
        b = json.loads(Path(V.validate_V(p, "V-001-x", 0, "bm", tmp / "r4.json", fetch_ok=False)["path"]).read_text())
        check("V_unverified_fetch_rejected", not b["engineering_pass"])
    src = (HERE / "v_backend.py").read_text()
    check("cx7_offload_recipe_removed", "offload_load" not in src and "text_encoder.to(dev)" not in src
          and "offload_text_encoder=False" in src)


def _cand(tmp, seed, verdict, reason=None, rc=None):
    d = tmp / f"cand_{seed}" / "hanging_mug"; d.mkdir(parents=True)
    acc = verdict == "accepted"
    (d / "seed_audit.jsonl").write_text(json.dumps({"seed": seed, "accepted": acc, "reason": "accepted" if acc else reason}) + "\n")
    if acc:
        (d.parent / "manifest.json").write_text(json.dumps({"benchmark": "robotwin", "task": "hanging_mug",
            "mode": "demo_randomized", "entries": [{"episode": 0, "seed": seed, "instructions": {"unseen": ["x"]}}],
            "manifest_hash": "sha256:x"}))
    return {"seed": seed, "dir": str(d.parent), "rc": (0 if acc else 1) if rc is None else rc, "fetch_ok": True, "run_id": f"r{seed}"}


def test_validate_Q():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        c = [_cand(tmp, 200000, "rejected", "expert_plan_fail"), _cand(tmp, 200001, "accepted"),
             _cand(tmp, 200002, "accepted"), _cand(tmp, 200003, "rejected", "unstable"), _cand(tmp, 200004, "accepted")]
        out = tmp / "Q"; out.mkdir()
        rec = V.validate_Q("Q_eng", c, "bm", out, tmp / "r.json")
        b = json.loads(Path(rec["path"]).read_text())
        check("Q_first3_contiguous_sealed", b["engineering_pass"] and b["accepted_scene_seeds"] == [200001, 200002, 200004]
              and [e["episode"] for e in json.loads((out / "manifest.json").read_text())["entries"]] == [0, 1, 2])
        t2 = tmp / "x"; t2.mkdir()
        cx = _cand(t2, 200000, "rejected", "expert_exception")
        check("cx2_expert_exception_is_infra", V.classify_candidate(Path(cx["dir"]), 1, 200000)["verdict"] == "infra")
        check("audit_seed_mismatch_is_infra", V.classify_candidate(Path(c[1]["dir"]), 0, 200002)["verdict"] == "infra")
        check("accepted_with_nonzero_rc_is_infra", V.classify_candidate(Path(c[1]["dir"]), 1, 200001)["verdict"] == "infra")
        b = json.loads(Path(V.validate_Q("Q_eng", c[:1] + [cx], "bm", t2, tmp / "r2.json")["path"]).read_text())
        check("Q_infra_blocks", not b["engineering_pass"] and b["infra_failures"] >= 1)


def test_geometry_and_window():
    pc = types.ModuleType("benchmarks.robotwin.probe_capture"); pc.ProbeCapture = object; pc.task_fields = lambda e: {}
    saved = {k: sys.modules.get(k) for k in ("benchmarks", "benchmarks.robotwin", "benchmarks.robotwin.probe_capture")}
    sys.modules.update({"benchmarks": types.ModuleType("benchmarks"), "benchmarks.robotwin": types.ModuleType("b.r"),
                        "benchmarks.robotwin.probe_capture": pc})
    try:
        spec = importlib.util.spec_from_file_location("rc", REPO / "benchmarks/robotwin/recovery_capture.py")
        rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    tf = lambda x: {"mug_functional_point": [x, 0.0, 0.9], "rack_pose": {"p": [0.0, 0.0, 0.7]},
                    "rack_functional_point": [0.1, 0.0, 0.9], "right_gripper": 1.0}
    check("cx1_official_rack_midpoint_target", rc.mug_geometric_success(tf(0.05)) and not rc.mug_geometric_success(tf(0.1)))
    full = {"complete": True, "termination": "none", "executed_count": 32, "inference_horizon": 32,
            "task_fields": tf(0.5), "post_obs": [{"offset": o, "task_fields": tf(0.5)} for o in range(4, 33, 4)]}
    part = dict(full, executed_count=11, termination="step_lim", post_obs=full["post_obs"][:2])
    check("partial_chunk_is_not_a_window", rc.full_window(full) and not rc.full_window(part))


def _trace(path, ff, noise, final, steps, sched, seeds, clamp_sha=None, proprio=None):
    torch.save({"meta": {"schedule": sched, "video_seed": seeds[0], "action_seed": seeds[1], "replay_mode": None,
                         "clamp_source_sha256": clamp_sha, "clamp_source_path": "x" if clamp_sha else None},
                "cond": {"first_frame_latents": ff, "proprio": proprio if proprio is not None else torch.zeros(3)},
                "init": {"video_noise": noise}, "steps": steps, "final": {"video_latents": final}}, path)


SCHED = [(1000.0, 1000.0), (700.0, 700.0), (357.0, 357.0), (0.0, 0.0)]


def _b2_steps(ff, eps, tgt):
    out = []
    for i, (tv, _) in enumerate(SCHED[:-1]):
        s = tv / 1000.0; z = eps.clone(); z[:, :, :1] = ff
        z[:, :, 1:] = ((1 - s) * tgt[:, :, 1:].float() + s * eps[:, :, 1:].float()).to(eps.dtype)
        out.append({"i": i, "t_v": tv, "video_latents_in": z})
    return out


def test_clamp_strict():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t); g = torch.Generator().manual_seed(1)
        ff = torch.randn(1, 2, 1, 2, 2, generator=g).to(torch.bfloat16)
        eps = torch.randn(1, 2, 3, 2, 2, generator=g).to(torch.bfloat16)
        tgt = torch.cat([ff, torch.randn(1, 2, 2, 2, 2, generator=g).to(torch.bfloat16)], 2)
        _trace(tmp / "ref.pt", ff, eps, tgt, [], SCHED, (1001, 2001))
        rsha = V.sha_file(tmp / "ref.pt")
        ref = torch.load(tmp / "ref.pt", weights_only=False)
        _trace(tmp / "c.pt", ff, eps, tgt, _b2_steps(ff, eps, tgt), SCHED, (3001, 4001), clamp_sha=rsha)
        check("clamp_valid_passes", V.check_clamp(ref, torch.load(tmp / "c.pt", weights_only=False), rsha) == [])
        _trace(tmp / "e.pt", ff, eps, tgt, [], SCHED, (3001, 4001), clamp_sha=rsha)
        check("cx3_empty_steps_rejected", V.check_clamp(ref, torch.load(tmp / "e.pt", weights_only=False), rsha) != [])
        st = _b2_steps(ff, eps, tgt); st[1]["video_latents_in"] = st[1]["video_latents_in"] + 1
        _trace(tmp / "b.pt", ff, eps, tgt, st, SCHED, (3001, 4001), clamp_sha=rsha)
        check("clamp_value_violation_rejected", V.check_clamp(ref, torch.load(tmp / "b.pt", weights_only=False), rsha) != [])
        st = _b2_steps(ff, eps, tgt); st[0]["video_latents_in"] = st[0]["video_latents_in"].float().mul(float("nan")).to(torch.bfloat16)
        _trace(tmp / "n.pt", ff, eps, tgt, st, SCHED, (3001, 4001), clamp_sha=rsha)
        check("clamp_nonfinite_rejected", V.check_clamp(ref, torch.load(tmp / "n.pt", weights_only=False), rsha) != [])


def _episode_fixture(tmp: Path, mutate=None):
    g = torch.Generator().manual_seed(0)
    ff = torch.randn(1, 2, 1, 2, 2, generator=g).to(torch.bfloat16)
    eps = torch.randn(1, 2, 3, 2, 2, generator=g).to(torch.bfloat16)
    tgt = torch.cat([ff, torch.randn(1, 2, 2, 2, 2, generator=g).to(torch.bfloat16)], 2)
    seeds = [200001, 200002, 200004]
    grid = []
    for o in range(3):
        for arm in ("B0", "B1", "B2"):
            cell = tmp / f"local/ep{o:02d}_{arm}"; remote = f"/remote/recovery_M2/ep{o:02d}_{arm}"
            ad = cell / arm; d = ad / "hanging_mug" / f"ep{o:04d}"
            d.mkdir(parents=True)
            trig = 2 if o == 0 else None
            n = 4
            events = [{"chunk": c, "arm": arm, "eligible_trigger": c == trig, "trigger": c == trig,
                       **({"correction_index": 0} if c == trig else {})} for c in range(n)]
            for c in range(n):
                cd = d / f"chunk{c:04d}"; cd.mkdir()
                diverged = trig is not None and c >= trig and arm != "B0"
                rec = {"chunk": c, "env_seed": seeds[o], "actions": [[c, {"B1": 1.0, "B2": 2.0}[arm] if diverged else 0.0]],
                       "task_fields": {"x": c if not (trig is not None and c > trig and arm != "B0") else c + 100},
                       "proprio": [0], "endpose": [0], "pngs": {"head": {"sha256": f"h{c}"}},
                       "post_obs": [{"offset": 4, "task_fields": {"x": c}}], "complete": True, "termination": "none"}
                if c == n - 1:
                    rec.update(termination="step_lim", terminal_task_fields={"x": 9}, terminal_eval_success=False)
                if c == trig and arm != "B0":
                    _trace(cd / "ref.pt", ff, eps, tgt, [], SCHED, (1001, 2001))
                    rsha = V.sha_file(cd / "ref.pt")
                    if arm == "B2":
                        _trace(cd / "cand.pt", ff, eps, tgt, _b2_steps(ff, eps, tgt), SCHED, (3001, 4001), clamp_sha=rsha)
                    else:
                        _trace(cd / "cand.pt", ff, eps, tgt, [], SCHED, (3001, 4001))
                    rp = f"{remote}/{arm}/hanging_mug/ep{o:04d}/chunk{c:04d}"
                    rec["recovery_receipt"] = {"arm": arm, "n_generations": 2,
                                               "ref": {"trace_path": f"{rp}/ref.pt", "sha256": rsha},
                                               "cand": {"trace_path": f"{rp}/cand.pt", "sha256": V.sha_file(cd / "cand.pt")}}
                (cd / "record.json").write_text(json.dumps(rec))
            (d / "recovery_events.json").write_text(json.dumps({"events": events}))
            (ad / "results.json").write_text(json.dumps({"episodes": [{"success": False}]}))
            (cell / "episode_info.json").write_text(json.dumps({"min_free_mib": 3000}))
            grid.append({"ordinal": o, "arm": arm, "seed": seeds[o], "status": "ran", "rc": 0, "run_id": f"M2-{o}{arm}",
                         "dir": str(cell), "fetch_ok": True, "remote_dir": remote})
    if mutate:
        mutate(tmp, grid)
    return {"accepted_scene_seeds": seeds, "manifest_sha256": "m"}, grid


def test_validate_episodes():
    cases = {
        "M2_valid_fixture_passes": (None, True),
        "M2_prefix_break_fails": (lambda tmp, g: _edit(tmp / "local/ep01_B2/B2/hanging_mug/ep0001/chunk0001/record.json", actions=[[9]]), False),
        "M2_trigger_state_diff_fails": (lambda tmp, g: _edit(tmp / "local/ep00_B1/B1/hanging_mug/ep0000/chunk0002/record.json", proprio=[1]), False),
        "M2_env_seed_mismatch_fails": (lambda tmp, g: _edit(tmp / "local/ep02_B0/B0/hanging_mug/ep0002/chunk0000/record.json", env_seed=1), False),
        "M2_duplicate_cell_fails": (lambda tmp, g: g.append(dict(g[0])), False),
        "M2_missing_cell_fails": (lambda tmp, g: g.pop(), False),
        "M2_result_term_inconsistent_fails": (lambda tmp, g: (tmp / "local/ep01_B1/B1/results.json").write_text(json.dumps({"episodes": [{"success": True}]})), False),
    }
    for name, (mut, expect) in cases.items():
        with tempfile.TemporaryDirectory() as t:
            q, grid = _episode_fixture(Path(t), mut)
            b = json.loads(Path(V.validate_episodes("M2", q, grid, "bm", "sci", False, Path(t) / "r.json")["path"]).read_text())
            ok = b["engineering_pass"] is expect
            if name == "M2_valid_fixture_passes":
                ok = ok and b["recovery_invocations_exercised"] == {"B1": 1, "B2": 1}
            if not ok:
                print("   problems:", b["problems"][:3])
            check(name, ok)


def _edit(p: Path, **kv):
    r = json.loads(p.read_text()); r.update(kv); p.write_text(json.dumps(r))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    out = {"n": len(RESULTS), "all_passed": all(RESULTS.values()), "checks": RESULTS}
    print(json.dumps(out))
    raise SystemExit(0 if out["all_passed"] else 1)
