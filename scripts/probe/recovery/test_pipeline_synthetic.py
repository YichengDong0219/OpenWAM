"""Synthetic CPU tests for the recovery pipeline (fake nvidia-smi, throwaway git repos, fake jobs with production
schema; no GPU, model or simulator). Regressions for Codex R87 cx1..cx7 and R88 A..E, plus an end-to-end run through
the real --bundle/--bundle-sha entry (run_pipeline) across V -> Q_eng -> M2 -> Q_eval -> M3."""
from __future__ import annotations

import hashlib
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
import executor as E  # noqa: E402
import validators as V  # noqa: E402
from ledger import Ledger  # noqa: E402

RESULTS = {}
PKG = ["envelope.py", "runtime_identity.py", "test_fake_jobs.py", "trace_plan.json"]


def check(name, cond):
    RESULTS[name] = bool(cond)
    print(("PASS " if cond else "FAIL ") + name, flush=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ fake world
def fake_smi(d: Path, gpus, apps, fail_gpus=False, fail_apps=False, apps_raw=None):
    (d / "gpus.txt").write_text("".join(f"{i}, {u}, {f}\n" for i, u, f in gpus))
    (d / "apps.txt").write_text(apps_raw if apps_raw is not None else "".join(f"{u}, 1234\n" for u in apps))
    smi = d / "nvidia-smi"
    smi.write_text(f"""#!/bin/bash
if [[ "$*" == *query-compute-apps* ]]; then cat {d}/apps.txt; {'exit 9' if fail_apps else 'exit 0'}
else cat {d}/gpus.txt; {'exit 9' if fail_gpus else 'exit 0'}; fi
""")
    smi.chmod(smi.stat().st_mode | stat.S_IEXEC)


def git_commit(repo: Path):
    for c in (["git", "init", "-q"], ["git", "add", "-A"],
              ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "t"]):
        subprocess.run(c, cwd=repo, check=True, capture_output=True)


def build_world(tmp: Path):
    repo, rt = tmp / "repo", tmp / "robotwin"
    (repo / "scripts/probe/recovery").mkdir(parents=True)
    for f in PKG:
        shutil.copy(HERE / f, repo / "scripts/probe/recovery" / f)
    (repo / "openwam").mkdir(); (repo / "openwam/__init__.py").write_text("")
    for d in ("assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboTwin-Full", "assets/video_backbone_ckpt"):
        (repo / d).mkdir(parents=True); (repo / d / "w.bin").write_bytes(b"weights")
    (repo / ".gitignore").write_text("assets/openwam_ckpt\nassets/video_backbone_ckpt\noutputs\n")
    git_commit(repo)
    (rt / "envs").mkdir(parents=True); (rt / "envs/hanging_mug.py").write_text("# task\n")
    (rt / "assets/objects/039_mug").mkdir(parents=True); (rt / "assets/objects/039_mug/m.glb").write_bytes(b"mug")
    (rt / ".gitignore").write_text("assets\n")
    git_commit(rt)
    host = {"name": "fake_remote", "ssh": None, "repo": str(repo), "eligible_gpus": [0, 1], "min_free_mib": 26624,
            "policy_backend": "remote_48GB", "probe_root": str(tmp / "probe"),
            "env": {"OPENWAM_PYTHON": sys.executable, "ROBOTWIN_PYTHON": sys.executable, "ROBOTWIN_PATH": str(rt),
                    "PATH": os.environ["PATH"]}}
    return repo, rt, host


def runtime_expect(host) -> dict:
    env = dict(os.environ, **host["env"])
    r = subprocess.run([sys.executable, "scripts/probe/recovery/runtime_identity.py"], cwd=host["repo"], env=env,
                       capture_output=True, text=True)
    out = json.loads(r.stdout)
    assert out["ok"], out["errors"]
    return {k: v for k, v in out.items() if k not in ("errors", "ok", "format")}


def make_bundle(tmp: Path, host, backend_sha="f" * 64) -> tuple[Path, str]:
    files = ["executor.py", "validators.py", "ledger.py", "gate_codex_v1.py", "policy_codex_R86.json", "envelope.py",
             "runtime_identity.py", "job.py", "trace_plan.json", "scientific_protocol.json"]
    b = {"format": "recovery-joint-root/1", "policy_sha256": E.POLICY_SHA, "reviewers": ["claude", "codex"],
         "bound_files": {str(HERE / f): sha(HERE / f) for f in files},
         "validators": {p: V.validator_sha() for p in E.ORDER}, "selection_algorithm_sha256": V.selection_algorithm_sha(),
         "scientific_bundle_sha256": "s" * 64, "backend_manifest_sha256": backend_sha, "runtime_expect": runtime_expect(host)}
    p = tmp / "bundle.json"; p.write_text(json.dumps(b, sort_keys=True))
    return p, sha(p)


class FakeExecutor(E.Executor):
    fail_m3_rc = 0
    fake_backend_sha = "f" * 64

    def argv_v(self, host):
        return [sys.executable, "scripts/probe/recovery/test_fake_jobs.py", "v", "--out",
                f"{host['probe_root']}/recovery_V/{{RUN_ID}}", "--run-id", "{RUN_ID}"]

    def argv_qualify(self, host, phase, seed, out):
        return [sys.executable, "scripts/probe/recovery/test_fake_jobs.py", "qualify", "--out", out, "--start-seed", str(seed)]

    def argv_episode(self, host, out, arm, manifest, manifest_sha, ordinal, seed):
        extra = ["--fail-rc", str(self.fail_m3_rc)] if (self.fail_m3_rc and "recovery_M3" in out) else []
        return [sys.executable, "scripts/probe/recovery/test_fake_jobs.py", "episode", "--out", out, "--arm", arm,
                "--manifest", manifest, "--expected-manifest-sha", manifest_sha, "--episode", str(ordinal),
                "--seed", str(seed), *extra]

    def backend_manifest(self, host):
        return self.fake_backend_sha, {"ok": True}


def world(tmp: Path, smi_gpus=((0, "GPU-aaaaaaaa-0000", 30000), (1, "GPU-bbbbbbbb-1111", 47000)), smi_apps=("GPU-bbbbbbbb-1111",)):
    repo, rt, host = build_world(tmp)
    fake_smi(tmp, list(smi_gpus), list(smi_apps))
    bpath, bsha = make_bundle(tmp, host)
    logs = []
    ex = FakeExecutor(tmp / "root", {"remote_reference_once": host}, json.loads(bpath.read_text()), bsha, logs.append)
    return ex, ex.host("remote_reference_once"), logs, repo, rt, bpath, bsha


# ------------------------------------------------------------------ tests
def test_end_to_end_bundle_entry():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t); old = os.environ["PATH"]; os.environ["PATH"] = f"{tmp}:{old}"
        E.POLL_S = 1
        try:
            ex, host, logs, *_ = world(tmp)
            t0 = time.time()
            rc = E.run_pipeline(ex, "M3", logs.append)
            st = ex.ledger.state()
            rec = {k: json.loads(Path(v["path"]).read_text()) for k, v in st["receipts"].items()}
            check("e2e_all_stages_auto_advance", rc == 0 and set(rec) == {"V", "Q_eng", "M2", "Q_eval", "M3"}
                  and not st["halted"])
            check("e2e_M2_engineering_pass", rec["M2"]["engineering_pass"] is True
                  and rec["M2"]["recovery_invocations_exercised"] == {"B1": 1, "B2": 1})
            acc = lambda start, n: [x for x in range(start, start + 100) if x % 3 != 0][:n]  # fake rule
            check("e2e_Q_seeds", rec["Q_eng"]["accepted_scene_seeds"] == acc(200000, 3)
                  and rec["Q_eval"]["accepted_scene_seeds"] == acc(300000, 20)
                  and rec["Q_eng"]["candidate_count"] == 4 and rec["Q_eval"]["candidate_count"] == 30)
            check("e2e_M3_full_grid_complete", rec["M3"]["complete_episodes"] == 60 and rec["M3"]["stage_status"] == "complete"
                  and rec["M3"]["problems"] == [])
            check("e2e_gpu_bound_and_accounting", st["bound_gpu_uuid"] == "GPU-aaaaaaaa-0000" and st["active_jobs"] == 0
                  and all(v == 0 for v in st["reserved_card_seconds"].values())
                  and st["attempted_invocations"]["M3"] == 60 and st["attempted_invocations"]["M2"] == 9)
            print(f"   e2e wall {time.time() - t0:.0f}s, invocations {sum(st['attempted_invocations'].values())}")
        finally:
            os.environ["PATH"] = old


def test_m3_process_failure_stops():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t); old = os.environ["PATH"]; os.environ["PATH"] = f"{tmp}:{old}"
        E.POLL_S = 1
        try:
            ex, host, logs, *_ = world(tmp)
            ex.fail_m3_rc = 1
            rc = E.run_pipeline(ex, "M3", logs.append)
            st = ex.ledger.state()
            m3 = json.loads(Path(st["receipts"]["M3"]["path"]).read_text())
            statuses = [g["status"] for g in m3["grid"]]
            check("C_M3_process_failure_stops_all_new_calls", rc == 4 and st["halted"] and st["attempted_invocations"]["M3"] == 1
                  and statuses.count("ran") == 1 and statuses.count("not_run") == 59 and m3["stage_status"] == "incomplete")
        finally:
            os.environ["PATH"] = old


def test_drift_rejected_before_compute():
    for name, mutate in (
        ("E_assume_unchanged_edit_rejected", lambda repo, rt: (
            subprocess.run(["git", "update-index", "--assume-unchanged", "scripts/probe/recovery/test_fake_jobs.py"], cwd=repo, check=True),
            (repo / "scripts/probe/recovery/test_fake_jobs.py").write_text("# changed\n"))),
        ("E_asset_change_rejected", lambda repo, rt: os.utime(repo / "assets/video_backbone_ckpt/w.bin", ns=(1, 1))),
        ("E_robotwin_asset_change_rejected", lambda repo, rt: (rt / "assets/objects/039_mug/m.glb").write_bytes(b"mug2")),
    ):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t); old = os.environ["PATH"]; os.environ["PATH"] = f"{tmp}:{old}"
            try:
                ex, host, logs, repo, rt, *_ = world(tmp)
                mutate(repo, rt)
                rc = E.run_pipeline(ex, "V", logs.append)
                st = ex.ledger.state()
                check(name, rc == 4 and st["halted"] and "runtime identity" in st["halt_reason"]
                      and st["attempted_invocations"]["V"] == 0 and not any(l.get("event") == "launch_check" for l in logs))
            finally:
                os.environ["PATH"] = old
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t); old = os.environ["PATH"]; os.environ["PATH"] = f"{tmp}:{old}"
        try:
            ex, host, logs, *_ = world(tmp)
            ex.fake_backend_sha = "0" * 64
            rc = E.run_pipeline(ex, "V", logs.append)
            st = ex.ledger.state()
            check("E_backend_manifest_drift_rejected_before_V", rc == 4 and st["halted"] and "backend manifest" in st["halt_reason"]
                  and st["attempted_invocations"]["V"] == 0)
            ex2 = FakeExecutor(tmp / "root2", ex.backends, {**ex.bundle, "reviewers": ["claude"]}, "x", logs.append)
            check("bundle_without_both_reviewers_rejected", E.run_pipeline(ex2, "V", logs.append) == 2)
            bad = dict(ex.bundle, bound_files={"relative/path.py": "0" * 64})
            ex3 = FakeExecutor(tmp / "root3", ex.backends, bad, "x", logs.append)
            check("bundle_relative_or_changed_bound_file_rejected", E.run_pipeline(ex3, "V", logs.append) == 2)
        finally:
            os.environ["PATH"] = old


def test_unresolved_allocation_kept():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t); old = os.environ["PATH"]; os.environ["PATH"] = f"{tmp}:{old}"
        E.POLL_S = 1
        try:
            ex, host, logs, *_ = world(tmp)
            real_sh = E.sh
            E.sh = lambda h, cmd, timeout, stdin=None: (subprocess.CompletedProcess([], 0, "", "") if cmd.startswith("cat ")
                                                      else real_sh(h, cmd, timeout, stdin))
            t0 = time.monotonic()
            try:
                res = ex.invoke("V", host, [sys.executable, "-c", "print(1)"], 240, {"t": "lost"},
                                {"validation_candidate": "remote_reference_once", "candidate_already_attempted": False})
            finally:
                E.sh = real_sh
            took = time.monotonic() - t0
            st = ex.ledger.state()
            check("B_missing_status_keeps_reservation_unresolved", res["status"] == "halted" and st["halted"]
                  and st["active_jobs"] == 1 and st["reserved_card_seconds"]["V"] == 240 and "unresolved_allocation" in st)
            check("B_polling_never_exceeds_hard", st["unresolved_allocation"]["observed"]["elapsed_s"] <= 240)
        finally:
            os.environ["PATH"] = old


def test_timeout_orphan_cleanup_within_hard():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t); old = os.environ["PATH"]; os.environ["PATH"] = f"{tmp}:{old}"
        E.POLL_S = 2
        try:
            ex, host, logs, *_ = world(tmp)
            child = ("import subprocess,sys,time,os;"
                     "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(900)'],start_new_session=True);"
                     "st=open(f'/proc/{p.pid}/stat').read().rsplit(')',1)[1].split()[19];"
                     "open(os.environ['JOB_PGIDS'],'a').write(f'{p.pid} {st}\\n'); time.sleep(900)")
            t0 = time.monotonic()
            res = ex.invoke("V", host, [sys.executable, "-c", child], 240, {"t": "timeout"},
                            {"validation_candidate": "remote_reference_once", "candidate_already_attempted": False})
            took = time.monotonic() - t0
            js = json.loads(Path(res["jobdir"], "status.json").read_text())
            gone = all(subprocess.run(["pgrep", "-g", str(g)], capture_output=True).returncode != 0 for g in js["cleanup"]["groups"])
            check("B_timeout_orphan_cleanup_within_hard", res["status"] == "ran" and js["timed_out"] and gone and took <= 240)
        finally:
            os.environ["PATH"] = old


def test_admission_strict():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t); old = os.environ["PATH"]; os.environ["PATH"] = f"{tmp}:{old}"
        try:
            ex, host, logs, *_ = world(tmp)
            g = [(0, "GPU-aaaaaaaa-0000", 30000)]
            for name, kw in (("cx6_first_query_failure_unknown", {"fail_gpus": True}),
                             ("second_query_failure_unknown", {"fail_apps": True}),
                             ("D_malformed_process_row_unknown", {"apps_raw": "malformed process row\n"}),
                             ("D_unknown_uuid_process_row_unknown", {"apps_raw": "GPU-zzzzzzzz-9999, 77\n"}),
                             ("D_bad_pid_unknown", {"apps_raw": "GPU-aaaaaaaa-0000, -3\n"})):
                fake_smi(tmp, g, [], **kw)
                a = ex.admission(host)
                check(name, a["ok"] is False and a.get("chosen") is None)
            fake_smi(tmp, [(0, "GPU-aaaaaaaa-0000", 30000), (0, "GPU-bbbbbbbb-1111", 30000)], [])
            check("D_duplicate_gpu_index_unknown", ex.admission(host)["ok"] is False)
            fake_smi(tmp, g, [])
            check("empty_process_table_is_clear", ex.admission(host)["chosen"]["uuid"] == "GPU-aaaaaaaa-0000")
            fake_smi(tmp, g + [(1, "GPU-bbbbbbbb-1111", 47000)], [])
            ex.ledger.update(bound_gpu_uuid="GPU-aaaaaaaa-0000")
            check("bound_gpu_never_switched", ex.admission(host)["chosen"]["uuid"] == "GPU-aaaaaaaa-0000")
            ex.ledger.halt("t")
            n = len(logs)
            check("halt_checked_before_any_query", ex.invoke("V", host, ["true"], 1800, {}, {})["status"] == "refused" and len(logs) == n)
        finally:
            os.environ["PATH"] = old


def test_lock_and_ledger():
    with tempfile.TemporaryDirectory() as t:
        L = Ledger(Path(t)); L.reserve("V", 1800, {})
        check("ledger_restart_with_active_halts", Ledger(Path(t)).state()["halted"])
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t); old = os.environ["PATH"]; os.environ["PATH"] = f"{tmp}:{old}"
        try:
            ex, host, logs, repo, rt, bpath, bsha = world(tmp)
            code = (f"import sys,json; sys.path.insert(0,{str(HERE)!r}); import executor as E; "
                    f"E.Executor(__import__('pathlib').Path({str(tmp / 'root')!r}), {{}}, {{}}, 'x', print)")
            r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
            check("cx4_second_controller_process_blocked", r.returncode != 0 and "lock" in (r.stdout + r.stderr))
        finally:
            os.environ["PATH"] = old


def test_validate_V():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        good = {"run_id": "V-001-x", "reference_sha256": V.V_REFERENCE_SHA, "source_trace_sha256": V.V_SOURCE_SHA,
                "code_dirty": False, "engineering_pass": True, "runs": [{"bitwise_equal_all": True}] * 2}
        p = tmp / "v.json"; p.write_text(json.dumps(good))
        rd = lambda n, *a, **k: json.loads(Path(V.validate_V(p, *a, tmp / n, **k)["path"]).read_text())["engineering_pass"]
        check("V_fresh_pass_accepted", rd("a", "V-001-x", 0, "bm", fetch_ok=True))
        check("cx5_V_rc1_with_old_pass_rejected", not rd("b", "V-001-x", 1, "bm", fetch_ok=True))
        check("V_stale_run_id_rejected", not rd("c", "V-002", 0, "bm", fetch_ok=True))
        check("V_unverified_fetch_rejected", not rd("d", "V-001-x", 0, "bm", fetch_ok=False))
    src = (HERE / "v_backend.py").read_text()
    check("cx7_offload_recipe_removed", "offload_load" not in src and "offload_text_encoder=False" in src)


def test_Q_classification():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)

        def cand(seed, acc, reason=None, rc=None):
            d = Path(tempfile.mkdtemp(dir=tmp)) / "hanging_mug"; d.mkdir(parents=True)
            (d / "seed_audit.jsonl").write_text(json.dumps({"seed": seed, "accepted": acc, "reason": "accepted" if acc else reason}) + "\n")
            if acc:
                (d.parent / "manifest.json").write_text(json.dumps({"benchmark": "robotwin", "entries": [{"episode": 0, "seed": seed}]}))
            return d.parent, (0 if acc else 1) if rc is None else rc
        for name, args, exp in (("cx2_expert_exception_infra", (200000, False, "expert_exception"), "infra"),
                                ("Q_registered_reject", (200000, False, "expert_plan_fail"), "rejected"),
                                ("Q_unstable_registered_reject", (200000, False, "unstable"), "rejected"),
                                ("Q_accepted", (200000, True), "accepted"),
                                ("Q_accepted_nonzero_rc_infra", (200000, True, None, 1), "infra")):
            d, rc = cand(*args)
            check(name, V.classify_candidate(d, rc, 200000)["verdict"] == exp)
        d, rc = cand(200001, True)
        check("Q_audit_seed_mismatch_infra", V.classify_candidate(d, rc, 200002)["verdict"] == "infra")


def test_geometry_window():
    pc = types.ModuleType("p"); pc.ProbeCapture = object; pc.task_fields = lambda e: {}
    saved = {k: sys.modules.get(k) for k in ("benchmarks", "benchmarks.robotwin", "benchmarks.robotwin.probe_capture")}
    sys.modules.update({"benchmarks": types.ModuleType("b"), "benchmarks.robotwin": types.ModuleType("br"),
                        "benchmarks.robotwin.probe_capture": pc})
    try:
        spec = importlib.util.spec_from_file_location("rc", REPO / "benchmarks/robotwin/recovery_capture.py")
        rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)
    finally:
        for k, v in saved.items():
            sys.modules.pop(k, None) if v is None else sys.modules.__setitem__(k, v)
    tf = lambda x: {"mug_functional_point": [x, 0.0, 0.9], "rack_pose": {"p": [0.0, 0.0, 0.7]},
                    "rack_functional_point": [0.1, 0.0, 0.9], "right_gripper": 1.0}
    check("cx1_official_rack_midpoint_target", rc.mug_geometric_success(tf(0.05)) and not rc.mug_geometric_success(tf(0.1)))
    full = {"complete": True, "termination": "none", "executed_count": 32, "inference_horizon": 32, "task_fields": tf(0.5),
            "post_obs": [{"offset": o, "task_fields": tf(0.5)} for o in range(4, 33, 4)]}
    check("partial_chunk_not_a_window", rc.full_window(full) and not rc.full_window(dict(full, executed_count=11)))


def test_trace_schema_and_clamp():
    sys.path.insert(0, str(HERE))
    import test_fake_jobs as F
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        ff, prop = F.rnd(F.PLAN["cond"]["first_frame_latents"], 1), F.rnd(F.PLAN["cond"]["proprio"], 2)
        rsha = F.trace(tmp / "r.pt", 1001, 2001, ff, prop)
        ref = torch.load(tmp / "r.pt", weights_only=False)
        csha = F.trace(tmp / "c.pt", 3001, 4001, ff, prop, target=ref["final"]["video_latents"], clamp_ref_sha=rsha)
        cand = torch.load(tmp / "c.pt", weights_only=False)
        check("schema_valid_ref_and_cand", V.check_trace(ref, (1001, 2001), "R") == [] and V.check_trace(cand, (3001, 4001), "c") == [])
        check("clamp_valid_passes", V.check_clamp(ref, cand, rsha) == [])
        degenerate = dict(cand, meta=dict(cand["meta"], schedule=[(0.0, 0.0)]), steps=[])
        check("A_degenerate_schedule_empty_steps_rejected", V.check_trace(degenerate, (3001, 4001), "c") != []
              and V.check_clamp(ref, degenerate, rsha) != [])
        trunc = dict(ref, steps=ref["steps"][:3])
        check("A_truncated_reference_rejected", V.check_trace(trunc, (1001, 2001), "R") != [])
        wrong_seed = V.check_trace(cand, (3002, 4002), "c")
        check("A_wrong_seed_rejected", wrong_seed != [])
        bad = dict(cand, steps=[dict(s) for s in cand["steps"]]); bad["steps"][4]["video_latents_in"] = bad["steps"][4]["video_latents_in"] + 1
        check("clamp_value_violation_rejected", V.check_clamp(ref, bad, rsha) != [])
        nanc = dict(cand, final=dict(cand["final"], video_latents=cand["final"]["video_latents"] * float("nan")))
        check("nonfinite_final_rejected", V.check_trace(nanc, (3001, 4001), "c") != [])
        real = torch.load("/newhome/dongyicheng/openwam_local/v_ref/reference_C0_v1001_a2001.pt", weights_only=False)
        real_deg = dict(real, meta=dict(real["meta"], schedule=[(0.0, 0.0)]), steps=[])
        check("A_real_P1A_reference_schema_ok_and_degenerate_rejected",
              V.check_trace(real, (1001, 2001), "P1A") == [] and V.check_trace(real_deg, (1001, 2001), "P1A") != [])


if __name__ == "__main__":
    only = sys.argv[1:]
    for name, fn in list(globals().items()):
        if name.startswith("test_") and (not only or name in only):
            fn()
    out = {"n": len(RESULTS), "all_passed": all(RESULTS.values()), "checks": RESULTS}
    print(json.dumps(out))
    raise SystemExit(0 if out["all_passed"] else 1)
