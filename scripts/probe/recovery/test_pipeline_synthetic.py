"""Synthetic CPU tests for ledger / executor invoke / validators (fake nvidia-smi; no GPU, no model, no simulator)."""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import validators as V  # noqa: E402
from executor import Executor  # noqa: E402
from ledger import Ledger  # noqa: E402

RESULTS = {}


def check(name, cond):
    RESULTS[name] = bool(cond)
    print(("PASS " if cond else "FAIL ") + name)


def fake_smi(dirpath: Path, gpus, apps):
    """gpus: [(index, uuid, free)], apps: [uuid] busy."""
    (dirpath / "gpus.txt").write_text("\n".join(f"{i}, {u}, {f}" for i, u, f in gpus) + "\n")
    (dirpath / "apps.txt").write_text("\n".join(f"{u}, 1234" for u in apps) + ("\n" if apps else ""))
    smi = dirpath / "nvidia-smi"
    smi.write_text(f"""#!/bin/bash
if [[ "$*" == *query-compute-apps* ]]; then cat {dirpath}/apps.txt; else cat {dirpath}/gpus.txt; fi
""")
    smi.chmod(smi.stat().st_mode | stat.S_IEXEC)


def make_executor(tmp: Path):
    host = {"name": "fake_local", "ssh": None, "repo": str(tmp), "eligible_gpus": [0, 1], "min_free_mib": 22528,
            "policy_backend": "local_3090", "env": {"PATH": f"{tmp}:{os.environ['PATH']}"}, "probe_root": str(tmp / "probe"),
            "residency": "offload"}
    logs = []
    ex = Executor(tmp / "root", {"fake_local": host}, None, logs.append)
    return ex, host, logs


def test_ledger():
    with tempfile.TemporaryDirectory() as t:
        L = Ledger(Path(t))
        L.reserve("V", 1800, {"x": 1})
        st = L.state()
        check("ledger_reserve_counts", st["reserved_card_seconds"]["V"] == 1800 and st["active_jobs"] == 1
              and st["attempted_invocations"]["V"] == 1)
        try:
            L.reserve("V", 10, {}); ok = False
        except RuntimeError:
            ok = True
        check("ledger_refuses_second_allocation", ok)
        L2 = Ledger(Path(t))  # restart while active
        check("ledger_restart_with_active_halts", L2.state()["halted"] and "leftover" in L2.state()["halt_reason"])
        L3 = Ledger(Path(tempfile.mkdtemp()))
        L3.reserve("Q_eng", 240, {}); L3.settle(37.5, {"rc": 0})
        st = L3.state()
        check("ledger_settle_charges_actual", st["used_card_seconds"]["Q_eng"] == 37.5 and st["reserved_card_seconds"]["Q_eng"] == 0
              and st["active_jobs"] == 0 and len(st["invocations"]) == 1)


def test_invoke():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        fake_smi(tmp, [(0, "GPU-A", 23000), (1, "GPU-B", 24000)], ["GPU-B"])
        old = os.environ["PATH"]; os.environ["PATH"] = f"{tmp}:{old}"
        try:
            ex, host, logs = make_executor(tmp)
            adm = ex.admission(host)
            check("admission_skips_busy_gpu", adm["ok"] and adm["chosen"]["uuid"] == "GPU-A")
            res = ex.invoke("V", host, [sys.executable, "-c", "print('hi')"], 1800, {"t": "ok"},
                            {"validation_candidate": "local_3090_offload_once", "candidate_already_attempted": False})
            st = ex.ledger.state()
            check("invoke_runs_and_settles", res["status"] == "ran" and res["rc"] == 0 and st["active_jobs"] == 0
                  and 0 < st["used_card_seconds"]["V"] < 60 and st["reserved_card_seconds"]["V"] == 0)
            res = ex.invoke("V", host, [sys.executable, "-c", "print(1)"], 1800, {"t": "bad cand"},
                            {"validation_candidate": "something_else", "candidate_already_attempted": False})
            check("gate_refuses_unapproved_candidate", res["status"] == "gate_refused")
            res = ex.invoke("Q_eng", host, [sys.executable, "-c", "print(1)"], 240, {"t": "no bundle"})
            check("gate_refuses_without_joint_bundle", res["status"] == "gate_refused"
                  and "missing exact joint pipeline freeze" in res["verdict"]["reasons"])
            # timeout: work budget = requested - grace; a sleeping job must be killed by `timeout`
            out = tmp / "probe" / "inv1"; out.mkdir(parents=True)
            child = (f"import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(600)'],"
                     f"start_new_session=True); open('{out}/pgids.txt','w').write(str(p.pid)+'\\n'); time.sleep(600)")
            t0 = time.monotonic()
            res = ex.invoke("V", host, [sys.executable, "-c", child], 75, {"t": "timeout"},
                            {"validation_candidate": "remote_reference_once", "candidate_already_attempted": False},
                            inv_out=str(out))
            took = time.monotonic() - t0
            pg = int((out / "pgids.txt").read_text())
            alive = subprocess.run(["pgrep", "-g", str(pg)], capture_output=True).returncode == 0
            st = ex.ledger.state()
            print("DEBUG timeout", res, round(took, 1), alive, st["invocations"][-1].get("own_process_clear"), st["halted"])
            check("timeout_kills_and_cleans_orphan_group", res["rc"] == 124 and took < 75 + 30 and not alive
                  and st["invocations"][-1]["own_process_clear"] is True and not st["halted"])
            fake_smi(tmp, [(0, "GPU-A", 1000)], [])
            adm = ex.admission(host)
            check("admission_low_memory_not_chosen", adm["ok"] and adm["chosen"] is None)
            (tmp / "nvidia-smi").write_text("#!/bin/bash\nexit 9\n")
            a2 = ex.admission(host)
            check("admission_query_failure_is_unknown", a2["ok"] is False and a2.get("chosen") is None)
        finally:
            os.environ["PATH"] = old


def test_validate_Q():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t); cands = []
        verdicts = {200000: "rejected", 200001: "accepted", 200002: "accepted", 200003: "rejected", 200004: "accepted"}
        for seed, v in verdicts.items():
            d = tmp / f"cand_{seed}" / "hanging_mug"; d.mkdir(parents=True)
            (d / "seed_audit.jsonl").write_text(json.dumps({"seed": seed, "accepted": v == "accepted",
                                                            "reason": "accepted" if v == "accepted" else "expert_plan_fail"}) + "\n")
            if v == "accepted":
                (d.parent / "manifest.json").write_text(json.dumps({"benchmark": "robotwin", "task": "hanging_mug",
                    "mode": "demo_randomized", "entries": [{"episode": 0, "seed": seed, "instructions": {"unseen": ["x"]}}],
                    "manifest_hash": "sha256:x"}))
            cands.append({"seed": seed, "dir": str(d.parent), "rc": 0 if v == "accepted" else 1})
        out = tmp / "Q"; out.mkdir()
        rec = V.validate_Q("Q_eng", cands, "bm", out, tmp / "r.json")
        body = json.loads(Path(rec["path"]).read_text())
        check("Q_accepts_first3_contiguous", body["engineering_pass"] and body["accepted_scene_seeds"] == [200001, 200002, 200004]
              and rec["manifest_sha256"] == body["manifest_sha256"])
        m = json.loads((out / "manifest.json").read_text())
        check("Q_manifest_renumbered_and_sealed", [e["episode"] for e in m["entries"]] == [0, 1, 2] and m["manifest_hash"].startswith("sha256:"))
        cands2 = cands[:2] + [{"seed": 200002, "dir": str(tmp / "nope"), "rc": 1}]
        body2 = json.loads(Path(V.validate_Q("Q_eng", cands2, "bm", tmp, tmp / "r2.json")["path"]).read_text())
        check("Q_infra_failure_blocks", not body2["engineering_pass"] and body2["infra_failures"] == 1)
        body3 = json.loads(Path(V.validate_Q("Q_eng", [cands[0], cands[2]], "bm", tmp, tmp / "r3.json")["path"]).read_text())
        check("Q_noncontiguous_blocks", not body3["engineering_pass"])


def _trace(path: Path, ff, noise, final, steps, clamp_sha=None):
    torch.save({"meta": {"clamp_source_sha256": clamp_sha, "clamp_source_path": "x" if clamp_sha else None},
                "cond": {"first_frame_latents": ff}, "init": {"video_noise": noise},
                "steps": steps, "final": {"video_latents": final}}, path)


def test_validate_episodes():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        g = torch.Generator().manual_seed(0)
        ff = torch.randn(1, 2, 1, 2, 2, generator=g).to(torch.bfloat16)
        eps = torch.randn(1, 2, 3, 2, 2, generator=g).to(torch.bfloat16)
        tgt = torch.randn(1, 2, 3, 2, 2, generator=g).to(torch.bfloat16)
        episodes = []
        for ordn in range(3):
            for arm in ("B0", "B1", "B2"):
                root = tmp / f"ep{ordn}_{arm}"; d = root / arm / "hanging_mug" / f"ep{ordn:04d}"
                (root / arm).mkdir(parents=True); (root / arm / "results.json").write_text("{}")
                (root / "episode_info.json").write_text(json.dumps({"min_free_mib": 3000}))
                trig = 2 if ordn == 0 else None
                events = [{"chunk": c, "eligible_trigger": c == trig, **({"correction_index": 0} if c == trig else {})}
                          for c in range(4)]
                for c in range(4):
                    cd = d / f"chunk{c:04d}"; cd.mkdir(parents=True)
                    rec = {"chunk": c, "actions": [[c, 0.0]] if (trig is None or c < trig or arm == "B0") else [[c, 1.0 if arm == "B1" else 2.0]],
                           "task_fields": {"x": c}, "complete": True}
                    if c == 3:
                        rec["terminal_task_fields"] = {"x": 3}
                    if c == trig and arm != "B0":
                        refp, candp = cd / "ref.pt", cd / "cand.pt"
                        steps_ref = [{"i": 0, "t_v": 1000.0, "video_latents_in": eps}]
                        _trace(refp, ff, eps, torch.cat([ff, tgt[:, :, 1:]], 2), steps_ref)
                        if arm == "B2":
                            st = []
                            for i, tv in enumerate((1000.0, 700.0, 357.0)):
                                s = tv / 1000.0; z = eps.clone(); z[:, :, :1] = ff
                                z[:, :, 1:] = ((1 - s) * torch.cat([ff, tgt[:, :, 1:]], 2)[:, :, 1:].float() + s * eps[:, :, 1:].float()).to(eps.dtype)
                                st.append({"i": i, "t_v": tv, "video_latents_in": z})
                            _trace(candp, ff, eps, tgt, st, clamp_sha=V.sha_file(refp))
                        else:
                            _trace(candp, ff, eps, tgt, steps_ref)
                        rec["recovery_receipt"] = {"arm": arm, "n_generations": 2,
                                                   "ref": {"trace_path": str(refp), "sha256": V.sha_file(refp)},
                                                   "cand": {"trace_path": str(candp), "sha256": V.sha_file(candp)}}
                    (cd / "record.json").write_text(json.dumps(rec))
                (d / "recovery_events.json").write_text(json.dumps({"events": events}))
                episodes.append({"ordinal": ordn, "arm": arm, "dir": str(root), "rc": 0})
        rec = V.validate_episodes("M2", {"manifest_sha256": "m"}, episodes, "bm", "sci", True, tmp / "m2.json")
        body = json.loads(Path(rec["path"]).read_text())
        check("M2_valid_synthetic_passes", body["engineering_pass"] and body["common_prefix_pass"] and body["invariants_pass"]
              and body["recovery_invocations_exercised"] == {"B1": 1, "B2": 1})
        # break the common prefix in ordinal 1 (no trigger) -> fail
        p = tmp / "ep1_B2" / "B2" / "hanging_mug" / "ep0001" / "chunk0001" / "record.json"
        r = json.loads(p.read_text()); r["actions"] = [[9, 9]]; p.write_text(json.dumps(r))
        body = json.loads(Path(V.validate_episodes("M2", {}, episodes, "bm", "sci", True, tmp / "m2b.json")["path"]).read_text())
        check("M2_prefix_break_fails", not body["engineering_pass"] and not body["common_prefix_pass"])
        r["actions"] = [[1, 0.0]]; p.write_text(json.dumps(r))
        # corrupt the B2 clamp
        cand = tmp / "ep0_B2" / "B2" / "hanging_mug" / "ep0000" / "chunk0002" / "cand.pt"
        c = torch.load(cand, weights_only=False); c["steps"][1]["video_latents_in"] = c["steps"][1]["video_latents_in"] + 1
        torch.save(c, cand)
        recp = cand.parent / "record.json"; rr = json.loads(recp.read_text()); rr["recovery_receipt"]["cand"]["sha256"] = V.sha_file(cand)
        recp.write_text(json.dumps(rr))
        body = json.loads(Path(V.validate_episodes("M2", {}, episodes, "bm", "sci", True, tmp / "m2c.json")["path"]).read_text())
        check("M2_clamp_violation_fails", not body["engineering_pass"] and any("clamp" in x for x in body["problems"]))
        # local memory floor
        (tmp / "ep2_B1" / "episode_info.json").write_text(json.dumps({"min_free_mib": 100}))
        body = json.loads(Path(V.validate_episodes("M2", {}, episodes, "bm", "sci", True, tmp / "m2d.json")["path"]).read_text())
        check("M2_memory_floor_fails", any("free-memory" in x for x in body["problems"]))
        body = json.loads(Path(V.validate_episodes("M2", {}, episodes[:8], "bm", "sci", False, tmp / "m2e.json")["path"]).read_text())
        check("M2_missing_episode_fails", not body["engineering_pass"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    out = {"n": len(RESULTS), "all_passed": all(RESULTS.values()), "checks": RESULTS}
    print(json.dumps(out))
    raise SystemExit(0 if out["all_passed"] else 1)
