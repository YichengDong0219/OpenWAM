"""Synthetic regression for g2_common / g2_q5 / g2_q6 protections (R24 review). CPU only, no model,
no real outputs: a fake runner stands in for OfflineRunner and every directory is temporary.

Covers: a failing identity is never indexed and still fails after a restart; a unit whose file sha
matches but whose stored identity is wrong is rejected; missing RUN_STAMP, duplicate index keys and
orphan files are rejected; an unbound deploy config, a wrong binding prereg and a resolved-config
mismatch are rejected before the runner is constructed.
"""
import argparse
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import g2_common as C  # noqa: E402
import g2_q5  # noqa: E402
import g2_q6  # noqa: E402

FAKE_DESC = {"engine_config_sha256": "0" * 64, "schedule": [[1.0, 1.0], [0.0, 0.0]], "loaded_repo_files": []}


def tensor(seed, shape=(1, 2, 3)):
    g = torch.Generator().manual_seed(int(seed))
    return torch.randn(shape, generator=g).to(torch.bfloat16)


class FakeArch:
    dtype, action_dim, device = torch.bfloat16, 20, "cpu"

    def normalize_deploy_proprio(self, x):
        return torch.as_tensor(np.asarray(x, dtype=np.float32))


FakeArch.__name__ = "FakeIdmArch"


class FakeEngine:
    cfg, _dit_cache, _prompt_embed_cache = None, None, {}

    def __init__(self, runner):
        self.runner = runner

    def generate(self, cond):
        probe, (v, a) = cond["probe"], (cond["probe"]["video_seed"], cond["probe"]["action_seed"])
        tr = self.runner.rerun(None, video_seed=v, action_seed=a)["probe_trace"]
        tr["meta"] = {"video_seed": v, "action_seed": a, "deploy_config_sha256": FAKE_DESC["engine_config_sha256"],
                      "schedule": FAKE_DESC["schedule"]}
        tr["cond"] = {"prompt": cond["prompt"], "image": np.asarray(cond["first_frame_image"][0], dtype=np.uint8),
                      "proprio": torch.as_tensor(np.asarray(cond["proprio"], dtype=np.float32)).to(torch.bfloat16),
                      "first_frame_latents": tensor(7, (1, 2, 1))}
        Path(probe["trace_path"]).parent.mkdir(parents=True, exist_ok=True)
        torch.save(tr, probe["trace_path"])
        return {"actions": tr["final"]["actions_physical"], "probe_receipt": {}}


class FakeRunner:
    """Deterministic stand-in: outputs depend on seeds and (for overrides) on the override tensor."""

    def __init__(self):
        self.arch, self.engine = FakeArch(), FakeEngine(self)
        self.perturb_identity = False

    def precompute_prompts(self, prompts, offload_text_encoder=True):
        return {}

    def canvas_from_pngs(self, pngs, prompt="", state=None):
        return {"image": Image.new("RGB", (4, 4), (1, 2, 3)), "prompt": prompt, "state": state}

    def gt_future_frames(self, rec):
        return [Image.new("RGB", (4, 4), (k, 2, 3)) for k in range(9)]

    def encode_frames(self, frames):
        lat = tensor(11, (1, 2, 3))
        lat[:, :, :1] = tensor(7, (1, 2, 1))
        return lat

    def rerun(self, src, *, video_seed=None, action_seed=None, replay=None, record=True, cond_future_override=None):
        v, a = int(video_seed), int(action_seed)
        fut = tensor(v, (1, 2, 3))
        fut[:, :, :1] = tensor(7, (1, 2, 1))
        cond_used = fut if cond_future_override is None else cond_future_override
        act_seed = a * 1000 + int(cond_used.float().sum().item() * 100)
        if cond_future_override is not None and self.perturb_identity and same_future(cond_future_override, fut):
            act_seed += 1
        acts = tensor(act_seed, (32, 20)).float().numpy()
        steps = [{"i": i, "t_v": 1.0 - i, "t_a": 1.0 - i, "video_latents_in": fut, "action_latents_in": tensor(act_seed + i),
                  "video_velocity": fut, "action_velocity": tensor(act_seed - i)} for i in range(2)]
        pt = {"schedule": FAKE_DESC["schedule"], "active_action_mask": torch.ones(20, dtype=torch.bool),
              "cond": {"first_frame_latents": tensor(7, (1, 2, 1))}, "init": {}, "steps": steps,
              "stage1_steps": [{"i": i, "t_v": 1.0 - i, "video_latents_in": fut, "video_velocity": fut} for i in range(2)],
              "final": {"stage1_future_latents": fut, "cond_video_latents_used": cond_used, "video_latents": fut,
                        "action_latents": tensor(act_seed), "actions_normalized": acts, "actions_physical": acts}}
        return {"actions": acts, "probe_trace": pt}


def same_future(a, b):
    return torch.equal(a, b)


class Base(unittest.TestCase):
    def setUp(self):
        C.PROBE_ROOT.mkdir(parents=True, exist_ok=True)
        self.tmp = Path(tempfile.mkdtemp(prefix="_check_g2_", dir=C.PROBE_ROOT))
        self.patches = [(g2_q5, "describe_runtime"), (g2_q6, "describe_runtime")]
        self.saved = [getattr(m, n) for m, n in self.patches]
        for m, n in self.patches:
            setattr(m, n, lambda runner: dict(FAKE_DESC))

    def tearDown(self):
        for (m, n), f in zip(self.patches, self.saved):
            setattr(m, n, f)
        shutil.rmtree(self.tmp)


class Q6Checks(Base):
    def prereg(self):
        rec = {"prompt": "p", "proprio": [0.5] * 20, "pngs": {}, "post_obs": []}
        rp = self.tmp / "record.json"
        rp.write_text(json.dumps(rec))
        pre = {"format": "openwam-g2-prereg/1", "video_seeds": [1001, 1002], "action_seeds": [2001, 2002],
               "pairs": [], "gt_eligibility": {"eligible_explore_obs_ids": []},
               "observations": [{"obs_id": "o1", "task": "t", "episode": 0, "record": str(rp),
                                 "record_sha256": C.sha256_file(rp), "pngs": []}]}
        p = self.tmp / "prereg.json"
        p.write_text(json.dumps(pre))
        return p

    def q6(self, runner, out="out"):
        args = argparse.Namespace(prereg=str(self.prereg_path), out=str(self.tmp / out), config=None, n_obs=1,
                                  freeze_receipt=None, runtime_binding=None)
        return g2_q6.Q6(args, "smoke", runner_factory=lambda *_: runner)

    def setUp(self):
        super().setUp()
        self.prereg_path = self.prereg()

    def test_clean_smoke(self):
        q = self.q6(FakeRunner())
        q.run()
        self.assertEqual(len(q.ledger.rows), 1 + 4 + 4)

    def test_failed_identity_never_indexed_and_still_fails_after_restart(self):
        r = FakeRunner()
        r.perturb_identity = True
        with self.assertRaises(SystemExit):
            self.q6(r).run()
        out = self.tmp / "out"
        rows = [json.loads(l) for l in (out / "index.jsonl").read_text().splitlines()]
        self.assertFalse([x for x in rows if x["kind"] == "identity"])
        self.assertTrue((out / "failures.jsonl").read_text().strip())
        self.assertFalse(list(out.rglob("identity/*.pt")))
        with self.assertRaises(SystemExit):  # restart: the identity is recomputed and rejected again
            self.q6(r).run()
        r.perturb_identity = False
        self.q6(r).run()

    def test_sha_consistent_but_wrong_identity_rejected(self):
        self.q6(FakeRunner()).run()
        out = self.tmp / "out"
        rows = [json.loads(l) for l in (out / "index.jsonl").read_text().splitlines()]
        row = next(x for x in rows if x["kind"] == "identity")
        tr = torch.load(row["path"], weights_only=False)
        tr["meta"]["obs_id"] = "other"
        torch.save(tr, row["path"])
        row["sha256"] = C.sha256_file(row["path"])
        (out / "index.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows))
        with self.assertRaises(SystemExit) as e:
            self.q6(FakeRunner()).run()
        self.assertIn("expected identity", str(e.exception))

    def test_ledger_stamp_duplicate_orphan(self):
        self.q6(FakeRunner()).run()
        out = self.tmp / "out"
        (out / "stray.pt").write_bytes(b"x")
        with self.assertRaises(SystemExit) as e:
            self.q6(FakeRunner())
        self.assertIn("unindexed", str(e.exception))
        (out / "stray.pt").unlink()
        idx = (out / "index.jsonl").read_text()
        (out / "index.jsonl").write_text(idx + idx.splitlines()[0] + "\n")
        with self.assertRaises(SystemExit) as e:
            self.q6(FakeRunner())
        self.assertIn("duplicate", str(e.exception))
        (out / "index.jsonl").write_text(idx)
        (out / "RUN_STAMP.json").unlink()
        with self.assertRaises(SystemExit) as e:
            self.q6(FakeRunner())
        self.assertIn("no RUN_STAMP", str(e.exception))


class Q5Checks(Base):
    def setUp(self):
        super().setUp()
        root = self.tmp / "g1"
        src = self.tmp / "src.pt"
        torch.save({"meta": {"format": "openwam-g0-probe-trace/1"}, "cond": {"prompt": "p"}}, src)
        rows = []
        for v in (1001, 1002):
            for a in (2001,):
                pt = FakeRunner().rerun(None, video_seed=v, action_seed=a)["probe_trace"]
                tr = {"meta": {"prereg_sha256": "g1" * 32, "source_trace_sha256": C.sha256_file(src),
                               "video_seed": v, "action_seed": a}, "steps": pt["steps"], "final": pt["final"]}
                p = root / "t" / "o1" / "grid" / f"v{v}_a{a}.pt"
                p.parent.mkdir(parents=True, exist_ok=True)
                torch.save(tr, p)
                rows.append({"kind": "grid", "path": str(p), "sha256": C.sha256_file(p)})
        (root / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        pre = {"format": "openwam-g2-prereg/1", "video_seeds": [1001, 1002], "action_seeds": [2001], "pairs": [],
               "provenance": {"g1_prereg": {"sha256": "g1" * 32}},
               "alpha_baseline_sources": {"explore": {"root": str(root), "index_sha256": C.sha256_file(root / "index.jsonl")}},
               "observations": [{"obs_id": "o1", "task": "t", "episode": 0, "trace": str(src),
                                 "trace_sha256": C.sha256_file(src), "source_g1_split": "explore"}]}
        self.prereg_path = self.tmp / "prereg.json"
        self.prereg_path.write_text(json.dumps(pre))

    def q5(self, runner):
        args = argparse.Namespace(prereg=str(self.prereg_path), out=str(self.tmp / "out"), config=None, n_obs=1,
                                  freeze_receipt=None, runtime_binding=None)
        return g2_q5.Q5(args, "smoke", runner_factory=lambda *_: runner)

    def test_identity_failure_and_forged_unit(self):
        class Bad(FakeRunner):
            def rerun(self, *a, **k):
                r = super().rerun(*a, **{x: y for x, y in k.items() if x != "replay"})
                r["probe_trace"]["final"]["actions_physical"] = r["probe_trace"]["final"]["actions_physical"] + 1
                return r

        class Good(FakeRunner):
            def rerun(self, *a, **k):
                return super().rerun(*a, **{x: y for x, y in k.items() if x != "replay"})

        with self.assertRaises(SystemExit):
            self.q5(Bad()).run()
        out = self.tmp / "out"
        self.assertFalse((out / "index.jsonl").exists() and (out / "index.jsonl").read_text().strip())
        with self.assertRaises(SystemExit):
            self.q5(Bad()).run()
        self.q5(Good()).run()
        rows = [json.loads(l) for l in (out / "index.jsonl").read_text().splitlines()]
        tr = torch.load(rows[0]["path"], weights_only=False)
        tr["meta"]["kind"] = "swap"
        torch.save(tr, rows[0]["path"])
        rows[0]["sha256"] = C.sha256_file(rows[0]["path"])
        (out / "index.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows))
        with self.assertRaises(SystemExit) as e:
            self.q5(Good()).run()
        self.assertIn("expected identity", str(e.exception))


class BindingChecks(Base):
    """Rejected before the runner factory is ever called (it would raise AssertionError)."""

    def build(self, *, config=None, bind_prereg=None, resolved_from=None):
        prereg = self.tmp / "prereg.json"
        prereg.write_text(json.dumps({"format": "openwam-g2-prereg/1", "gt_eligibility": {"eligible_explore_obs_ids": []},
                                      "observations": []}))
        real_cfg = (C.REPO / "configs/deploy_probe.yaml").resolve()
        cfg9 = self.tmp / "deploy_9steps.yaml"
        cfg9.write_text(real_cfg.read_text().replace("denoise_steps: 10", "denoise_steps: 9"))
        bound_cfg = real_cfg
        review = self.tmp / "review.json"
        review.write_text("{}")
        snap_from = resolved_from or bound_cfg
        man = {"format": C.MANIFEST_FORMAT, "scope": "Q6_idm", "prereg_sha256": C.sha256_file(prereg),
               "runtime_commit": "x", "checkpoint_dir": str(g2_q6.IDM_CKPT.resolve()), "deploy_config": str(bound_cfg),
               "resolved_config_sha256": C.cfg_sha(C.resolve_config(str(snap_from), str(g2_q6.IDM_CKPT))),
               "artifacts": [{"path": str(bound_cfg), "sha256": C.sha256_file(bound_cfg)}],
               "runtime_description": FAKE_DESC}
        mp = self.tmp / "manifest.json"
        mp.write_text(json.dumps(man))
        bind = {"format": C.BINDING_FORMAT, "scope": "Q6_idm", "status": "jointly_bound",
                "signatures": {"claude": "x", "codex": "y"}, "prereg_sha256": bind_prereg or C.sha256_file(prereg),
                "manifest_path": str(mp), "manifest_sha256": C.sha256_file(mp),
                "reviews": [{"path": str(review), "sha256": C.sha256_file(review)}]}
        bp = self.tmp / "binding.json"
        bp.write_text(json.dumps(bind))
        rec = self.tmp / "receipt.json"
        rec.write_text(json.dumps({"status": "frozen", "prereg_sha256": C.sha256_file(prereg)}))
        return argparse.Namespace(prereg=str(prereg), out=str(self.tmp / "out"), config=str(config or bound_cfg),
                                  n_obs=1, freeze_receipt=str(rec), runtime_binding=str(bp)), cfg9

    def factory(self, *_):
        raise AssertionError("runner constructed despite a bad binding")

    def expect_reject(self, args, msg):
        with self.assertRaises(SystemExit) as e:
            g2_q6.Q6(args, "run", runner_factory=self.factory)
        self.assertIn(msg, str(e.exception))
        self.assertFalse((self.tmp / "out").exists(), "output dir touched before the binding was verified")

    def test_unbound_config(self):
        args, cfg9 = self.build()
        args.config = str(cfg9)
        self.expect_reject(args, "not the bound one")

    def test_wrong_binding_prereg(self):
        args, _ = self.build(bind_prereg="0" * 64)
        self.expect_reject(args, "another prereg")

    def test_resolved_config_mismatch(self):
        args, cfg9 = self.build(resolved_from=self.tmp / "deploy_9steps.yaml")
        self.expect_reject(args, "resolved deploy config differs")

    def test_good_binding_reaches_the_runner(self):
        args, _ = self.build()
        with self.assertRaises(AssertionError):
            g2_q6.Q6(args, "run", runner_factory=self.factory)


class HoldoutChecks(Base):
    """--split holdout: gate before anything else; Q5 makes its own baselines; Q6 GT ids from the selection."""

    def write(self, name, obj):
        p = self.tmp / name
        p.write_text(json.dumps(obj))
        return p

    def setUp(self):
        super().setUp()
        self.prereg_path = self.write("prereg.json", {"format": "openwam-g2-prereg/1", "video_seeds": [1001, 1002],
                                                      "action_seeds": [2001], "pairs": [], "gt_eligibility": {"eligible_explore_obs_ids": []},
                                                      "provenance": {"g1_prereg": {"sha256": "g1" * 32}}, "observations": []})
        psha = C.sha256_file(self.prereg_path)
        appr = self.write("approval.json", {})
        ap = [{"by": b, "path": str(appr), "sha256": C.sha256_file(appr)} for b in ("codex", "claude")]
        self.freeze = self.write("freeze.json", {"format": "openwam-g2-freeze-receipt/1", "status": "frozen",
                                                 "prereg_sha256": psha, "approvals": ap})
        interp = self.write("interp.md", {})
        self.explore = self.write("explore.json", {"format": "openwam-g2-explore-freeze-receipt/1", "status": "frozen",
                                                   "prereg_sha256": psha, "approvals": ap,
                                                   "artifacts": [{"role": r, "path": str(interp), "sha256": C.sha256_file(interp)}
                                                                 for r in ("analysis_script", "interpretation")]})
        src = self.tmp / "src.pt"
        pt = FakeRunner().rerun(None, video_seed=1001, action_seed=2001)["probe_trace"]
        self.src_actions = pt["final"]["actions_physical"]
        torch.save({"meta": {"format": "openwam-g0-probe-trace/1", "video_seed": 1001, "action_seed": 2001},
                    "cond": {"prompt": "p"}, "final": {"actions_physical": self.src_actions}}, src)
        rec = self.write("record.json", {"prompt": "p", "proprio": [0.5] * 20, "pngs": {}, "post_obs": []})
        self.obs = [{"obs_id": "h1", "task": "t", "episode": 0, "split": "holdout", "trace": str(src),
                     "trace_sha256": C.sha256_file(src), "record": str(rec), "record_sha256": C.sha256_file(rec),
                     "pngs": [], "gt": {"eligible": False}}]
        self.selection = self.write("selection.json", {"format": C.HOLDOUT_SELECTION_FORMAT, "prereg_sha256": psha,
                                                       "freeze_sha256": C.sha256_file(self.freeze),
                                                       "explore_freeze_sha256": C.sha256_file(self.explore),
                                                       "observations": self.obs})
        self.ap = ap
        self.manifest = self.write_manifest()

    def write_manifest(self, **over):
        m = {"format": C.HOLDOUT_MANIFEST_FORMAT, "status": "frozen", "prereg_sha256": C.sha256_file(self.prereg_path),
             "approvals": self.ap, "n_observations": len(self.obs),
             "selection": {"path": str(self.selection), "sha256": C.sha256_file(self.selection)}}
        m.update(over)
        return self.write("holdout_manifest.json", m)

    def gate(self):
        return C.check_holdout_gate(self.prereg_path, self.freeze, self.explore, self.manifest)

    def test_gate_accepts_and_rejects(self):
        self.assertEqual(len(self.gate()["observations"]), 1)
        for over, msg in (({"approvals": self.ap[:1]}, "joint approvals"), ({"n_observations": 2}, "count"),
                          ({"status": "draft"}, "not a frozen"), ({"selection": {"path": str(self.selection), "sha256": "0" * 64}}, "differs")):
            self.manifest = self.write_manifest(**over)
            with self.assertRaises(SystemExit) as e:
                self.gate()
            self.assertIn(msg, str(e.exception))
        self.manifest = self.write_manifest()
        sel = json.loads(self.selection.read_text())
        sel["freeze_sha256"] = "0" * 64
        self.selection.write_text(json.dumps(sel))
        self.manifest = self.write_manifest()
        with self.assertRaises(SystemExit) as e:
            self.gate()
        self.assertIn("does not name", str(e.exception))

    def q5(self, runner, mode="smoke"):
        args = argparse.Namespace(prereg=str(self.prereg_path), out=str(self.tmp / "out5"), config=None, n_obs=1,
                                  freeze_receipt=str(self.freeze), runtime_binding=None, split="holdout",
                                  explore_freeze=str(self.explore), holdout_manifest=str(self.manifest), selection=str(self.selection))
        return g2_q5.Q5(args, mode, runner_factory=lambda *_: runner)

    def test_q5_holdout_makes_baselines_then_identity(self):
        class Good(FakeRunner):
            def rerun(self, *a, **k):
                return super().rerun(*a, **{x: y for x, y in k.items() if x != "replay"})

        q = self.q5(Good())
        q.run()
        kinds = sorted(r["kind"] for r in q.ledger.rows.values())
        self.assertEqual(kinds, ["baseline", "baseline", "identity", "identity"])
        stamp = json.loads((self.tmp / "out5" / "RUN_STAMP.json").read_text())
        self.assertEqual(stamp["split"], "holdout")
        self.q5(Good()).run()  # resume re-checks everything, generates nothing new

    def test_q5_holdout_online_source_mismatch_not_indexed(self):
        class Off(FakeRunner):
            def rerun(self, *a, **k):
                r = super().rerun(*a, **{x: y for x, y in k.items() if x != "replay"})
                r["probe_trace"]["final"]["actions_physical"] = r["probe_trace"]["final"]["actions_physical"] + 1
                return r

        with self.assertRaises(SystemExit) as e:
            self.q5(Off()).run()
        self.assertIn("online source", str(e.exception))
        idx = self.tmp / "out5" / "index.jsonl"
        self.assertFalse(idx.exists() and idx.read_text().strip())

    def test_run_gate_rejects_before_runner(self):
        self.manifest = self.write_manifest(status="draft")

        def factory(*_):
            raise AssertionError("runner constructed")

        args = argparse.Namespace(prereg=str(self.prereg_path), out=str(self.tmp / "out6"), config=None, n_obs=1,
                                  freeze_receipt=str(self.freeze), runtime_binding=None, split="holdout",
                                  explore_freeze=str(self.explore), holdout_manifest=str(self.manifest), selection=None)
        saved = g2_q6.check_binding
        g2_q6.check_binding = lambda *a, **k: {"manifest": {}, "runtime_binding": {"sha256": "b" * 64}}
        try:
            with self.assertRaises(SystemExit) as e:
                g2_q6.Q6(args, "run", runner_factory=factory)
            self.assertIn("not a frozen", str(e.exception))
            self.assertFalse((self.tmp / "out6").exists())
        finally:
            g2_q6.check_binding = saved

    def test_q6_holdout_gt_ids_from_selection(self):
        self.obs[0]["gt"]["eligible"] = True
        sel = json.loads(self.selection.read_text())
        sel["observations"] = self.obs
        self.selection.write_text(json.dumps(sel))
        args = argparse.Namespace(prereg=str(self.prereg_path), out=str(self.tmp / "out7"), config=None, n_obs=1,
                                  freeze_receipt=None, runtime_binding=None, split="holdout",
                                  explore_freeze=None, holdout_manifest=None, selection=str(self.selection))
        q = g2_q6.Q6(args, "smoke", runner_factory=lambda *_: FakeRunner())
        self.assertEqual(q.gt_ids, {"h1"})


if __name__ == "__main__":
    unittest.main()
