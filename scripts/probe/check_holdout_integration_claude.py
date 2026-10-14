"""Claude's independent review of Codex's holdout analysis adapter (g2_holdout_analysis_codex.py, 9834e7f):
an end-to-end synthetic integration. My Q5/Q6 wrappers, in RUN mode with a fake model, write a full
18-observation holdout (real prereg, real G2 freeze and exploration freeze; synthetic selection, holdout
manifest, bindings, traces and records), then Codex's adapter runs UNMODIFIED on those outputs.
No model, no real holdout data, no real effects. Everything lives in a temporary directory.

    python check_holdout_integration_claude.py [--keep]
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import g2_common as C  # noqa: E402
import g2_q5  # noqa: E402
import g2_q6  # noqa: E402
from check_g2_wrappers import FAKE_DESC, FakeRunner  # noqa: E402

PROBE = C.PROBE_ROOT
PREREG = PROBE / "g2_protocol_20261014/G2_PREREG.json"
FREEZE = PROBE / "g2_protocol_20261014/G2_FREEZE_RECEIPT.json"
EXPLORE = PROBE / "g2_explore_freeze_R28/G2_EXPLORE_FREEZE_RECEIPT_R28.json"


class GTRunner(FakeRunner):
    """GT frames whose frame 0 is the fake conditioning canvas, as for real records."""

    def gt_future_frames(self, rec):
        from PIL import Image

        return [Image.new("RGB", (4, 4), (1, 2, 3))] + [Image.new("RGB", (4, 4), (k, 9, 3)) for k in range(1, 9)]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def wj(p, obj):
    Path(p).write_text(json.dumps(obj, indent=1))
    return Path(p)


def build(tmp: Path, gt_every: int):
    pre = json.loads(PREREG.read_text())
    tasks = list(pre["holdout"]["expert_seed_starts"])
    obs = []
    for ti, task in enumerate(tasks):
        for ep in range(3):
            for k in range(2):
                oid = f"{task}_ep{ep:04d}_chunk{k:04d}"
                src = tmp / "src" / f"{oid}.pt"
                src.parent.mkdir(parents=True, exist_ok=True)
                pt = FakeRunner().rerun(None, video_seed=42, action_seed=42)["probe_trace"]
                torch.save({"meta": {"format": "openwam-g0-probe-trace/1", "video_seed": 42, "action_seed": 42,
                                     "num_frames": 9, "action_num_frames": 33, "denoise_steps": 1, "seed": 42},
                            "cond": {"prompt": f"p-{task}"}, "steps": pt["steps"], "final": pt["final"]}, src)
                eligible = (len(obs) % gt_every == 0)
                pngs = [{"offset": off, "camera": cam, "path": f"/fake/{oid}/{off}/{cam}.png", "sha256": f"{off:02d}{cam}"}
                        for off in range(0, 33, 4) for cam in ("head", "left", "right")]
                rec = {"prompt": f"p-{task}", "proprio": [0.5] * 20,
                       "actions": np.random.default_rng(len(obs)).normal(size=(32, 20)).tolist(),
                       "pngs": {p["camera"]: {"path": p["path"], "sha256": p["sha256"]} for p in pngs if p["offset"] == 0},
                       "post_obs": [{"offset": off, "pngs": {p["camera"]: {"path": p["path"], "sha256": p["sha256"]}
                                                             for p in pngs if p["offset"] == off}} for off in range(4, 33, 4)]}
                (tmp / "rec").mkdir(exist_ok=True)
                rp = wj(tmp / "rec" / f"{oid}.json", rec)
                obs.append({"obs_id": oid, "task": task, "episode": ep, "chunk": k, "split": "holdout",
                            "relative_stage": "early" if k == 0 else "late", "trace": str(src), "trace_sha256": sha(src),
                            "record": str(rp), "record_sha256": sha(rp), "pngs": pngs if eligible else pngs[:6],
                            "gt": {"eligible": eligible}})
    sel = wj(tmp / "selection.json", {"format": C.HOLDOUT_SELECTION_FORMAT, "prereg_sha256": sha(PREREG),
                                      "freeze_sha256": sha(FREEZE), "explore_freeze_sha256": sha(EXPLORE),
                                      "observations": obs, "note": "SYNTHETIC integration test only"})
    appr = wj(tmp / "approval.json", {"synthetic": True})
    ap = [{"by": b, "path": str(appr), "sha256": sha(appr)} for b in ("codex", "claude")]
    hm = wj(tmp / "holdout_manifest.json", {"format": C.HOLDOUT_MANIFEST_FORMAT, "status": "frozen",
                                            "prereg_sha256": sha(PREREG), "approvals": ap, "n_observations": len(obs),
                                            "selection": {"path": str(sel), "sha256": sha(sel)}})
    review = wj(tmp / "review.json", {"synthetic": True})
    man = wj(tmp / "manifest.json", {"synthetic": True})
    binds = {}
    for scope in ("Q5_alpha", "Q6_idm"):
        binds[scope] = wj(tmp / f"binding_{scope}.json",
                          {"format": C.BINDING_FORMAT, "scope": scope, "status": "jointly_bound", "prereg_sha256": sha(PREREG),
                           "manifest_path": str(man), "manifest_sha256": sha(man),
                           "reviews": [{"path": str(review), "sha256": sha(review)}],
                           "signatures": {"claude": "synthetic", "codex": "synthetic"}})
    return obs, hm, binds


def run_wrappers(tmp, hm, binds):
    outs = {}
    for name, mod, cls, scope in (("q5", g2_q5, "Q5", "Q5_alpha"), ("q6", g2_q6, "Q6", "Q6_idm")):
        bp = binds[scope]
        saved = (mod.check_binding, mod.check_runtime, mod.check_imports_bound)
        mod.check_binding = lambda *a, _bp=bp, **k: {"manifest": {"artifacts": []},
                                                    "runtime_binding": {"path": str(_bp), "sha256": sha(_bp)},
                                                    "runtime_manifest": {"path": "synthetic", "sha256": "0" * 64, "commit": "x"}}
        mod.check_runtime = lambda runner, manifest: dict(FAKE_DESC)
        mod.check_imports_bound = lambda manifest: None
        try:
            out = tmp / "outputs" / name
            args = argparse.Namespace(prereg=str(PREREG), out=str(out), config=None, n_obs=0, freeze_receipt=str(FREEZE),
                                      runtime_binding=str(bp), split="holdout", explore_freeze=str(EXPLORE),
                                      holdout_manifest=str(hm), selection=None)
            getattr(mod, cls)(args, "run", runner_factory=lambda *_: GTRunner()).run()
            outs[name] = out
        finally:
            mod.check_binding, mod.check_runtime, mod.check_imports_bound = saved
    return outs


def adapter(tmp, hm, binds, outs, tag, optimize=False):
    cmd = [sys.executable] + (["-O"] if optimize else []) + [str(HERE / "g2_holdout_analysis_codex.py"), "--prereg", str(PREREG), "--freeze", str(FREEZE),
           "--explore-freeze", str(EXPLORE), "--holdout-manifest", str(hm), "--q5", str(outs["q5"]), "--q6", str(outs["q6"]),
           "--q5-binding", str(binds["Q5_alpha"]), "--q6-binding", str(binds["Q6_idm"]), "--out", str(tmp / f"analysis_{tag}")]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(HERE))
    last = (p.stderr.strip().splitlines() or [""])[-1]
    return p.returncode, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="_holdout_integration_", dir=PROBE))
    report = {}
    try:
        obs, hm, binds = build(tmp, gt_every=3)
        outs = run_wrappers(tmp, hm, binds)
        report["wrapper_units"] = {k: sum(1 for _ in open(v / "index.jsonl")) for k, v in outs.items()}
        report["q5_extra_files"] = sorted(p.name for p in outs["q5"].iterdir() if p.is_file())
        code, last = adapter(tmp, hm, binds, outs, "as_written")
        report["adapter_as_written"] = {"returncode": code, "last_stderr": last}
        if code == 0:
            res = json.loads((tmp / "analysis_as_written" / "analysis.json").read_text())
            report["adapter_counts"] = res["counts"]
            report["gt_groups_tasks"] = {t: v.get("n_rows") for t, v in res["groups"]["GT_command_consistency"]["tasks"].items()}
        journal = outs["q5"] / "source_checks.jsonl"
        good = journal.read_text()
        lines = good.splitlines()

        def case(tag, text=None, extra_file=None, optimize=False):
            if text is not None:
                journal.write_text(text)
            if extra_file:
                (outs["q5"] / extra_file).write_text('{"key": "x", "error": "synthetic"}\n')
            c, l = adapter(tmp, hm, binds, outs, tag, optimize=optimize)
            journal.write_text(good)
            if extra_file:
                (outs["q5"] / extra_file).unlink()
            return {"returncode": c, "last_stderr": l}

        bad_result = json.loads(lines[3])
        bad_result["result"] = "differs"
        report["neg_wrong_result"] = case("neg_result", "\n".join(lines[:3] + [json.dumps(bad_result)] + lines[4:]) + "\n")
        bad_bind = json.loads(lines[5])
        bad_bind["runtime_binding_sha256"] = "0" * 64
        report["neg_wrong_binding"] = case("neg_bind", "\n".join(lines[:5] + [json.dumps(bad_bind)] + lines[6:]) + "\n")
        report["neg_partial_last_invocation"] = case("neg_partial", good + "\n".join(lines[:5]) + "\n")
        report["pos_two_complete_invocations"] = case("pos_twice", good + good)
        report["neg_failures_file"] = case("neg_failures", extra_file="failures.jsonl")
        report["neg_missing_journal_python_O"] = case("neg_missing_O", text="", optimize=True)
    finally:
        if a.keep:
            print("kept", tmp)
        else:
            shutil.rmtree(tmp)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
