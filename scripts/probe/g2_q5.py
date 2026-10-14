"""G2 K5: alpha full-trajectory video swap (Q5) on the G2 observations.

For every observation the receiver (v, a) is regenerated while the donor's recorded video latents
replace the receiver's at EVERY denoising step (``OfflineRunner.rerun(replay=donor)``); actions
evolve under the receiver's own action noise, prompt, proprio and first frame. Baselines are the
G1 grid generations (``<g1 root>/<task>/<obs>/grid/v{V}_a{A}.pt``), reused only after the G1 index
sha, the per-file indexed sha, the G1 prereg sha, the source trace sha and the seeds all match.

Before any swap of an observation, all 16 receivers are replayed against THEMSELVES (identity) and
must reproduce the baseline actions and every recorded step tensor bit for bit; a failing identity
is recorded in failures.jsonl, never indexed, and stops the run. Resumed units are re-checked:
stored identity (prereg, binding, observation, seeds, baseline references) and, for identity
units, the bitwise comparison with the baseline again.

Protections (g2_common): in ``run`` mode the binding, manifest, every artifact, the selected deploy
config/checkpoint and the resolved config are verified before the output dir is touched or the model
loaded; after loading, the live runtime must equal the bound description before any generation.

No Q5 effect metric is computed here; the analysis (A3) reads the traces.

    describe : load the model and write the runtime description to --out (for the manifest)
    smoke    : identity replays only on the first N observations (no swap, no binding)
    run      : formal

``--split holdout``: observations come from the jointly frozen holdout manifest (checked with the G2
freeze and the exploration freeze before anything else); the 16 alpha baselines of each new
observation are generated first (the G1 grid procedure: rerun from the source trace, no replay; the
source trace is first regenerated at its own online seeds and must match bit for bit, on every
invocation; a grid cell at those seeds is compared on fresh and resumed runs), then identity and swaps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "probe"))

from g2_common import (Ledger, check_binding, check_imports_bound, check_meta, check_runtime,  # noqa: E402
                       describe_runtime, holdout_observations, same, sha256_file, write_description)

ALPHA_CKPT = REPO / "assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboTwin-Full"
DEFAULT_CONFIG = REPO / "configs/deploy_probe.yaml"
STEP_KEYS = ("t_v", "t_a", "video_latents_in", "action_latents_in", "video_velocity", "action_velocity")
FINAL_KEYS = ("video_latents", "action_latents", "actions_normalized", "actions_physical")
FORMAT = "openwam-g2-q5/2"


def identity_mismatches(base: dict, tr: dict) -> list:
    """Names of every recorded tensor where a self-replay differs from its baseline (empty = bitwise)."""
    if len(base["steps"]) != len(tr["steps"]):
        return ["n_steps"]
    bad = []
    for s, t in zip(base["steps"], tr["steps"]):
        bad += [f"step{s['i']}.{k}" for k in STEP_KEYS if k in s and (k not in t or not same(s[k], t[k]))]
    return bad + [f"final.{k}" for k in FINAL_KEYS if not same(base["final"][k], tr["final"][k])]


def make_runner(ckpt, config):
    from offline_runner import OfflineRunner

    return OfflineRunner(str(ckpt), str(config))


class Q5:
    def __init__(self, args, mode: str, runner_factory=make_runner):
        self.mode = mode
        self.prereg_path = Path(args.prereg).resolve()
        self.prereg = json.loads(self.prereg_path.read_text())
        self.prereg_sha = sha256_file(self.prereg_path)
        if self.prereg.get("format") != "openwam-g2-prereg/1":
            raise SystemExit("not a G2 prereg")
        self.config = Path(args.config or DEFAULT_CONFIG).resolve()
        self.receipts, manifest = {}, None
        if mode == "run":
            rec = json.loads(Path(args.freeze_receipt).read_text())
            if rec.get("status") != "frozen" or rec.get("prereg_sha256") != self.prereg_sha:
                raise SystemExit("G2 freeze receipt does not freeze this prereg")
            bound = check_binding(args.runtime_binding, self.prereg_path, "Q5_alpha", self.config, ALPHA_CKPT)
            manifest = bound.pop("manifest")
            self.receipts = {"freeze_receipt": {"path": str(Path(args.freeze_receipt).resolve()),
                                                "sha256": sha256_file(args.freeze_receipt)}, **bound}
        self.manifest = manifest
        self.binding_sha = self.receipts.get("runtime_binding", {}).get("sha256")
        self.split = getattr(args, "split", "explore") or "explore"
        stamp = {"prereg_sha256": self.prereg_sha, "mode": mode, "runtime_binding_sha256": self.binding_sha}
        if self.split == "holdout":
            # no G1 grid exists for new episodes: the 16 alpha baselines are generated here first
            self.observations, refs = holdout_observations(args, mode, self.prereg_path)
            self.receipts.update(refs)
            self.baseline_index = None
            stamp.update(split="holdout", holdout_refs=refs)
        else:
            self.g1_prereg_sha = self.prereg["provenance"]["g1_prereg"]["sha256"]
            self.baseline_index = self._baseline_indexes()
            obs = self.prereg["observations"]
            self.observations = obs[: args.n_obs] if mode == "smoke" else obs
        for o in self.observations:
            if sha256_file(o["trace"]) != o["trace_sha256"]:
                raise SystemExit(f"source trace sha mismatch for {o['obs_id']}")
        self.ledger = Ledger(args.out, stamp)
        self.runner = runner_factory(ALPHA_CKPT, self.config)
        self.runtime = check_runtime(self.runner, manifest) if mode == "run" else describe_runtime(self.runner)
        from openwam.deploy.engine import _code_version

        self.runtime_code = dict(_code_version())
        if mode == "run" and (self.runtime_code.get("dirty") or not self.runtime_code.get("commit")):
            raise SystemExit(f"formal run needs a clean committed tree; got {self.runtime_code}")
        self._sources = {}
        info = self.runner.precompute_prompts([self.source(o)["cond"]["prompt"] for o in self.observations])
        print(f"[init] {mode}: {len(self.observations)} obs; prompts {info}; code {self.runtime_code}", flush=True)

    def _baseline_indexes(self) -> dict:
        out = {}
        for split, src in self.prereg["alpha_baseline_sources"].items():
            idx = Path(src["root"]) / "index.jsonl"
            if sha256_file(idx) != src["index_sha256"]:
                raise SystemExit(f"G1 {split} index changed")
            for line in idx.read_text().splitlines():
                row = json.loads(line)
                if row["kind"] == "grid":
                    out[row["path"]] = row["sha256"]
        return out

    def source(self, obs: dict) -> dict:
        if obs["obs_id"] not in self._sources:
            from offline_runner import load_trace

            self._sources[obs["obs_id"]] = load_trace(obs["trace"])
        return self._sources[obs["obs_id"]]

    def baseline(self, obs: dict, v: int, a: int) -> dict:
        if self.split == "holdout":
            key = f"baseline|{obs['obs_id']}|{v}-{a}"
            row = self.ledger.get(key)
            if row is None:
                raise RuntimeError(f"holdout baseline not generated yet: {key}")
            return {"trace": self.ledger.load(key), "ref": {"path": row["path"], "sha256": row["sha256"]}}
        root = Path(self.prereg["alpha_baseline_sources"][obs["source_g1_split"]]["root"])
        path = root / obs["task"] / obs["obs_id"] / "grid" / f"v{v}_a{a}.pt"
        want = self.baseline_index.get(str(path))
        if want is None or sha256_file(path) != want:
            raise RuntimeError(f"baseline missing or not matching G1 index: {path}")
        tr = torch.load(path, map_location="cpu", weights_only=False)
        m = tr["meta"]
        if (m.get("prereg_sha256") != self.g1_prereg_sha or m.get("source_trace_sha256") != obs["trace_sha256"]
                or int(m["video_seed"]) != v or int(m["action_seed"]) != a):
            raise RuntimeError(f"baseline identity mismatch: {path}")
        return {"trace": tr, "ref": {"path": str(path), "sha256": want}}

    # -- holdout baselines -------------------------------------------------------------------------
    def source_check(self, obs: dict) -> None:
        """Holdout, every invocation (fresh or resumed): regenerate at the source trace's own online seeds
        (e.g. 42/42, outside the grid), no replay; every step tensor and final action must equal the source."""
        src = self.source(obs)
        sm = src["meta"]
        v, a = int(sm["video_seed"]), int(sm["action_seed"])
        key = f"source_check|{obs['obs_id']}|{v}-{a}"
        pt = self.runner.rerun(src, video_seed=v, action_seed=a, record=True)["probe_trace"]
        bad = identity_mismatches(src, pt)
        if self.mode == "run" and [list(map(float, s)) for s in pt["schedule"]] != self.runtime["schedule"]:
            bad.append("schedule")
        if bad:
            self.ledger.fail(key, f"offline regeneration at the online seeds differs from the source: {bad[:6]}")
            raise SystemExit(f"source check failed for {obs['obs_id']} at seeds {v}/{a}: {bad[:6]}")
        self.ledger.note({"key": key, "obs_id": obs["obs_id"], "source_trace_sha256": obs["trace_sha256"],
                          "seeds": [v, a], "result": "steps_and_final_bitwise_equal",
                          "runtime_binding_sha256": self.binding_sha})

    def make_baselines(self, obs: dict) -> list:
        """Holdout only: the 4x4 grid exactly as G1 made it (rerun from the source trace, no replay), after the
        source check. A grid cell at the online seeds is compared with the source on fresh AND resumed runs."""
        self.source_check(obs)
        src, keys = self.source(obs), []
        sm = src["meta"]
        online_seeds = (int(sm["video_seed"]), int(sm["action_seed"]))
        for v in self.prereg["video_seeds"]:
            for a in self.prereg["action_seeds"]:
                key = f"baseline|{obs['obs_id']}|{v}-{a}"
                path = self.ledger.out / obs["task"] / obs["obs_id"] / "baseline" / f"v{v}_a{a}.pt"
                expect = {"format": FORMAT, "kind": "baseline", "prereg_sha256": self.prereg_sha,
                          "runtime_binding_sha256": self.binding_sha, "obs_id": obs["obs_id"], "task": obs["task"],
                          "episode": obs["episode"], "source_trace_sha256": obs["trace_sha256"],
                          "video_seed": v, "action_seed": a, "donor": None}
                keys.append(key)
                fresh = not self.ledger.get(key)
                if not fresh:
                    tr = self.ledger.load(key)
                    check_meta(tr, expect, key)
                else:
                    res = self.runner.rerun(src, video_seed=v, action_seed=a, record=True)
                    pt = res["probe_trace"]
                    if self.mode == "run" and [list(map(float, s)) for s in pt["schedule"]] != self.runtime["schedule"]:
                        raise SystemExit(f"generation schedule differs from the bound runtime for {key}")
                    tr = {"meta": {**expect, "schedule": pt["schedule"], "active_action_mask": pt["active_action_mask"],
                                   # what a replay of this baseline reads (as in G1 grid traces)
                                   **{k: sm[k] for k in ("num_frames", "action_num_frames", "denoise_steps", "seed")},
                                   "runtime_code": self.runtime_code,
                                   "runtime_engine_config_sha256": self.runtime["engine_config_sha256"],
                                   "receipts": self.receipts, "role": "baseline"},
                          "cond": src["cond"], "init": pt["init"], "steps": pt["steps"],
                          "final": {**pt["final"], "actions_returned": np.asarray(res["actions"]).copy()}}
                if (v, a) == online_seeds:
                    bad = identity_mismatches(src, tr)
                    if bad:
                        if fresh:
                            self.ledger.fail(key, f"baseline at the online seeds differs from the source: {bad[:6]}")
                        raise SystemExit(f"baseline at the online seeds != source trace for {key}: {bad[:6]}")
                if fresh:
                    self.ledger.commit(key, path, tr, kind="baseline", obs_id=obs["obs_id"], task=obs["task"],
                                       episode=obs["episode"], receiver=[v, a], donor=None)
        return keys

    # -- units -----------------------------------------------------------------------------------
    def expect(self, obs, kind, vr, ar, vd, ad, base_r, base_d) -> dict:
        return {"format": FORMAT, "kind": kind, "prereg_sha256": self.prereg_sha, "runtime_binding_sha256": self.binding_sha,
                "obs_id": obs["obs_id"], "task": obs["task"], "episode": obs["episode"],
                "source_trace_sha256": obs["trace_sha256"], "video_seed": vr, "action_seed": ar, "donor": [vd, ad],
                "receiver_baseline": base_r["ref"], "donor_baseline": base_d["ref"]}

    def generate(self, obs, vr, ar, donor_trace) -> tuple[dict, dict]:
        res = self.runner.rerun(self.source(obs), video_seed=vr, action_seed=ar, replay=donor_trace, record=True)
        pt = res["probe_trace"]
        if self.mode == "run" and [list(map(float, s)) for s in pt["schedule"]] != self.runtime["schedule"]:
            raise SystemExit(f"generation schedule differs from the bound runtime for {obs['obs_id']}")
        return res, pt

    def unit(self, obs, kind, key, path, vr, ar, vd, ad) -> None:
        base_r, base_d = self.baseline(obs, vr, ar), self.baseline(obs, vd, ad)
        expect = self.expect(obs, kind, vr, ar, vd, ad, base_r, base_d)
        if self.ledger.get(key):
            tr = self.ledger.load(key)
            check_meta(tr, expect, key)
            if kind == "identity":
                bad = identity_mismatches(base_r["trace"], tr)
                if bad:
                    raise SystemExit(f"stored identity no longer matches its baseline {key}: {bad[:6]}")
            return
        res, pt = self.generate(obs, vr, ar, base_d["trace"])
        tr = {"meta": {**expect, "schedule": pt["schedule"], "active_action_mask": pt["active_action_mask"],
                       "runtime_code": self.runtime_code, "runtime_engine_config_sha256": self.runtime["engine_config_sha256"],
                       "receipts": self.receipts, "role": "identity" if kind == "identity" else "primary_same_action"},
              "cond": self.source(obs)["cond"], "init": pt["init"], "steps": pt["steps"],
              "final": {**pt["final"], "actions_returned": np.asarray(res["actions"]).copy()}}
        if kind == "identity":
            bad = identity_mismatches(base_r["trace"], tr)
            if bad:
                self.ledger.fail(key, f"identity differs from baseline: {bad[:6]}")
                raise SystemExit(f"identity replay differs from baseline {key}: {bad[:6]}")
        self.ledger.commit(key, path, tr, kind=kind, obs_id=obs["obs_id"], task=obs["task"], episode=obs["episode"],
                           receiver=[vr, ar], donor=[vd, ad])

    def keys(self, obs) -> list:
        o, d = obs["obs_id"], self.ledger.out / obs["task"] / obs["obs_id"]
        out = [("identity", f"identity|{o}|{v}-{a}", d / "identity" / f"r{v}-{a}.pt", v, a, v, a)
               for v in self.prereg["video_seeds"] for a in self.prereg["action_seeds"]]
        if self.mode == "run":
            out += [("swap", f"swap|{o}|{p['receiver'][0]}-{p['receiver'][1]}|{p['donor'][0]}-{p['donor'][1]}",
                     d / "swap" / f"r{p['receiver'][0]}-{p['receiver'][1]}_d{p['donor'][0]}-{p['donor'][1]}.pt",
                     *p["receiver"], *p["donor"]) for p in self.prereg["pairs"]]
        return out

    def run(self) -> None:
        expected = []
        for obs in self.observations:
            if self.split == "holdout":
                expected += self.make_baselines(obs)
            for kind, key, path, vr, ar, vd, ad in self.keys(obs):
                self.unit(obs, kind, key, path, vr, ar, vd, ad)
                expected.append(key)
            print(f"[{self.mode}] {obs['obs_id']} done", flush=True)
        self.ledger.require_exact(expected)
        if self.mode == "run":
            check_imports_bound(self.manifest)
        print(f"[{self.mode}] complete: {len(expected)} units", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["run", "smoke", "describe"])
    p.add_argument("--prereg")
    p.add_argument("--out")
    p.add_argument("--freeze-receipt")
    p.add_argument("--runtime-binding")
    p.add_argument("--config", default=None)
    p.add_argument("--n-obs", type=int, default=1, help="smoke only: first N observations")
    p.add_argument("--split", choices=["explore", "holdout"], default="explore")
    p.add_argument("--explore-freeze", help="holdout run: jointly frozen exploration receipt")
    p.add_argument("--holdout-manifest", help="holdout run: jointly frozen holdout manifest receipt")
    p.add_argument("--selection", help="holdout smoke only: a selection file (no gate, no swaps)")
    args = p.parse_args(argv)
    if args.mode == "describe":
        if not args.out:
            raise SystemExit("describe needs --out (a new .json file)")
        write_description(make_runner(ALPHA_CKPT, Path(args.config or DEFAULT_CONFIG).resolve()), args.out)
        return
    if not (args.prereg and args.out):
        raise SystemExit("--prereg and --out are required")
    if args.mode == "run" and not (args.freeze_receipt and args.runtime_binding):
        raise SystemExit("run needs --freeze-receipt and --runtime-binding")
    Q5(args, args.mode).run()


if __name__ == "__main__":
    main()
