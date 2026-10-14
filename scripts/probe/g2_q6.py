"""G2 K8: IDM clean-future injection (Q6) and GT-future command consistency on the G2 observations.

Per observation (inputs rebuilt from the G0 record, not from alpha's trace):
  source    one engine.generate (v1001, a2001) with a probe trace -> meta/cond for every rerun;
            indexed only after its seeds, prompt, conditioning image, proprio and config hash check
  baseline  16 full IDM generations (4 video x 4 action seeds); (1001, 2001) must equal the engine
            source; stage-1 steps and future identical across action seeds at each video seed
  identity  16 reruns with cond_future_override = own stage-1 future; must reproduce the baseline
            stage-2 steps, actions and clean condition bit for bit BEFORE being indexed
  alt       48 primary pairs: receiver (vr, ar) with the donor baseline's (vd, ad) stage-1 future
  gt        GT-eligible observations only: 16 receivers with the executed future (record PNGs at
            offsets 0,4..32 -> server canvas -> IDM VAE), PNG set bound exactly to the prereg
Every override run must leave the receiver's own stage-1 future unchanged. Failures go to
failures.jsonl and are never indexed; resumed units are re-checked (identity, references, and for
identity/gt_future the bitwise conditions again). Protections as in g2_q5 (g2_common).
No Q6 effect metric is computed here; A3 reads the traces.

    describe : load the model and write the runtime description to --out (for the manifest)
    smoke    : source + baseline + identity (+ GT future construction) on the first N observations
    run      : formal
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "probe"))

from g2_common import (Ledger, check_binding, check_imports_bound, check_meta, check_runtime,  # noqa: E402
                       describe_runtime, same, sha256_file, write_description)

IDM_CKPT = REPO / "assets/openwam_ckpt/openwam_study/robotwin_dual_system_idm"
DEFAULT_CONFIG = REPO / "configs/deploy_probe.yaml"
STAGE2_KEYS = ("t_a", "action_latents_in", "action_velocity")
STAGE1_KEYS = ("t_v", "video_latents_in", "video_velocity")
FINAL_KEYS = ("action_latents", "actions_normalized", "actions_physical", "cond_video_latents_used")
SOURCE_SEEDS = (1001, 2001)
FORMAT = "openwam-g2-q6/2"


def override_mismatches(base: dict, tr: dict) -> list:
    if len(base["steps"]) != len(tr["steps"]):
        return ["n_steps"]
    bad = [f"step{s['i']}.{k}" for s, t in zip(base["steps"], tr["steps"]) for k in STAGE2_KEYS
           if k not in t or not same(s[k], t[k])]
    return bad + [f"final.{k}" for k in FINAL_KEYS if not same(base["final"][k], tr["final"][k])]


def array_sha(arr) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(arr)).tobytes()).hexdigest()


def make_runner(ckpt, config):
    from offline_runner import OfflineRunner

    return OfflineRunner(str(ckpt), str(config))


class Q6:
    def __init__(self, args, mode: str, runner_factory=make_runner):
        self.mode = mode
        self.prereg_path = Path(args.prereg).resolve()
        self.prereg = json.loads(self.prereg_path.read_text())
        self.prereg_sha = sha256_file(self.prereg_path)
        if self.prereg.get("format") != "openwam-g2-prereg/1":
            raise SystemExit("not a G2 prereg")
        self.config = Path(args.config or DEFAULT_CONFIG).resolve()
        self.receipts, self.manifest = {}, None
        if mode == "run":
            rec = json.loads(Path(args.freeze_receipt).read_text())
            if rec.get("status") != "frozen" or rec.get("prereg_sha256") != self.prereg_sha:
                raise SystemExit("G2 freeze receipt does not freeze this prereg")
            bound = check_binding(args.runtime_binding, self.prereg_path, "Q6_idm", self.config, IDM_CKPT)
            self.manifest = bound.pop("manifest")
            self.receipts = {"freeze_receipt": {"path": str(Path(args.freeze_receipt).resolve()),
                                                "sha256": sha256_file(args.freeze_receipt)}, **bound}
        self.binding_sha = self.receipts.get("runtime_binding", {}).get("sha256")
        self.gt_ids = set(self.prereg["gt_eligibility"]["eligible_explore_obs_ids"])
        obs = self.prereg["observations"]
        self.observations = obs[: args.n_obs] if mode == "smoke" else obs
        self.records = {}
        for o in self.observations:
            if sha256_file(o["record"]) != o["record_sha256"]:
                raise SystemExit(f"record sha mismatch for {o['obs_id']}")
            self.records[o["obs_id"]] = json.loads(Path(o["record"]).read_text())
        self.ledger = Ledger(args.out, {"prereg_sha256": self.prereg_sha, "mode": mode, "runtime_binding_sha256": self.binding_sha})
        self.runner = runner_factory(IDM_CKPT, self.config)
        if "idm" not in type(self.runner.arch).__name__.lower():
            raise SystemExit(f"checkpoint is not an IDM architecture: {type(self.runner.arch).__name__}")
        self.runtime = check_runtime(self.runner, self.manifest) if mode == "run" else describe_runtime(self.runner)
        from openwam.deploy.engine import _code_version

        self.runtime_code = dict(_code_version())
        if mode == "run" and (self.runtime_code.get("dirty") or not self.runtime_code.get("commit")):
            raise SystemExit(f"formal run needs a clean committed tree; got {self.runtime_code}")
        info = self.runner.precompute_prompts([r["prompt"] for r in self.records.values()])
        print(f"[init] {mode}: {len(self.observations)} obs; prompts {info}; code {self.runtime_code}", flush=True)

    # -- helpers ---------------------------------------------------------------------------------
    def odir(self, obs) -> Path:
        return self.ledger.out / obs["task"] / obs["obs_id"]

    def ref(self, key) -> dict:
        row = self.ledger.get(key)
        return {"path": row["path"], "sha256": row["sha256"]}

    def ident(self, obs, kind, **extra) -> dict:
        return {"format": FORMAT, "kind": kind, "prereg_sha256": self.prereg_sha, "runtime_binding_sha256": self.binding_sha,
                "obs_id": obs["obs_id"], "task": obs["task"], "episode": obs["episode"],
                "record_sha256": obs["record_sha256"], **extra}

    def check_schedule(self, pt, where) -> None:
        if self.mode == "run" and [list(map(float, s)) for s in pt["schedule"]] != self.runtime["schedule"]:
            raise SystemExit(f"generation schedule differs from the bound runtime ({where})")

    def trace(self, expect, pt, actions, keep_stage1) -> dict:
        tr = {"meta": {**expect, "schedule": pt["schedule"], "active_action_mask": pt["active_action_mask"],
                       "runtime_code": self.runtime_code, "runtime_engine_config_sha256": self.runtime["engine_config_sha256"],
                       "receipts": self.receipts},
              "cond": pt["cond"], "init": pt["init"], "steps": pt["steps"],
              "final": {**pt["final"], "actions_returned": np.asarray(actions).copy()}}
        if keep_stage1:
            tr["stage1_steps"] = pt["stage1_steps"]
        return tr

    def commit(self, key, path, tr, obs, kind, receiver, donor) -> None:
        self.ledger.commit(key, path, tr, kind=kind, obs_id=obs["obs_id"], task=obs["task"], episode=obs["episode"],
                           receiver=receiver, donor=donor)

    # -- source ------------------------------------------------------------------------------------
    def canvas(self, obs) -> dict:
        rec = self.records[obs["obs_id"]]
        return self.runner.canvas_from_pngs(rec["pngs"], prompt=rec["prompt"], state=rec.get("proprio"))

    def check_source(self, obs, tr, canvas) -> list:
        m, rec = tr["meta"], self.records[obs["obs_id"]]
        bad = [k for k, ok in (
            ("seeds", (int(m["video_seed"]), int(m["action_seed"])) == SOURCE_SEEDS),
            ("prompt", tr["cond"]["prompt"] == rec["prompt"]),
            ("image", same(np.asarray(canvas["image"].convert("RGB"), dtype=np.uint8), tr["cond"]["image"])),
            ("proprio", same(tr["cond"]["proprio"].reshape(-1),
                             self.runner.arch.normalize_deploy_proprio(rec.get("proprio")).reshape(-1).to(tr["cond"]["proprio"].dtype))),
            ("config", m.get("deploy_config_sha256") == self.runtime["engine_config_sha256"]),
        ) if not ok]
        if self.mode == "run" and [list(map(float, s)) for s in m["schedule"]] != self.runtime["schedule"]:
            bad.append("schedule")
        return bad

    def source(self, obs) -> dict:
        key, path = f"source|{obs['obs_id']}", self.odir(obs) / "source.pt"
        canvas = self.canvas(obs)
        if not self.ledger.get(key):
            v, a = SOURCE_SEEDS
            self.runner.engine.generate({"first_frame_image": [canvas["image"]], "prompt": canvas["prompt"],
                                         "proprio": canvas.get("state"),
                                         "probe": {"trace_path": str(path), "video_seed": v, "action_seed": a}})
            tr = torch.load(path, map_location="cpu", weights_only=False)
            bad = self.check_source(obs, tr, canvas)
            if bad:
                self.ledger.fail(key, f"source checks failed: {bad}")
                raise SystemExit(f"engine source failed its checks for {obs['obs_id']}: {bad}")
            self.ledger.adopt(key, path, kind="source", obs_id=obs["obs_id"], task=obs["task"], episode=obs["episode"],
                              receiver=list(SOURCE_SEEDS), donor=None)
            return tr
        tr = self.ledger.load(key)
        bad = self.check_source(obs, tr, canvas)
        if bad:
            raise SystemExit(f"stored source no longer passes its checks for {obs['obs_id']}: {bad}")
        return tr

    # -- baselines -------------------------------------------------------------------------------
    def baselines(self, obs, src) -> dict:
        src_ref, out = self.ref(f"source|{obs['obs_id']}"), {}
        for v in self.prereg["video_seeds"]:
            for a in self.prereg["action_seeds"]:
                key, path = f"baseline|{obs['obs_id']}|{v}-{a}", self.odir(obs) / "baseline" / f"v{v}_a{a}.pt"
                expect = self.ident(obs, "baseline", video_seed=v, action_seed=a, donor=None, source=src_ref)
                if self.ledger.get(key):
                    tr = self.ledger.load(key)
                    check_meta(tr, expect, key)
                else:
                    res = self.runner.rerun(src, video_seed=v, action_seed=a)
                    pt = res["probe_trace"]
                    self.check_schedule(pt, key)
                    tr = self.trace(expect, pt, res["actions"], keep_stage1=True)
                    if (v, a) == SOURCE_SEEDS and not same(pt["final"]["actions_physical"], src["final"]["actions_physical"]):
                        self.ledger.fail(key, "offline baseline != engine source")
                        raise SystemExit(f"offline baseline != engine source for {obs['obs_id']}")
                    self.commit(key, path, tr, obs, "baseline", [v, a], None)
                if (v, a) == SOURCE_SEEDS and not same(tr["final"]["actions_physical"], src["final"]["actions_physical"]):
                    raise SystemExit(f"stored baseline != engine source for {obs['obs_id']}")
                out[(v, a)] = {"trace": tr, "ref": self.ref(key)}
            first = out[(v, self.prereg["action_seeds"][0])]["trace"]
            for a in self.prereg["action_seeds"][1:]:
                other = out[(v, a)]["trace"]
                if not (same(first["final"]["stage1_future_latents"], other["final"]["stage1_future_latents"])
                        and len(first["stage1_steps"]) == len(other["stage1_steps"])
                        and all(same(s[k], t[k]) for s, t in zip(first["stage1_steps"], other["stage1_steps"]) for k in STAGE1_KEYS)):
                    raise SystemExit(f"stage-1 depends on the action seed at v{v} for {obs['obs_id']}")
        return out

    # -- overrides -------------------------------------------------------------------------------
    def override_unit(self, obs, src, kind, key, path, base_r, vr, ar, future, donor, extra) -> None:
        expect = self.ident(obs, kind, video_seed=vr, action_seed=ar, donor=donor,
                            source=self.ref(f"source|{obs['obs_id']}"), receiver_baseline=base_r["ref"], **extra)
        if self.ledger.get(key):
            tr = self.ledger.load(key)
            check_meta(tr, expect, key)
            if not same(tr["final"]["stage1_future_latents"], base_r["trace"]["final"]["stage1_future_latents"]):
                raise SystemExit(f"stored unit changed the receiver stage-1 future: {key}")
            if kind == "identity" and override_mismatches(base_r["trace"], tr):
                raise SystemExit(f"stored identity no longer matches its baseline: {key}")
            return
        res = self.runner.rerun(src, video_seed=vr, action_seed=ar, cond_future_override=future)
        pt = res["probe_trace"]
        self.check_schedule(pt, key)
        tr = self.trace(expect, pt, res["actions"], keep_stage1=False)
        problems = []
        if not same(pt["final"]["stage1_future_latents"], base_r["trace"]["final"]["stage1_future_latents"]):
            problems.append("receiver stage-1 future changed")
        if kind == "identity":
            problems += override_mismatches(base_r["trace"], tr)
        if problems:
            self.ledger.fail(key, f"{kind} rejected: {problems[:6]}")
            raise SystemExit(f"{kind} rejected for {key}: {problems[:6]}")
        self.commit(key, path, tr, obs, kind, [vr, ar], donor)

    def identity(self, obs, src, base) -> None:
        for (v, a), b in base.items():
            self.override_unit(obs, src, "identity", f"identity|{obs['obs_id']}|{v}-{a}",
                               self.odir(obs) / "identity" / f"r{v}-{a}.pt", b, v, a,
                               b["trace"]["final"]["stage1_future_latents"], [v, a], {})

    def alt(self, obs, src, base) -> None:
        for pair in self.prereg["pairs"]:
            (vr, ar), (vd, ad) = pair["receiver"], pair["donor"]
            d = base[(vd, ad)]
            self.override_unit(obs, src, "alt", f"alt|{obs['obs_id']}|{vr}-{ar}|{vd}-{ad}",
                               self.odir(obs) / "alt" / f"r{vr}-{ar}_d{vd}-{ad}.pt", base[(vr, ar)], vr, ar,
                               d["trace"]["final"]["stage1_future_latents"], [vd, ad],
                               {"donor_baseline": d["ref"], "role": pair["pair_kind"]})

    # -- GT --------------------------------------------------------------------------------------
    def check_gt_pngs(self, obs) -> None:
        rec = self.records[obs["obs_id"]]
        used = {(p["path"], p["sha256"]) for p in rec["pngs"].values()}
        for post in rec["post_obs"]:
            if int(post["offset"]) in range(4, 33, 4):
                used |= {(p["path"], p["sha256"]) for p in post["pngs"].values()}
        bound = {(p["path"], p["sha256"]) for p in obs["pngs"]}
        if len(bound) != 27 or used != bound:
            raise SystemExit(f"GT PNGs (offset 0,4..32 x 3 cameras) disagree with the prereg for {obs['obs_id']}")

    def gt_future(self, obs, src) -> tuple[torch.Tensor, dict]:
        key, path = f"gt_future|{obs['obs_id']}", self.odir(obs) / "gt_future.pt"
        expect = self.ident(obs, "gt_future", source=self.ref(f"source|{obs['obs_id']}"))
        ffl = src["cond"]["first_frame_latents"]
        if self.ledger.get(key):
            g = self.ledger.load(key)
            check_meta(g, expect, key)
        else:
            self.check_gt_pngs(obs)
            frames = self.runner.gt_future_frames(self.records[obs["obs_id"]])
            if not same(np.asarray(frames[0].convert("RGB"), dtype=np.uint8), src["cond"]["image"]):
                self.ledger.fail(key, "frame0 canvas != source image")
                raise SystemExit(f"GT frame 0 canvas != source image for {obs['obs_id']}")
            lat = self.runner.encode_frames(frames).cpu()
            if not same(lat[:, :, : ffl.shape[2]], ffl.to(lat.dtype)):
                self.ledger.fail(key, "GT first latent != receiver first-frame latent")
                raise SystemExit(f"GT first latent != receiver first-frame latent for {obs['obs_id']}")
            g = {"meta": {**expect, "canvas_sha256": [array_sha(f.convert("RGB")) for f in frames]}, "latents": lat}
            self.ledger.commit(key, path, g, kind="gt_future", obs_id=obs["obs_id"], task=obs["task"],
                               episode=obs["episode"], receiver=None, donor=None)
        if not same(g["latents"][:, :, : ffl.shape[2]], ffl.to(g["latents"].dtype)):
            raise SystemExit(f"stored GT first latent != receiver first-frame latent for {obs['obs_id']}")
        return g["latents"], self.ref(key)

    def gt(self, obs, src, base) -> None:
        lat, ref = self.gt_future(obs, src)
        for (v, a), b in base.items():
            self.override_unit(obs, src, "gt", f"gt|{obs['obs_id']}|{v}-{a}", self.odir(obs) / "gt" / f"r{v}-{a}.pt",
                               b, v, a, lat, "gt", {"gt_future": ref})

    # -- driver ----------------------------------------------------------------------------------
    def expected_keys(self) -> list:
        keys = []
        for obs in self.observations:
            o = obs["obs_id"]
            keys.append(f"source|{o}")
            for v in self.prereg["video_seeds"]:
                for a in self.prereg["action_seeds"]:
                    keys += [f"baseline|{o}|{v}-{a}", f"identity|{o}|{v}-{a}"]
                    if self.mode == "run" and o in self.gt_ids:
                        keys.append(f"gt|{o}|{v}-{a}")
            if o in self.gt_ids:
                keys.append(f"gt_future|{o}")
            if self.mode == "run":
                keys += [f"alt|{o}|{p['receiver'][0]}-{p['receiver'][1]}|{p['donor'][0]}-{p['donor'][1]}"
                         for p in self.prereg["pairs"]]
        return keys

    def run(self) -> None:
        for obs in self.observations:
            src = self.source(obs)
            base = self.baselines(obs, src)
            self.identity(obs, src, base)
            if self.mode == "run":
                self.alt(obs, src, base)
                if obs["obs_id"] in self.gt_ids:
                    self.gt(obs, src, base)
            elif obs["obs_id"] in self.gt_ids:
                self.gt_future(obs, src)  # smoke: build and check the GT future, no override run
            print(f"[{self.mode}] {obs['obs_id']} done", flush=True)
        keys = self.expected_keys()
        self.ledger.require_exact(keys)
        if self.mode == "run":
            check_imports_bound(self.manifest)
        print(f"[{self.mode}] complete: {len(keys)} units", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["run", "smoke", "describe"])
    p.add_argument("--prereg")
    p.add_argument("--out")
    p.add_argument("--freeze-receipt")
    p.add_argument("--runtime-binding")
    p.add_argument("--config", default=None)
    p.add_argument("--n-obs", type=int, default=1, help="smoke only: first N observations")
    args = p.parse_args(argv)
    if args.mode == "describe":
        if not args.out:
            raise SystemExit("describe needs --out (a new .json file)")
        write_description(make_runner(IDM_CKPT, Path(args.config or DEFAULT_CONFIG).resolve()), args.out)
        return
    if not (args.prereg and args.out):
        raise SystemExit("--prereg and --out are required")
    if args.mode == "run" and not (args.freeze_receipt and args.runtime_binding):
        raise SystemExit("run needs --freeze-receipt and --runtime-binding")
    Q6(args, args.mode).run()


if __name__ == "__main__":
    main()
