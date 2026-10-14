"""G2 K8: IDM clean-future injection (Q6) and GT-future command consistency on the G2 observations.

Per observation (inputs rebuilt from the G0 record, not from alpha's trace):
  source    one engine.generate (v1001, a2001) with a probe trace -> meta/cond for every rerun
  baseline  16 full IDM generations (4 video x 4 action seeds); stage-1 futures must be identical
            across action seeds at each video seed (the verified stage1_reuse condition)
  identity  16 reruns with cond_future_override = own stage-1 future; must reproduce the baseline
            actions, stage-2 steps and clean condition bit for bit
  alt       48 primary pairs: receiver (vr, ar) with the donor baseline's (vd, ad) stage-1 future
  gt        GT-eligible observations only: 16 receivers with the executed future (record PNGs at
            offsets 0,4..32 -> server canvas -> IDM VAE), PNG hashes bound to the prereg
Every override run must leave the receiver's own stage-1 future unchanged. Writes traces + index
only; no Q6 effect metric is computed here (A3 reads the traces).

    run   : formal; needs the G2 freeze receipt, a jointly bound IDM runtime binding and a clean tree
    smoke : source + baseline + identity (+ GT future construction) on the first N observations; no alt/gt runs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "probe"))

from g2_q5 import append_jsonl, atomic_torch_save, same  # noqa: E402
from offline_runner import OfflineRunner, sha256_file  # noqa: E402

IDM_CKPT = REPO / "assets/openwam_ckpt/openwam_study/robotwin_dual_system_idm"
STAGE2_KEYS = ("t_a", "action_latents_in", "action_velocity")
FINAL_KEYS = ("action_latents", "actions_normalized", "actions_physical", "cond_video_latents_used")
SOURCE_SEEDS = (1001, 2001)


def override_mismatches(base: dict, pt: dict) -> list:
    if len(base["steps"]) != len(pt["steps"]):
        return ["n_steps"]
    bad = [f"step{s['i']}.{k}" for s, t in zip(base["steps"], pt["steps"]) for k in STAGE2_KEYS if not same(s[k], t[k])]
    return bad + [f"final.{k}" for k in FINAL_KEYS if not same(base["final"][k], pt["final"][k])]


class Q6:
    def __init__(self, args, mode: str):
        self.mode = mode
        self.prereg_path = Path(args.prereg).resolve()
        self.prereg = json.loads(self.prereg_path.read_text())
        self.prereg_sha = sha256_file(self.prereg_path)
        if self.prereg.get("format") != "openwam-g2-prereg/1":
            raise SystemExit("not a G2 prereg")
        self.receipts = self._receipts(args) if mode == "run" else {}
        self.gt_ids = set(self.prereg["gt_eligibility"]["eligible_explore_obs_ids"])
        self.out = Path(args.out).resolve()
        root = (REPO / "outputs" / "probe").resolve()
        if not str(self.out).startswith(str(root) + os.sep):
            raise SystemExit(f"--out must be below {root}")
        stamp = self.out / "RUN_STAMP.json"
        want_stamp = {"prereg_sha256": self.prereg_sha, "mode": mode,
                      "runtime_binding_sha256": self.receipts.get("runtime_binding", {}).get("sha256")}
        if stamp.exists() and json.loads(stamp.read_text()) != want_stamp:
            raise SystemExit("output dir belongs to a different prereg/mode/binding; refusing to mix runs")
        self.out.mkdir(parents=True, exist_ok=True)
        stamp.write_text(json.dumps(want_stamp, sort_keys=True))
        self.index = self.out / "index.jsonl"
        self.done = {}
        if self.index.exists():
            for line in self.index.read_text().splitlines():
                row = json.loads(line)
                self.done[row["path"]] = row["sha256"]
        obs = self.prereg["observations"]
        self.observations = obs[: args.n_obs] if mode == "smoke" else obs
        self.records = {}
        for o in self.observations:
            if sha256_file(o["record"]) != o["record_sha256"]:
                raise SystemExit(f"record sha mismatch for {o['obs_id']}")
            self.records[o["obs_id"]] = json.loads(Path(o["record"]).read_text())
        self.runner = OfflineRunner(str(IDM_CKPT), args.config)
        if type(self.runner.arch).__name__.lower().find("idm") < 0:
            raise SystemExit(f"checkpoint is not an IDM architecture: {type(self.runner.arch).__name__}")
        from openwam.deploy.engine import _cfg_sha256, _code_version

        self.runtime_code = dict(_code_version())
        self.runtime_config_sha256 = _cfg_sha256(self.runner.engine.cfg)
        if mode == "run" and (self.runtime_code.get("dirty") or not self.runtime_code.get("commit")):
            raise SystemExit(f"formal run needs a clean committed tree; got {self.runtime_code}")
        info = self.runner.precompute_prompts([r["prompt"] for r in self.records.values()])
        print(f"[init] {mode}: {len(self.observations)} obs; prompts {info}; code {self.runtime_code}", flush=True)

    def _receipts(self, args) -> dict:
        rec = json.loads(Path(args.freeze_receipt).read_text())
        if rec.get("status") != "frozen" or rec.get("prereg_sha256") != self.prereg_sha:
            raise SystemExit("G2 freeze receipt does not freeze this prereg")
        bind_path = Path(args.runtime_binding)
        bind = json.loads(bind_path.read_text())
        if bind.get("status") != "jointly_bound" or bind.get("scope") != "Q6_idm":
            raise SystemExit("IDM runtime binding is not jointly bound for Q6")
        if not all(isinstance((bind.get("signatures") or {}).get(k), str) and bind["signatures"][k].strip()
                   for k in ("claude", "codex")):
            raise SystemExit("runtime binding lacks both signatures")
        man_path = Path(bind["manifest_path"])
        if sha256_file(man_path) != bind["manifest_sha256"]:
            raise SystemExit("runtime manifest differs from binding")
        man = json.loads(man_path.read_text())
        if man.get("prereg_sha256") != self.prereg_sha:
            raise SystemExit("runtime manifest names another prereg")
        for art in man["artifacts"]:
            if sha256_file(art["path"]) != art["sha256"]:
                raise SystemExit(f"runtime artifact changed: {art['path']}")
        print(f"[init] {len(man['artifacts'])} runtime artifacts verified", flush=True)
        return {"freeze_receipt": {"path": str(Path(args.freeze_receipt).resolve()), "sha256": sha256_file(args.freeze_receipt)},
                "runtime_binding": {"path": str(bind_path.resolve()), "sha256": sha256_file(bind_path)},
                "runtime_manifest": {"path": str(man_path), "sha256": bind["manifest_sha256"],
                                     "commit": man["runtime_commit"]}}

    # -- io --------------------------------------------------------------------------------------
    def odir(self, obs) -> Path:
        return self.out / obs["task"] / obs["obs_id"]

    def reuse(self, path: Path) -> bool:
        return str(path) in self.done and path.exists() and sha256_file(path) == self.done[str(path)]

    def index_row(self, obs, kind, path, sha, **extra) -> None:
        row = {"path": str(path), "sha256": sha, "kind": kind, "obs_id": obs["obs_id"], "task": obs["task"],
               "episode": obs["episode"], "prereg_sha256": self.prereg_sha, **extra}
        append_jsonl(self.index, row)
        self.done[str(path)] = sha

    def save(self, obs, kind, path, source, pt, actions, vr, ar, extra, keep_stage1: bool) -> None:
        tr = {"meta": {"format": "openwam-g2-q6/1", "kind": kind, "prereg_sha256": self.prereg_sha,
                       "record_sha256": obs["record_sha256"], "source": source, "schedule": pt["schedule"],
                       "video_seed": vr, "action_seed": ar, "active_action_mask": pt["active_action_mask"],
                       "runtime_code": self.runtime_code, "runtime_config_sha256": self.runtime_config_sha256,
                       "receipts": self.receipts, **extra},
              "cond": pt["cond"], "init": pt["init"], "steps": pt["steps"],
              "final": {**pt["final"], "actions_returned": np.asarray(actions).copy()}}
        if keep_stage1:
            tr["stage1_steps"] = pt["stage1_steps"]
        self.index_row(obs, kind, path, atomic_torch_save(tr, path), receiver=[vr, ar], **extra)

    def load(self, path: Path) -> dict:
        if not self.reuse(path):
            raise RuntimeError(f"missing or changed output: {path}")
        return torch.load(path, map_location="cpu", weights_only=False)

    # -- work ------------------------------------------------------------------------------------
    def source(self, obs) -> tuple[dict, dict]:
        rec = self.records[obs["obs_id"]]
        path = self.odir(obs) / "source.pt"
        if not self.reuse(path):
            canvas = self.runner.canvas_from_pngs(rec["pngs"], prompt=rec["prompt"], state=rec.get("proprio"))
            v, a = SOURCE_SEEDS
            res = self.runner.engine.generate({"first_frame_image": [canvas["image"]], "prompt": canvas["prompt"],
                                               "proprio": canvas.get("state"),
                                               "probe": {"trace_path": str(path), "video_seed": v, "action_seed": a}})
            receipt = res["probe_receipt"]
            self.index_row(obs, "source", path, receipt["sha256"], receiver=[v, a])
        tr = self.load(path)
        return tr, {"path": str(path), "sha256": self.done[str(path)]}

    def baselines(self, obs, src, src_ref) -> dict:
        out = {}
        for v in self.prereg["video_seeds"]:
            for a in self.prereg["action_seeds"]:
                path = self.odir(obs) / "baseline" / f"v{v}_a{a}.pt"
                if not self.reuse(path):
                    res = self.runner.rerun(src, video_seed=v, action_seed=a)
                    pt = res["probe_trace"]
                    if (v, a) == SOURCE_SEEDS and not same(pt["final"]["actions_physical"], src["final"]["actions_physical"]):
                        raise RuntimeError(f"offline baseline != engine source for {obs['obs_id']}")
                    self.save(obs, "baseline", path, src_ref, pt, res["actions"], v, a, {"donor": None}, keep_stage1=True)
                tr = self.load(path)
                out[(v, a)] = {"trace": tr, "path": str(path), "sha256": self.done[str(path)]}
            fut = [out[(v, a)]["trace"]["final"]["stage1_future_latents"] for a in self.prereg["action_seeds"]]
            if not all(same(fut[0], f) for f in fut[1:]):
                raise RuntimeError(f"stage-1 future depends on the action seed at v{v} for {obs['obs_id']}")
        return out

    def override(self, obs, src, src_ref, kind, path, base_r, vr, ar, future, extra) -> dict:
        res = self.runner.rerun(src, video_seed=vr, action_seed=ar, cond_future_override=future)
        pt = res["probe_trace"]
        if not same(pt["final"]["stage1_future_latents"], base_r["trace"]["final"]["stage1_future_latents"]):
            raise RuntimeError(f"receiver stage-1 future changed under override: {path}")
        self.save(obs, kind, path, src_ref, pt, res["actions"], vr, ar,
                  {**extra, "receiver_baseline": {"path": base_r["path"], "sha256": base_r["sha256"]}}, keep_stage1=False)
        return pt

    def identity(self, obs, src, src_ref, base) -> None:
        for (v, a), b in base.items():
            path = self.odir(obs) / "identity" / f"r{v}-{a}.pt"
            if self.reuse(path):
                continue
            pt = self.override(obs, src, src_ref, "identity", path, b, v, a,
                               b["trace"]["final"]["stage1_future_latents"], {"donor": [v, a]})
            bad = override_mismatches(b["trace"], pt)
            if bad:
                raise RuntimeError(f"identity override differs from baseline {obs['obs_id']} v{v} a{a}: {bad[:6]}")

    def alt(self, obs, src, src_ref, base) -> None:
        for pair in self.prereg["pairs"]:
            (vr, ar), (vd, ad) = pair["receiver"], pair["donor"]
            path = self.odir(obs) / "alt" / f"r{vr}-{ar}_d{vd}-{ad}.pt"
            if self.reuse(path):
                continue
            d = base[(vd, ad)]
            self.override(obs, src, src_ref, "alt", path, base[(vr, ar)], vr, ar,
                          d["trace"]["final"]["stage1_future_latents"],
                          {"donor": [vd, ad], "role": pair["pair_kind"],
                           "donor_baseline": {"path": d["path"], "sha256": d["sha256"]}})

    def gt_future(self, obs, src) -> tuple[torch.Tensor, dict]:
        rec = self.records[obs["obs_id"]]
        used = {(p["path"], p["sha256"]) for p in rec["pngs"].values()}
        for post in rec["post_obs"]:
            if int(post["offset"]) in range(4, 33, 4):
                used |= {(p["path"], p["sha256"]) for p in post["pngs"].values()}
        bound = {(p["path"], p["sha256"]) for p in obs["pngs"]}
        if len(bound) != 27 or used != bound:
            raise RuntimeError(f"GT PNGs (offset 0,4..32 x 3 cameras) disagree with the prereg for {obs['obs_id']}")
        path = self.odir(obs) / "gt_future.pt"
        if not self.reuse(path):
            frames = self.runner.gt_future_frames(rec)
            if not same(np.asarray(frames[0].convert("RGB")), src["cond"]["image"]):
                raise RuntimeError(f"GT frame 0 canvas != source conditioning image for {obs['obs_id']}")
            lat = self.runner.encode_frames(frames).cpu()
            ffl = src["cond"]["first_frame_latents"]
            if not same(lat[:, :, : ffl.shape[2]], ffl.to(lat.dtype)):
                raise RuntimeError(f"GT first latent != receiver first-frame latent for {obs['obs_id']}")
            canv = [hashlib_sha(np.asarray(f.convert("RGB"))) for f in frames]
            sha = atomic_torch_save({"latents": lat, "canvas_sha256": canv, "record_sha256": obs["record_sha256"]}, path)
            self.index_row(obs, "gt_future", path, sha)
        return self.load(path)["latents"], {"path": str(path), "sha256": self.done[str(path)]}

    def gt(self, obs, src, src_ref, base) -> None:
        lat, ref = self.gt_future(obs, src)
        for (v, a), b in base.items():
            path = self.odir(obs) / "gt" / f"r{v}-{a}.pt"
            if self.reuse(path):
                continue
            self.override(obs, src, src_ref, "gt", path, b, v, a, lat, {"donor": "gt", "gt_future": ref})

    def run(self) -> None:
        for obs in self.observations:
            src, src_ref = self.source(obs)
            base = self.baselines(obs, src, src_ref)
            self.identity(obs, src, src_ref, base)
            if self.mode == "run":
                self.alt(obs, src, src_ref, base)
                if obs["obs_id"] in self.gt_ids:
                    self.gt(obs, src, src_ref, base)
            elif obs["obs_id"] in self.gt_ids:
                self.gt_future(obs, src)  # smoke: build and check the GT future, no override run
            print(f"[{self.mode}] {obs['obs_id']} done", flush=True)
        kinds = {}
        for line in self.index.read_text().splitlines():
            k = json.loads(line)["kind"]
            kinds[k] = kinds.get(k, 0) + 1
        n = len(self.observations)
        n_gt = sum(o["obs_id"] in self.gt_ids for o in self.observations)
        want = {"source": n, "baseline": 16 * n, "identity": 16 * n, "gt_future": n_gt}
        if self.mode == "run":
            want.update(alt=len(self.prereg["pairs"]) * n, gt=16 * n_gt)
        want = {k: v for k, v in want.items() if v}
        if kinds != want:
            raise SystemExit(f"index counts {kinds} != expected {want}")
        print(f"[{self.mode}] complete: {kinds}", flush=True)


def hashlib_sha(arr: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["run", "smoke"])
    p.add_argument("--prereg", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--freeze-receipt")
    p.add_argument("--runtime-binding")
    p.add_argument("--config", default=None)
    p.add_argument("--n-obs", type=int, default=1, help="smoke only: first N observations")
    args = p.parse_args(argv)
    if args.mode == "run" and not (args.freeze_receipt and args.runtime_binding):
        raise SystemExit("run needs --freeze-receipt and --runtime-binding")
    Q6(args, args.mode).run()


if __name__ == "__main__":
    main()
