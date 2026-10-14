"""G2 K5: alpha full-trajectory video swap (Q5) on the G2 observations.

For every observation the receiver (v, a) is regenerated while the donor's recorded video latents
replace the receiver's at EVERY denoising step (``OfflineRunner.rerun(replay=donor)``); actions
evolve under the receiver's own action noise, prompt, proprio and first frame. Baselines are the
G1 grid generations (``<g1 root>/<task>/<obs>/grid/v{V}_a{A}.pt``), reused only after the G1 index
sha, the per-file indexed sha, the G1 prereg sha, the source trace sha and the seeds all match.

Before any swap of an observation, all 16 receivers are replayed against THEMSELVES (identity)
and must reproduce the baseline actions and every recorded step tensor bit for bit; otherwise the
run stops (the current runtime would not be the one that made the baselines).

This script writes traces and an index only. It computes no Q5 effect metric; the analysis (A3)
reads the traces.

    run   : formal; needs the G2 freeze receipt, a jointly bound alpha runtime binding and a clean tree
    smoke : identity replays only on the first N observations (no swap, no binding needed)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "probe"))

from offline_runner import OfflineRunner, load_trace, sha256_file  # noqa: E402

ALPHA_CKPT = REPO / "assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboTwin-Full"
STEP_KEYS = ("t_v", "t_a", "video_latents_in", "action_latents_in", "video_velocity", "action_velocity")
FINAL_KEYS = ("video_latents", "action_latents", "actions_normalized", "actions_physical")


def atomic_torch_save(obj, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    if path.exists():
        tmp.unlink()
        raise FileExistsError(f"refusing to overwrite {path}")
    os.replace(tmp, path)
    return sha256_file(path)


def append_jsonl(path: Path, row: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def same(a, b) -> bool:
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, (float, int)) or isinstance(b, (float, int)):
        return float(a) == float(b)
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)


def identity_mismatches(base: dict, pt: dict) -> list:
    """Names of every recorded tensor where a self-replay differs from its baseline (empty = bitwise)."""
    bad = []
    if len(base["steps"]) != len(pt["steps"]):
        return ["n_steps"]
    for s, t in zip(base["steps"], pt["steps"]):
        bad += [f"step{s['i']}.{k}" for k in STEP_KEYS if k in s and not same(s[k], t[k])]
    bad += [f"final.{k}" for k in FINAL_KEYS if not same(base["final"][k], pt["final"][k])]
    return bad


class Q5:
    def __init__(self, args, mode: str):
        self.mode = mode
        self.prereg_path = Path(args.prereg).resolve()
        self.prereg = json.loads(self.prereg_path.read_text())
        self.prereg_sha = sha256_file(self.prereg_path)
        if self.prereg.get("format") != "openwam-g2-prereg/1":
            raise SystemExit("not a G2 prereg")
        self.receipts = self._receipts(args) if mode == "run" else {}
        self.g1_prereg_sha = self.prereg["provenance"]["g1_prereg"]["sha256"]
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
        self.baseline_index = self._baseline_indexes()
        obs = self.prereg["observations"]
        self.observations = obs[: args.n_obs] if mode == "smoke" else obs
        for o in self.observations:
            if sha256_file(o["trace"]) != o["trace_sha256"]:
                raise SystemExit(f"source trace sha mismatch for {o['obs_id']}")
        self.runner = OfflineRunner(str(ALPHA_CKPT), args.config)
        from openwam.deploy.engine import _cfg_sha256, _code_version

        self.runtime_code = dict(_code_version())
        self.runtime_config_sha256 = _cfg_sha256(self.runner.engine.cfg)
        if mode == "run":
            # the manifest pins every runtime file by sha256; the commit itself may move (other owners' files)
            if self.runtime_code.get("dirty") or not self.runtime_code.get("commit"):
                raise SystemExit(f"formal run needs a clean committed tree; got {self.runtime_code}")
        self._sources = {}
        info = self.runner.precompute_prompts([self.source(o)["cond"]["prompt"] for o in self.observations])
        print(f"[init] {mode}: {len(self.observations)} obs; prompts {info}; code {self.runtime_code}", flush=True)

    # -- protections ---------------------------------------------------------------------------
    def _receipts(self, args) -> dict:
        rec = json.loads(Path(args.freeze_receipt).read_text())
        if rec.get("status") != "frozen" or rec.get("prereg_sha256") != self.prereg_sha:
            raise SystemExit("G2 freeze receipt does not freeze this prereg")
        bind_path = Path(args.runtime_binding)
        bind = json.loads(bind_path.read_text())
        if bind.get("status") != "jointly_bound" or bind.get("scope") != "Q5_alpha":
            raise SystemExit("alpha runtime binding is not jointly bound for Q5")
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
                "runtime_manifest": {"path": str(man_path), "sha256": bind["manifest_sha256"]},
                "runtime_manifest_commit": man["runtime_commit"]}

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
            self._sources[obs["obs_id"]] = load_trace(obs["trace"])
        return self._sources[obs["obs_id"]]

    def baseline(self, obs: dict, v: int, a: int) -> dict:
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
        return {"trace": tr, "path": str(path), "sha256": want}

    def reuse(self, path: Path) -> bool:
        return str(path) in self.done and path.exists() and sha256_file(path) == self.done[str(path)]

    def save(self, obs, kind, path, src, res, vr, ar, base_r, base_d, extra) -> None:
        pt = res["probe_trace"]
        tr = {"meta": {"format": "openwam-g2-q5/1", "kind": kind, "prereg_sha256": self.prereg_sha,
                       "source_trace_sha256": obs["trace_sha256"], "schedule": pt["schedule"],
                       "video_seed": vr, "action_seed": ar,
                       "receiver_baseline": {"path": base_r["path"], "sha256": base_r["sha256"]},
                       "donor_baseline": {"path": base_d["path"], "sha256": base_d["sha256"]},
                       "active_action_mask": pt["active_action_mask"],
                       "runtime_code": self.runtime_code, "runtime_config_sha256": self.runtime_config_sha256,
                       "receipts": self.receipts, **extra},
              "cond": src["cond"], "init": pt["init"], "steps": pt["steps"],
              "final": {**pt["final"], "actions_returned": np.asarray(res["actions"]).copy()}}
        sha = atomic_torch_save(tr, path)
        row = {"path": str(path), "sha256": sha, "kind": kind, "obs_id": obs["obs_id"], "task": obs["task"],
               "episode": obs["episode"], "receiver": [vr, ar], "donor": extra["donor"], "prereg_sha256": self.prereg_sha}
        append_jsonl(self.index, row)
        self.done[str(path)] = sha

    # -- work ------------------------------------------------------------------------------------
    def identity(self, obs: dict) -> None:
        src, d = self.source(obs), self.out / obs["task"] / obs["obs_id"] / "identity"
        for v in self.prereg["video_seeds"]:
            for a in self.prereg["action_seeds"]:
                path = d / f"r{v}-{a}.pt"
                if self.reuse(path):
                    continue
                base = self.baseline(obs, v, a)
                res = self.runner.rerun(src, video_seed=v, action_seed=a, replay=base["trace"], record=True)
                bad = identity_mismatches(base["trace"], res["probe_trace"])
                if bad:
                    raise RuntimeError(f"identity replay differs from baseline {obs['obs_id']} v{v} a{a}: {bad[:6]}")
                self.save(obs, "identity", path, src, res, v, a, base, base, {"donor": [v, a], "role": "identity"})

    def swap(self, obs: dict) -> None:
        src, d = self.source(obs), self.out / obs["task"] / obs["obs_id"] / "swap"
        for pair in self.prereg["pairs"]:
            (vr, ar), (vd, ad) = pair["receiver"], pair["donor"]
            path = d / f"r{vr}-{ar}_d{vd}-{ad}.pt"
            if self.reuse(path):
                continue
            base_r, base_d = self.baseline(obs, vr, ar), self.baseline(obs, vd, ad)
            res = self.runner.rerun(src, video_seed=vr, action_seed=ar, replay=base_d["trace"], record=True)
            self.save(obs, "swap", path, src, res, vr, ar, base_r, base_d,
                      {"donor": [vd, ad], "role": pair["pair_kind"]})

    def run(self) -> None:
        for obs in self.observations:
            self.identity(obs)
            if self.mode == "run":
                self.swap(obs)
            print(f"[{self.mode}] {obs['obs_id']} done", flush=True)
        kinds = {}
        for line in self.index.read_text().splitlines():
            k = json.loads(line)["kind"]
            kinds[k] = kinds.get(k, 0) + 1
        want = {"identity": 16 * len(self.observations)}
        if self.mode == "run":
            want["swap"] = len(self.prereg["pairs"]) * len(self.observations)
        if kinds != want:
            raise SystemExit(f"index counts {kinds} != expected {want}")
        print(f"[{self.mode}] complete: {kinds}", flush=True)


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
    Q5(args, args.mode).run()


if __name__ == "__main__":
    main()
