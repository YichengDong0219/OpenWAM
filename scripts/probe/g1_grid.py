"""G1 compute engine (K1/K2/K4): seed grid, donor swap, single-step response, future decoding.

All work is offline on recorded G0 observations (no simulator). Inputs come from a frozen
pre-registration file (``G1_PREREG.json``, Codex T1). Expected schema (subset used here):

    {
      "prereg_sha256_note": "...",
      "video_seeds": [..4..], "action_seeds": [..4..],
      "observations": [
        {"obs_id": "...", "task": "...", "episode": int, "split": "explore"|"holdout",
         "trace": "/abs/path/chunkNNNN.pt", "trace_sha256": "...", "record": "/abs/.../record.json",
         "sigma_steps": {"high": i, "mid": j, "low": k}}        # schedule indices for THIS trace
      ],
      "pairs": [{"receiver": [v, a], "donor": [v, a], "pair_kind": "primary_same_action"|...}],
      "isolated_checks": [{"obs_id": "..."}]                    # path-blocking checks, all 3 bands
    }

``--split`` restricts every subcommand to observations of that split (explore first; the holdout
split is only run after the exploration analysis is frozen). ``swap`` (full-trajectory donor
replay) belongs to G2 and is excluded from ``all``.

Outputs (under ``--out``, which must sit below outputs/probe and must not already hold a
different prereg): ``<task>/<obs_id>/grid/v{V}_a{A}.pt`` (G0-trace-like dict with meta/cond/
init/steps/final), ``<task>/<obs_id>/swap/r{V}-{A}_d{V}-{A}.pt``, ``step_response.jsonl``,
``<task>/<obs_id>/decoded/v{V}_a{A}/frame_k.png`` and ``index.jsonl``. Every file is written
atomically and existing outputs are reused only when their recorded sha256 matches (resumable).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "probe"))

from offline_runner import OfflineRunner, load_trace, sha256_file  # noqa: E402
import g1_metrics as M  # noqa: E402

CANVAS_SLOTS = {  # assemble_multiview_layout on a 384x320 canvas (top 2/3, bottom split in half)
    "head": (slice(0, 256), slice(0, 320)),
    "left_wrist": (slice(256, 384), slice(0, 160)),
    "right_wrist": (slice(256, 384), slice(160, 320)),
}


def atomic_torch_save(obj, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return sha256_file(path)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


class Run:
    def __init__(self, prereg_path: Path, out: Path, ckpt_dir: str, config: str | None, split: str | None = None):
        self.split = split
        self.prereg_path = prereg_path.resolve()
        self.prereg = json.loads(self.prereg_path.read_text())
        self.prereg_sha = sha256_file(self.prereg_path)
        self.out = out.resolve()
        root = (REPO / "outputs" / "probe").resolve()
        if not str(self.out).startswith(str(root) + os.sep):
            raise SystemExit(f"--out must be below {root}")
        stamp = self.out / "PREREG_SHA256"
        if stamp.exists() and stamp.read_text().strip() != self.prereg_sha:
            raise SystemExit("output dir belongs to a different prereg; refusing to mix runs")
        self.out.mkdir(parents=True, exist_ok=True)
        stamp.write_text(self.prereg_sha + "\n")
        self.index = self.out / "index.jsonl"
        self.done = {}
        if self.index.exists():
            for line in self.index.read_text().splitlines():
                row = json.loads(line)
                self.done[row["path"]] = row["sha256"]
        self.runner = OfflineRunner(ckpt_dir, config)
        self._sources = {}
        prompts = [self.source(o)["cond"]["prompt"] for o in self.observations()]
        info = self.runner.precompute_prompts(prompts, offload_text_encoder=True)
        print(f"[init] prompts {info}", flush=True)

    # -- helpers -----------------------------------------------------------------------------
    def observations(self):
        obs = self.prereg["observations"]
        if self.split is not None:
            obs = [o for o in obs if o.get("split") == self.split]
            if not obs:
                raise SystemExit(f"no observations with split={self.split!r}")
        return obs

    def source(self, obs: dict) -> dict:
        key = obs["obs_id"]
        if key not in self._sources:
            if sha256_file(obs["trace"]) != obs["trace_sha256"]:
                raise ValueError(f"source trace sha mismatch for {key}")
            self._sources[key] = load_trace(obs["trace"])
        return self._sources[key]

    def obs_dir(self, obs: dict) -> Path:
        return self.out / obs["task"] / obs["obs_id"]

    def reuse(self, path: Path) -> bool:
        return str(path) in self.done and path.exists() and sha256_file(path) == self.done[str(path)]

    def record(self, path: Path, sha: str, kind: str, obs: dict, **extra) -> None:
        row = {"path": str(path), "sha256": sha, "kind": kind, "obs_id": obs["obs_id"], "task": obs["task"],
               "split": obs.get("split"), "prereg_sha256": self.prereg_sha, **extra}
        append_jsonl(self.index, row)
        self.done[str(path)] = sha

    def as_trace(self, src: dict, result: dict, *, video_seed: int, action_seed: int, extra_meta=None) -> dict:
        pt = result["probe_trace"]
        meta = {
            "format": "openwam-g1-generation/1",
            "source_trace_sha256": None,
            "schedule": pt["schedule"],
            "num_frames": src["meta"]["num_frames"],
            "action_num_frames": src["meta"]["action_num_frames"],
            "denoise_steps": src["meta"]["denoise_steps"],
            "active_action_mask": pt["active_action_mask"],
            "active_action_mask_source": pt["active_action_mask_source"],
            "seed": src["meta"]["seed"],
            "video_seed": int(video_seed),
            "action_seed": int(action_seed),
            "code": src["meta"]["code"],
            "prereg_sha256": self.prereg_sha,
            **(extra_meta or {}),
        }
        return {"meta": meta, "cond": src["cond"], "init": pt["init"], "steps": pt["steps"],
                "final": {**pt["final"], "actions_returned": np.asarray(result["actions"]).copy()}}

    def grid_path(self, obs, v, a) -> Path:
        return self.obs_dir(obs) / "grid" / f"v{v}_a{a}.pt"

    # -- K1: grid ----------------------------------------------------------------------------
    def grid(self) -> None:
        for obs in self.observations():
            src = self.source(obs)
            for v in self.prereg["video_seeds"]:
                for a in self.prereg["action_seeds"]:
                    path = self.grid_path(obs, v, a)
                    if self.reuse(path):
                        continue
                    res = self.runner.rerun(src, video_seed=v, action_seed=a, record=True)
                    tr = self.as_trace(src, res, video_seed=v, action_seed=a)
                    tr["meta"]["source_trace_sha256"] = obs["trace_sha256"]
                    self.record(path, atomic_torch_save(tr, path), "grid", obs, video_seed=v, action_seed=a)
            print(f"[grid] {obs['obs_id']} done", flush=True)

    # -- K1: donor swap ----------------------------------------------------------------------
    def swap(self) -> None:
        for obs in self.observations():
            src = self.source(obs)
            for pair in self.prereg["pairs"] + [None]:
                if pair is None:  # built-in identity control: donor == receiver must reproduce it
                    (vr, ar) = (vd, ad) = (self.prereg["video_seeds"][0], self.prereg["action_seeds"][0])
                    role = "identity_control"
                else:
                    (vr, ar), (vd, ad), role = pair["receiver"], pair["donor"], pair["pair_kind"]
                path = self.obs_dir(obs) / "swap" / f"r{vr}-{ar}_d{vd}-{ad}.pt"
                if self.reuse(path):
                    continue
                donor = torch.load(self.grid_path(obs, vd, ad), map_location="cpu", weights_only=False)
                receiver = torch.load(self.grid_path(obs, vr, ar), map_location="cpu", weights_only=False)
                res = self.runner.rerun(src, video_seed=vr, action_seed=ar, replay=donor, record=True)
                tr = self.as_trace(src, res, video_seed=vr, action_seed=ar,
                                   extra_meta={"replay_donor": [vd, ad], "role": role})
                a_swap = np.asarray(res["actions"])
                a_r = np.asarray(receiver["final"]["actions_physical"])
                a_d = np.asarray(donor["final"]["actions_physical"])
                summary = {"vs_receiver": M.chunk_diff(a_r, a_swap)["xyz"], "vs_donor": M.chunk_diff(a_d, a_swap)["xyz"],
                           "donor_vs_receiver": M.chunk_diff(a_r, a_d)["xyz"]}
                if role == "identity_control" and summary["vs_receiver"]["max"] != 0.0:
                    raise RuntimeError(f"identity swap did not reproduce the receiver for {obs['obs_id']}: {summary}")
                self.record(path, atomic_torch_save(tr, path), "swap", obs, receiver=[vr, ar], donor=[vd, ad],
                            role=role, summary=summary)
            print(f"[swap] {obs['obs_id']} done", flush=True)

    # -- K2: single-step response ------------------------------------------------------------
    def step(self) -> None:
        out = self.out / "step_response.jsonl"
        seen = set()
        if out.exists():
            for line in out.read_text().splitlines():
                r = json.loads(line)
                seen.add((r["obs_id"], tuple(r["receiver"]), tuple(r["donor"]), r["step"], r["pair_kind"]))
        for obs in self.observations():
            src = self.source(obs)
            inputs = self.runner._inputs_shared(src)
            steps = obs["sigma_steps"]
            cache_traces, cache_noop = {}, {}

            def grid_trace(v, a):
                if (v, a) not in cache_traces:
                    cache_traces[(v, a)] = torch.load(self.grid_path(obs, v, a), map_location="cpu", weights_only=False)
                return cache_traces[(v, a)]

            for pair in self.prereg["pairs"]:
                (vr, ar), (vd, ad), kind = pair["receiver"], pair["donor"], pair["pair_kind"]
                rec, don = grid_trace(vr, ar), grid_trace(vd, ad)
                mask = rec["meta"]["active_action_mask"]
                for band, i in steps.items():
                    key = (obs["obs_id"], (vr, ar), (vd, ad), int(i), kind)
                    if key in seen:
                        continue
                    s_rec = next(s for s in rec["steps"] if int(s["i"]) == int(i))
                    s_don = next(s for s in don["steps"] if int(s["i"]) == int(i))
                    if ((vr, ar), int(i)) not in cache_noop:
                        _, v_noop = self.runner.step_response(rec, i, inputs_shared=inputs)
                        cache_noop[((vr, ar), int(i))] = (v_noop, M.rel_velocity_response(v_noop, s_rec["action_velocity"], mask))
                    v_noop, noop = cache_noop[((vr, ar), int(i))]
                    _, v_swap = self.runner.step_response(rec, i, video_latents=s_don["video_latents_in"],
                                                          inputs_shared=inputs)
                    row = {"obs_id": obs["obs_id"], "task": obs["task"], "episode": obs["episode"],
                           "split": obs.get("split"), "receiver": [vr, ar], "donor": [vd, ad], "pair_kind": kind,
                           "band": band, "step": int(i), "t_v": float(s_rec["t_v"]), "t_a": float(s_rec["t_a"]),
                           "sigma_v": float(s_rec["sigma_v"]), "sigma_a": float(s_rec["sigma_a"]),
                           "noop_vs_recorded": noop,
                           "swap_response": M.rel_velocity_response(v_swap, v_noop, mask),
                           "natural_donor_vs_receiver": M.rel_velocity_response(s_don["action_velocity"],
                                                                                s_rec["action_velocity"], mask),
                           "prereg_sha256": self.prereg_sha}
                    append_jsonl(out, row)
            print(f"[step] {obs['obs_id']} done", flush=True)

    # -- path-blocking check (isolated mask) -------------------------------------------------
    def isolated(self) -> None:
        out = self.out / "isolated_check.jsonl"
        wanted = {c["obs_id"] for c in self.prereg.get("isolated_checks", [])}
        for obs in self.observations():
            if obs["obs_id"] not in wanted:
                continue
            src = self.source(obs)
            inputs = self.runner._inputs_shared(src)
            vr, ar = self.prereg["video_seeds"][0], self.prereg["action_seeds"][0]
            vd = self.prereg["video_seeds"][1]
            rec = torch.load(self.grid_path(obs, vr, ar), map_location="cpu", weights_only=False)
            don = torch.load(self.grid_path(obs, vd, ar), map_location="cpu", weights_only=False)
            mask = rec["meta"]["active_action_mask"]
            for band, i in obs["sigma_steps"].items():
                s_don = next(s for s in don["steps"] if int(s["i"]) == int(i))
                row = {"obs_id": obs["obs_id"], "task": obs["task"], "band": band, "step": int(i),
                       "receiver": [vr, ar], "donor": [vd, ar], "prereg_sha256": self.prereg_sha}
                for mode in ("own", "isolated"):
                    ctx = self.runner.attention_mask_mode("isolated") if mode == "isolated" else _null()
                    with ctx:
                        _, v0 = self.runner.step_response(rec, i, inputs_shared=inputs)
                        _, v1 = self.runner.step_response(rec, i, video_latents=s_don["video_latents_in"],
                                                          inputs_shared=inputs)
                    row[f"{mode}_mask_response"] = M.rel_velocity_response(v1, v0, mask)
                append_jsonl(out, row)
            print(f"[isolated] {obs['obs_id']} done", flush=True)

    # -- K4: decode predicted futures --------------------------------------------------------
    @torch.no_grad()
    def decode(self) -> None:
        from PIL import Image

        vb = self.runner.arch.video_backbone
        out_rows = self.out / "decode_proxy.jsonl"
        for obs in self.observations():
            for v in self.prereg["video_seeds"]:
                for a in self.prereg["action_seeds"]:
                    d = self.obs_dir(obs) / "decoded" / f"v{v}_a{a}"
                    if (d / "DONE").exists():
                        continue
                    tr = torch.load(self.grid_path(obs, v, a), map_location="cpu", weights_only=False)
                    lat = tr["final"]["video_latents"].to(self.runner.arch.device)
                    frames = vb.decode_video(lat, tiled=True)
                    d.mkdir(parents=True, exist_ok=True)
                    arrs = []
                    for k, fr in enumerate(frames):
                        fr.save(d / f"frame_{k}.png")
                        arrs.append(np.asarray(fr.convert("RGB"), dtype=np.float32))
                    proxy = {}
                    for slot, (rs, cs) in CANVAS_SLOTS.items():
                        diffs = [float(np.abs(arrs[k + 1][rs, cs] - arrs[k][rs, cs]).mean()) for k in range(len(arrs) - 1)]
                        proxy[slot] = {"framediff_mean": float(np.mean(diffs)), "framediff_per_step": diffs}
                    append_jsonl(out_rows, {"obs_id": obs["obs_id"], "task": obs["task"], "video_seed": v,
                                            "action_seed": a, "n_frames": len(arrs),
                                            "frame_shape": list(arrs[0].shape), "slot_framediff": proxy,
                                            "prereg_sha256": self.prereg_sha})
                    (d / "DONE").write_text("ok\n")
            print(f"[decode] {obs['obs_id']} done", flush=True)


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["grid", "step", "isolated", "decode", "all", "swap"])
    p.add_argument("--split", default=None, help="restrict to observations of this split (e.g. explore)")
    p.add_argument("--prereg", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--ckpt-dir", default=str(REPO / "assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboTwin-Full"))
    p.add_argument("--config", default=None)
    args = p.parse_args(argv)
    run = Run(args.prereg, args.out, args.ckpt_dir, args.config, split=args.split)
    for cmd in (["grid", "step", "isolated", "decode"] if args.cmd == "all" else [args.cmd]):
        getattr(run, cmd)()


if __name__ == "__main__":
    main()
