"""G0 offline probe runner (M4): rebuild a generation from a recorded trace without a server.

Python API (used by G1/G2):
    runner = OfflineRunner(ckpt_dir, config)
    trace = load_trace(path)
    out = runner.rerun(trace, video_seed=..., action_seed=..., replay=donor_trace)   # full chunk
    vel = runner.step_response(trace, i, video_latents=...)                          # one forward

CLI:
    python scripts/probe/offline_runner.py check --ckpt-dir CK --trace T.pt [--record R/record.json]

``check`` reports, for one trace:
  * receipt   -- trace sha256 equals the client record's probe receipt (when --record is given)
  * first_frame -- recomputed first-frame latents vs the recorded ones
  * rerun     -- offline full-chunk regeneration vs the online ``final.actions_physical``
                 (cross-process reproducibility; online server and this process differ)
  * step_noop -- per denoising step, recomputed action velocity from the recorded inputs vs the
                 recorded action velocity (no intervention)
  * isolated  -- path-blocking check: perturb only the future video latents at one step and
                 compare the action-velocity change under the checkpoint's own mask and under a
                 temporarily rebuilt ``isolated`` mask (action sees only the first frame)
Numbers are reported, not judged: the no-op tolerance comes from the V0 suite.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def load_trace(path) -> dict:
    trace = torch.load(path, map_location="cpu", weights_only=False)
    if trace.get("meta", {}).get("format") != "openwam-g0-probe-trace/1":
        raise ValueError(f"{path} is not an openwam-g0-probe-trace/1 file")
    return trace


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _maxabs(a, b) -> float:
    a = torch.as_tensor(np.asarray(a) if not torch.is_tensor(a) else a).float()
    b = torch.as_tensor(np.asarray(b) if not torch.is_tensor(b) else b).float()
    return float((a - b).abs().max())


class OfflineRunner:
    def __init__(self, ckpt_dir: str, config: str | None = None, device: str = "cuda"):
        from openwam.deploy.server import _build_argparser, build_server_from_config, resolve_deploy_args

        config = config or str(REPO / "configs" / "deploy_probe.yaml")
        args = _build_argparser().parse_args(["--config", config, "--ckpt-dir", ckpt_dir])
        cfg, ckpt = resolve_deploy_args(args)
        self.server = build_server_from_config(cfg=cfg, ckpt_dir=ckpt, device=device, ckpt_name=None,
                                               readiness_token=None)
        self.server._init_policy()  # builds the ObsPreprocessor used by inputs_from_record()
        self.engine = self.server.engine
        self.arch = self.engine.architecture
        if self.engine._dit_cache is not None:
            raise ValueError("offline probe runner requires dit_cache disabled (use configs/deploy_probe.yaml)")

    # -- memory: GPU-exact prompt embeddings, then text encoder off the GPU ---------------------
    @torch.no_grad()
    def precompute_prompts(self, prompts, offload_text_encoder: bool = True) -> dict:
        """Encode ``prompts`` on the GPU exactly as the online server does, cache them in the engine's
        prompt cache, then (optionally) park the text encoder on the CPU to free ~11 GB.

        To make room on a shared GPU the video DiT is moved to CPU only while encoding and restored
        afterwards, so the embeddings are numerically identical to the online path. Once the text
        encoder is on CPU, an un-cached prompt fails loudly (device mismatch) instead of silently taking
        a different numeric path.
        """
        from openwam.model.video_backbone.wan import encode as wan_encode

        cache = self.engine._prompt_embed_cache
        if cache is None:
            raise ValueError("deploy config disables prompt_embed_cache; enable it for the offline runner")
        vb, device = self.arch.video_backbone, self.arch.device
        todo = [p for p in dict.fromkeys(prompts) if p not in cache]
        if todo:
            if len(todo) > getattr(cache, "_maxsize", 32):
                raise ValueError(f"{len(todo)} prompts exceed the prompt cache size")
            if next(vb.text_encoder.parameters()).device.type != "cuda":
                raise RuntimeError("text encoder already offloaded; cannot encode new prompts")
            vb.dit.to("cpu")
            torch.cuda.empty_cache()
            try:
                for p in todo:
                    wan_encode.encode_text_for_inference(p, vace_cache=None, prompt_embed_cache=cache,
                                                         tokenizer=vb._tokenizer, text_encoder=vb.text_encoder,
                                                         device=device)
            finally:
                vb.dit.to(device)
        if offload_text_encoder:
            vb.text_encoder.to("cpu")
            torch.cuda.empty_cache()
        return {"encoded": len(todo), "cached": len(cache), "text_encoder_offloaded": bool(offload_text_encoder)}

    # -- condition reconstruction -------------------------------------------------------------
    def _shift(self) -> float:
        ab = getattr(self.arch, "action_backbone", None)
        shift = getattr(ab, "shift_action", None) if ab is not None else None
        return 5.0 if shift is None else float(shift)

    def generate_kwargs(self, trace: dict, *, video_seed=None, action_seed=None) -> dict:
        meta, cond = trace["meta"], trace["cond"]
        image = cond["image"]
        if image is None:
            raise ValueError("trace has no conditioning image")
        return {
            "schedule": [tuple(x) for x in meta["schedule"]],
            "prompt": cond["prompt"],
            "first_frame_image": [Image.fromarray(np.asarray(image, dtype=np.uint8))],
            "num_frames": int(meta["num_frames"]),
            "action_num_frames": int(meta["action_num_frames"]),
            "height": int(image.shape[0]),
            "width": int(image.shape[1]),
            "seed": int(meta["seed"]),
            "video_seed": int(meta["video_seed"] if video_seed is None else video_seed),
            "action_seed": int(meta["action_seed"] if action_seed is None else action_seed),
            "tiled": True,
            "num_inference_steps": int(meta["denoise_steps"]),
            "shift": self._shift(),
            "dit_cache": None,
            "decode_video": False,
            "vace_cache": {},
            "prompt_embed_cache": self.engine._prompt_embed_cache,
            "proprio": cond.get("proprio"),
        }

    def _inputs_shared(self, trace: dict, video_seed=None) -> dict:
        """Mirror the preprocessing at the top of ``BaseWAMArchitecture.generate``."""
        kw = self.generate_kwargs(trace, video_seed=video_seed)
        arch, vb = self.arch, self.arch.video_backbone
        inputs = vb.preprocess_input_for_inference(
            prompt=kw["prompt"], vace_video=None, first_frame_image=kw["first_frame_image"],
            num_frames=kw["num_frames"], height=kw["height"], width=kw["width"], seed=kw["video_seed"],
            num_inference_steps=kw["num_inference_steps"], shift=kw["shift"], tiled=True, vace_cache={},
            prompt_embed_cache=kw["prompt_embed_cache"], cfg_scale=1.0, cfg_merge=False,
        )
        if arch.uses_proprioception:
            inputs["proprio"] = kw["proprio"].to(device=arch.device, dtype=arch.dtype)
        return inputs

    def canvas_from_pngs(self, pngs: dict, prompt: str = "", state=None):
        """Server-side canvas (PIL) for one set of client PNGs, via the real ObsPreprocessor."""
        from benchmarks.utils import client as ws_client

        cams = {}
        for label in ("head", "left", "right"):
            entry = (pngs or {}).get(label)
            if entry is None:
                cams[label] = None
                continue
            path = Path(entry["path"])
            if sha256_file(path) != entry.get("sha256"):
                raise ValueError(f"PNG sha256 mismatch for {path}")
            cams[label] = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        payload = ws_client.build_payload(
            head=ws_client.encode_numpy_b64(cams["head"]),
            left_wrist=ws_client.encode_numpy_b64(cams["left"]) if cams["left"] is not None else None,
            right_wrist=ws_client.encode_numpy_b64(cams["right"]) if cams["right"] is not None else None,
            prompt=prompt, state=state,
        )
        payload.pop("type", None)
        return self.server._obs_preprocessor.preprocess(payload)

    def gt_future_frames(self, record: dict, horizon: int = 32, stride: int = 4) -> list:
        """The executed future as 1 + horizon/stride canvases (request frame + post_obs offsets).

        Only for chunks that executed the whole horizon with every aligned observation present;
        otherwise raises (a truncated chunk has no real future to stand in for the prediction).
        """
        if int(record.get("executed_count", -1)) != horizon or record.get("termination") not in (None, "none"):
            raise ValueError(f"chunk did not execute the full {horizon}-step horizon without terminating")
        by_offset = {int(p["offset"]): p for p in record.get("post_obs", [])}
        offsets = list(range(stride, horizon + 1, stride))
        missing = [o for o in offsets if o not in by_offset]
        if missing:
            raise ValueError(f"missing post_obs offsets {missing}")
        # the preprocessor insists on a state for proprio checkpoints; it does not touch the canvas
        state = record.get("proprio")
        frames = [self.canvas_from_pngs(record["pngs"], state=state)["image"]]
        frames += [self.canvas_from_pngs(by_offset[o]["pngs"], state=state)["image"] for o in offsets]
        return frames

    @torch.no_grad()
    def encode_frames(self, frames: list):
        """Encode canvases exactly as training does (preprocess_video + VAE encode_video)."""
        from openwam.model.video_backbone.wan import encode as wan_encode

        vb, arch = self.arch.video_backbone, self.arch
        x = wan_encode.preprocess_video(frames, encoder=vb.video_encoder, dtype=arch.dtype, device=arch.device)
        lat = wan_encode.encode_video(x, vae=vb.vae, encoder=vb.video_encoder)
        return lat.to(dtype=arch.dtype, device=arch.device)

    def inputs_from_record(self, record: dict) -> dict:
        """Rebuild the server-side preprocessed observation from a client chunk record.

        Uses the client's own payload builder and the server's ObsPreprocessor on the saved
        lossless PNGs, so the comparison with the trace exercises the real online path.
        """
        from benchmarks.utils import client as ws_client

        cams = {}
        for label in ("head", "left", "right"):
            entry = (record.get("pngs") or {}).get(label)
            if entry is None:
                cams[label] = None
                continue
            path = Path(entry["path"])
            if sha256_file(path) != entry.get("sha256"):
                raise ValueError(f"PNG sha256 mismatch for {path}")
            cams[label] = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        payload = ws_client.build_payload(
            head=ws_client.encode_numpy_b64(cams["head"]),
            left_wrist=ws_client.encode_numpy_b64(cams["left"]) if cams["left"] is not None else None,
            right_wrist=ws_client.encode_numpy_b64(cams["right"]) if cams["right"] is not None else None,
            prompt=record["prompt"],
            state=record.get("proprio"),
        )
        payload.pop("type", None)
        obs = self.server._obs_preprocessor.preprocess(payload)
        proprio = obs.get("state")
        if proprio is not None:
            proprio = self.arch.normalize_deploy_proprio(proprio)
        return {"image": np.asarray(obs["image"].convert("RGB"), dtype=np.uint8), "prompt": obs.get("prompt"),
                "proprio": proprio}

    # -- full chunk ----------------------------------------------------------------------------
    @torch.no_grad()
    def rerun(self, trace: dict, *, video_seed=None, action_seed=None, replay: dict | None = None,
              record: bool = True, cond_future_override=None) -> dict:
        """Full generation from a trace's conditions. ``cond_future_override`` (IDM only) replaces the
        future frames of the clean stage-2 condition; ``replay`` (joint models) replays a video trajectory."""
        kw = self.generate_kwargs(trace, video_seed=video_seed, action_seed=action_seed)
        probe = {"record": record}
        if cond_future_override is not None:
            probe["cond_future_override"] = cond_future_override
        if replay is not None:
            rmeta = replay["meta"]
            probe["replay_video_steps"] = {int(s["i"]): s["video_latents_in"] for s in replay["steps"]}
            probe["replay_t_v"] = {int(s["i"]): float(s["t_v"]) for s in replay["steps"]}
            probe["replay_video_final"] = replay["final"]["video_latents"]
            probe["replay_schedule"] = [tuple(x) for x in rmeta["schedule"]]
            probe["replay_meta"] = {k: rmeta[k] for k in ("num_frames", "action_num_frames")}
            probe["replay_active_mask"] = rmeta.get("active_action_mask")
        return self.arch.generate(**kw, probe=probe)

    # -- single step -----------------------------------------------------------------------------
    @torch.no_grad()
    def step_response(self, trace: dict, i: int, *, video_latents=None, action_latents=None,
                      inputs_shared: dict | None = None):
        """One joint forward at recorded step ``i``; returns (video_velocity, action_velocity).

        ``video_latents`` replaces the video stream entering the step (the first frame is always
        reset to this trace's own first-frame latents); ``action_latents`` defaults to the recorded
        action state, so only the video side changes.
        """
        arch = self.arch
        step = next(s for s in trace["steps"] if int(s["i"]) == int(i))
        inputs = dict(inputs_shared) if inputs_shared is not None else self._inputs_shared(trace)
        lat = (step["video_latents_in"] if video_latents is None else video_latents)
        lat = lat.to(device=arch.device, dtype=inputs["latents"].dtype).clone()
        ref = inputs.get("first_frame_latents")
        if ref is not None:
            lat[:, :, : ref.shape[2]] = ref
        inputs["latents"] = lat
        al = (step["action_latents_in"] if action_latents is None else action_latents)
        al = al.to(device=arch.device, dtype=arch.dtype)
        v_t = torch.tensor([step["t_v"]], dtype=arch.dtype, device=arch.device)
        a_t = torch.tensor([step["t_a"]], dtype=arch.dtype, device=arch.device)
        torch.compiler.cudagraph_mark_step_begin()
        video_vel, action_vel = arch.forward(al, a_t, **inputs, timestep=v_t)
        return video_vel.detach().cpu(), action_vel.detach().cpu()

    @contextlib.contextmanager
    def attention_mask_mode(self, mode: str):
        """Temporarily rebuild the MoT driver with another cross-modal mask (path-blocking checks)."""
        arch = self.arch
        kwargs = getattr(arch, "_mot_driver_kwargs", None)
        if kwargs is None or "attention_mask_mode" not in kwargs:
            raise ValueError(f"{type(arch).__name__} does not expose a rebuildable attention_mask_mode")
        old_mode, old_driver = kwargs["attention_mask_mode"], arch._mot_driver
        kwargs["attention_mask_mode"] = mode
        arch._mot_driver = None
        arch.build_mot_driver()
        try:
            yield
        finally:
            kwargs["attention_mask_mode"] = old_mode
            arch._mot_driver = old_driver if old_driver is not None else None
            if arch._mot_driver is None:
                arch.build_mot_driver()


def check(runner: OfflineRunner, trace_path: str, record_path: str | None, isolated_step: int | None) -> dict:
    trace = load_trace(trace_path)
    report: dict = {"trace": str(trace_path), "trace_sha256": sha256_file(trace_path),
                    "mask_mode": getattr(runner.arch, "_mot_driver_kwargs", {}).get("attention_mask_mode")}
    if record_path:
        record = json.loads(Path(record_path).read_text())
        receipt = record.get("probe_receipt") or {}
        report["receipt_sha_match"] = receipt.get("sha256") == report["trace_sha256"]
        report["receipt_path_match"] = os.path.realpath(receipt.get("trace_path", "")) == os.path.realpath(trace_path)
        report["record_actions_vs_trace_maxabs"] = _maxabs(record["actions"], trace["final"]["actions_returned"])
        rebuilt = runner.inputs_from_record(record)
        report["record_image_vs_trace_maxabs"] = _maxabs(rebuilt["image"].astype(np.int16),
                                                         np.asarray(trace["cond"]["image"]).astype(np.int16))
        report["record_prompt_match"] = rebuilt["prompt"] == trace["cond"]["prompt"]
        if rebuilt["proprio"] is not None and trace["cond"].get("proprio") is not None:
            report["record_proprio_vs_trace_maxabs"] = _maxabs(rebuilt["proprio"].cpu(), trace["cond"]["proprio"])

    runner.precompute_prompts([trace["cond"]["prompt"]], offload_text_encoder=False)
    inputs = runner._inputs_shared(trace)
    ref = inputs.get("first_frame_latents")
    rec_ref = trace["cond"].get("first_frame_latents")
    if ref is not None and rec_ref is not None:
        report["first_frame_latents_maxabs"] = _maxabs(ref.cpu(), rec_ref)

    out = runner.rerun(trace)
    report["rerun_actions_maxabs"] = _maxabs(out["actions"], trace["final"]["actions_physical"])
    report["rerun_step_action_vel_maxabs"] = max(
        _maxabs(a["action_velocity"], b["action_velocity"])
        for a, b in zip(out["probe_trace"]["steps"], trace["steps"])
    )

    # Fail-closed replay validation: incompatible source traces must raise, never be cast.
    negatives = {}
    bad_dtype = {**trace, "steps": [dict(x) for x in trace["steps"]]}
    bad_dtype["steps"][0]["video_latents_in"] = bad_dtype["steps"][0]["video_latents_in"].float()
    bad_ta = {**trace, "meta": {**trace["meta"], "schedule": [list(x) for x in trace["meta"]["schedule"]]}}
    bad_ta["meta"]["schedule"][1][1] += 1.0
    for name, bad in (("video_dtype_float32", bad_dtype), ("action_timestep_plus1", bad_ta)):
        try:
            runner.rerun(trace, replay=bad, record=False)
            negatives[name] = "NOT RAISED"
        except ValueError as exc:
            negatives[name] = f"rejected: {exc}"
    report["replay_negative_cases"] = negatives
    report["replay_negatives_all_rejected"] = all(v.startswith("rejected") for v in negatives.values())

    noop = []
    for s in trace["steps"]:
        _, av = runner.step_response(trace, s["i"], inputs_shared=inputs)
        noop.append(_maxabs(av, s["action_velocity"]))
    report["step_noop_action_vel_maxabs"] = noop

    if isolated_step is not None:
        s = next(x for x in trace["steps"] if int(x["i"]) == isolated_step)
        g = torch.Generator().manual_seed(1234)
        perturbed = s["video_latents_in"].clone()
        perturbed[:, :, 1:] = torch.randn(perturbed[:, :, 1:].shape, generator=g).to(perturbed.dtype)
        _, base = runner.step_response(trace, isolated_step, inputs_shared=inputs)
        _, pert = runner.step_response(trace, isolated_step, video_latents=perturbed, inputs_shared=inputs)
        report["future_perturb_action_vel_maxabs_own_mask"] = _maxabs(base, pert)
        with runner.attention_mask_mode("isolated"):
            _, base_i = runner.step_response(trace, isolated_step, inputs_shared=inputs)
            _, pert_i = runner.step_response(trace, isolated_step, video_latents=perturbed, inputs_shared=inputs)
        report["future_perturb_action_vel_maxabs_isolated"] = _maxabs(base_i, pert_i)
        report["isolated_step"] = isolated_step
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--ckpt-dir", required=True)
    c.add_argument("--config", default=None)
    c.add_argument("--trace", required=True, nargs="+")
    c.add_argument("--record", nargs="*", default=None, help="client record.json per trace (same order)")
    c.add_argument("--isolated-step", type=int, default=5)
    c.add_argument("--out", default=None)
    args = p.parse_args(argv)

    runner = OfflineRunner(args.ckpt_dir, args.config)
    records = args.record or [None] * len(args.trace)
    if len(records) != len(args.trace):
        raise SystemExit("--record must match --trace in length")
    reports = [check(runner, t, r, args.isolated_step) for t, r in zip(args.trace, records)]
    text = json.dumps(reports, indent=1)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
