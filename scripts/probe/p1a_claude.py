"""P1A (Claude): open-loop future-content intervention on OpenWAM-alpha, per the R75 candidate protocol.

For each receiver observation (a recorded G0 chunk: conditioning canvas, prompt, proprio) and receiver seeds (v, a):
  C0        native joint generation (record=True; this trace is the base and the clamp's epsilon source)
  identity  replay of the receiver's OWN C0 trace -> must reproduce C0 bit for bit (steps + final) before any
            condition of that (obs, seed) is kept
  C1/C2/C3  clamp the FUTURE latent frames to a clean target from an EXECUTED record, at every denoising step:
            z_i[future] = (1 - sigma_i) * target[future] + sigma_i * eps[future],  sigma_i = t_v,i / num_train_timesteps
            (OpenWAM flow matching, add_noise = (1-sigma)*x0 + sigma*noise; t_v=1000 is the noise end), eps = the
            receiver's own initial video noise (trace init.video_noise); the receiver's first frame is kept (the
            replay hook resets it anyway). C1 = own record (self-execution future), C2 = same-task other-episode
            donor, C3 = other-task donor (donors fixed by the manifest).
  C7_2/C7_3 the C2 / C3 clean target with its future (T,H,W) latent cells permuted by ONE fixed permutation shared by
            all channels (seed = first 8 bytes big-endian of SHA256("P1A-R75|scramble|<donor_obs_id>"), NumPy PCG64)
  C5        the receiver's current canvas repeated 9 times, encoded by the same path (static future)
  scatter   native generations at v=1001, a=2003..2008 (with C0's a=2001,2002 -> 8 action seeds at one video seed)
Targets: 9 canvases (request frame + post_obs offsets 4..32, OfflineRunner.gt_future_frames) -> encode_frames (the
training VAE path) -> [1,48,3,24,20]; frames 1..2 are the future. For C1 the encoded first latent must equal the
receiver's first_frame_latents (path check). Every source is re-verified (trace / record / every PNG SHA, full
32-step execution, offsets 4..32, actions [32,20]) before use.

Outputs (new dir, never overwritten): units/<obs>/<cond>_v<v>_a<a>.npz (actions_physical / actions_normalized and the
SHA-256 of every written video latent), c0 traces under traces/, targets/<source>.pt, journal.jsonl (append-only,
one record per unit), failures.jsonl. A unit file present on resume is re-verified (its journal record and SHA);
units are written atomically. No metric or effect is computed here (Codex's p1a_metrics reads the units).

Usage:
  p1a_claude.py describe --out D                     runtime binding description (model, config, repo commit)
  p1a_claude.py run --manifest M --out D --stage E|P [--obs ID ...] [--shard K/N] [--max-runs N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "probe"))

CKPT = REPO / "assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboTwin-Full"
CONFIG = REPO / "configs/deploy_probe.yaml"
VIDEO_SEEDS, ACTION_SEEDS = (1001, 1002, 1003, 1004), (2001, 2002)
SCATTER = (1001, (2003, 2004, 2005, 2006, 2007, 2008))
CONDS = ("C1", "C2", "C3", "C7_2", "C7_3", "C5")
FORMAT = "p1a-claude-unit/1"


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def tensor_sha(t: torch.Tensor) -> str:
    t = t.detach().cpu().contiguous()
    return sha_bytes(t.view(torch.uint8).numpy().tobytes() if t.dtype == torch.bfloat16 else t.numpy().tobytes())


# ---------------------------------------------------------------- pure (CPU-testable) construction
def scramble_permutation(donor_obs_id: str, n_cells: int) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(f"P1A-R75|scramble|{donor_obs_id}".encode()).digest()[:8], "big")
    return np.random.Generator(np.random.PCG64(seed)).permutation(n_cells)


def scramble_future(target: torch.Tensor, perm: np.ndarray) -> torch.Tensor:
    """Permute the future (T,H,W) cells of [1,C,T,H,W] with one permutation shared by all channels; frame 0 kept."""
    if target.ndim != 5 or target.shape[0] != 1 or target.shape[2] < 2:
        raise ValueError(f"target shape {tuple(target.shape)}")
    fut = target[:, :, 1:]
    c, t, h, w = fut.shape[1:]
    if perm.shape != (t * h * w,) or sorted(perm.tolist()) != list(range(t * h * w)):
        raise ValueError("permutation does not match the future cells")
    flat = fut.reshape(1, c, t * h * w)
    out = target.clone()
    out[:, :, 1:] = flat[:, :, torch.as_tensor(perm, dtype=torch.long)].reshape(1, c, t, h, w)
    return out


def clamp_latents(step_latents: torch.Tensor, target: torch.Tensor, eps: torch.Tensor, sigma: float) -> torch.Tensor:
    """Future frames := (1-sigma)*target + sigma*eps (computed in float32, cast to the step dtype); frame 0 kept."""
    if not (0.0 <= sigma <= 1.0):
        raise ValueError(f"sigma {sigma} outside [0,1]")
    if step_latents.shape != target.shape or eps.shape != target.shape:
        raise ValueError("shape mismatch between step latents, target and eps")
    z = step_latents.clone()
    mix = (1.0 - sigma) * target[:, :, 1:].float() + sigma * eps[:, :, 1:].float()
    z[:, :, 1:] = mix.to(step_latents.dtype)
    return z


def clamp_replay(c0_trace: dict, target: torch.Tensor, num_train_timesteps: float) -> dict:
    """A replay dict (OfflineRunner.rerun format) that clamps the receiver's future to ``target`` at every step."""
    eps = c0_trace["init"]["video_noise"]
    steps = []
    for s in c0_trace["steps"]:
        sigma = float(s["t_v"]) / float(num_train_timesteps)
        steps.append({"i": int(s["i"]), "t_v": float(s["t_v"]),
                      "video_latents_in": clamp_latents(s["video_latents_in"], target.to(s["video_latents_in"].dtype), eps.to(s["video_latents_in"].dtype), sigma)})
    final = target.to(c0_trace["final"]["video_latents"].dtype).clone()
    final[:, :, :1] = c0_trace["final"]["video_latents"][:, :, :1]
    meta = {k: c0_trace["meta"][k] for k in ("schedule", "num_frames", "action_num_frames", "active_action_mask")}
    return {"meta": meta, "steps": steps, "final": {"video_latents": final}}


def endpoint_checks(c0_trace: dict, target: torch.Tensor, num_train_timesteps: float) -> dict:
    """sigma=1 step equals the receiver's own step-0 latents bit for bit (target-independent); CPU sigma=0 = target."""
    rep = clamp_replay(c0_trace, target, num_train_timesteps)
    s0 = next(s for s in c0_trace["steps"] if int(s["i"]) == 0)
    r0 = next(s for s in rep["steps"] if s["i"] == 0)
    zero = clamp_latents(s0["video_latents_in"], target.to(s0["video_latents_in"].dtype),
                         c0_trace["init"]["video_noise"].to(s0["video_latents_in"].dtype), 0.0)
    return {"sigma0_step_is_noise_end": float(s0["t_v"]) / num_train_timesteps == 1.0,
            "sigma1_equals_receiver_step0": bool(torch.equal(r0["video_latents_in"], s0["video_latents_in"])),
            "sigma0_cpu_equals_target_future": bool(torch.equal(zero[:, :, 1:], target[:, :, 1:].to(zero.dtype))),
            "frame0_kept": all(torch.equal(r["video_latents_in"][:, :, :1], s["video_latents_in"][:, :, :1])
                               for r, s in zip(rep["steps"], c0_trace["steps"]))}


def verify_source(src: dict) -> dict:
    """Re-verify a manifest source's bytes and eligibility; returns the loaded record."""
    if sha_file(src["trace"]) != src["trace_sha256"]:
        raise ValueError(f"trace sha mismatch {src['trace']}")
    if sha_file(src["record"]) != src["record_sha256"]:
        raise ValueError(f"record sha mismatch {src['record']}")
    for p in src["pngs"]:
        if sha_file(p["path"]) != p["sha256"]:
            raise ValueError(f"png sha mismatch {p['path']}")
    rec = json.loads(Path(src["record"]).read_text())
    acts = np.asarray(rec["actions"], dtype=np.float64)
    offs = sorted(int(p["offset"]) for p in rec.get("post_obs", []))
    if not (rec.get("executed_count") == 32 and rec.get("termination") in (None, "none") and rec.get("complete") is True
            and acts.shape == (32, 20) and np.isfinite(acts).all() and offs == list(range(4, 33, 4))
            and len(rec.get("actions_executed") or []) == 32):
        raise ValueError(f"source {src['record']} not eligible (full 32-step execution, offsets 4..32, [32,20] actions)")
    declared = {(int(p["offset"]), p["camera"], p["path"], p["sha256"]) for p in src["pngs"]}
    used = {(0, cam, e["path"], e["sha256"]) for cam, e in rec["pngs"].items()}
    used |= {(int(po["offset"]), cam, e["path"], e["sha256"]) for po in rec["post_obs"] for cam, e in po["pngs"].items()}
    if declared != used:
        raise ValueError(f"declared PNG set differs from the record's for {src['record']}")
    return rec


def atomic_save(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    write(tmp)
    os.replace(tmp, path)


def repo_state() -> dict:
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    return {"commit": commit, "tracked_clean": dirty == "", "script_sha256": sha_file(__file__)}


# ---------------------------------------------------------------- GPU driver
class P1A:
    def __init__(self, out: Path, manifest: dict, manifest_sha: str):
        from offline_runner import OfflineRunner
        self.out, self.manifest, self.manifest_sha = out, manifest, manifest_sha
        self.runner = OfflineRunner(str(CKPT), str(CONFIG))
        self.ntt = float(self.runner.arch.video_scheduler.num_train_timesteps)
        self.journal = out / "journal.jsonl"
        self.done = {}
        if self.journal.exists():
            for line in self.journal.read_text().splitlines():
                r = json.loads(line)
                self.done[r["unit"]] = r
        self.sources = manifest["sources"]
        self.records, self.traces, self.targets = {}, {}, {}

    # sources
    def source(self, obs_id):
        if obs_id not in self.records:
            from offline_runner import load_trace
            src = self.sources[obs_id]
            self.records[obs_id] = verify_source(src)
            self.traces[obs_id] = load_trace(src["trace"])
        return self.records[obs_id], self.traces[obs_id]

    def target(self, obs_id, kind="exec"):
        key = (obs_id, kind)
        if key in self.targets:
            return self.targets[key]
        rec, tr = self.source(obs_id)
        path = self.out / "targets" / f"{obs_id}__{kind}.pt"
        frames = self.runner.gt_future_frames(rec)
        if kind == "static":
            frames = [frames[0]] * len(frames)
        lat = self.runner.encode_frames(frames).detach().cpu()
        if tuple(lat.shape) != (1, 48, 3, 24, 20):
            raise ValueError(f"target latent shape {tuple(lat.shape)}")
        first_ok = torch.equal(lat[:, :, :1].to(tr["cond"]["first_frame_latents"].dtype), tr["cond"]["first_frame_latents"])
        if path.exists():
            old = torch.load(path, map_location="cpu", weights_only=False)
            if not torch.equal(old["latents"], lat):
                raise ValueError(f"re-encoded target differs from the stored one: {path}")
        else:
            atomic_save(path, lambda p: torch.save({"latents": lat, "first_latent_equals_trace": first_ok,
                                                     "canvas_sha256": [tensor_sha(torch.as_tensor(np.asarray(f.convert("RGB"))))
                                                                        for f in frames]}, p))
        self.targets[key] = (lat, first_ok)
        return self.targets[key]

    # units
    def log(self, rec):
        with open(self.journal, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.done[rec["unit"]] = rec

    def fail(self, unit, why):
        with open(self.out / "failures.jsonl", "a") as f:
            f.write(json.dumps({"unit": unit, "error": why, "time": time.time()}) + "\n")

    def save_unit(self, unit, obs, cond, v, a, result, extra):
        path = self.out / "units" / obs / f"{cond}_v{v}_a{a}.npz"
        tr = result["probe_trace"]
        phys = np.asarray(result["actions"]) if "actions" in result else None
        fin = tr["final"]
        arrays = {"actions_physical": np.asarray(fin["actions_physical"], dtype=np.float64),
                  "actions_normalized": np.asarray(fin["actions_normalized"], dtype=np.float64)}
        written = [tensor_sha(s["video_latents_in"]) for s in tr["steps"]]
        def write(p):
            with open(p, "wb") as f:
                np.savez(f, **arrays)
        atomic_save(path, write)
        rec = {"format": FORMAT, "unit": unit, "obs": obs, "cond": cond, "video_seed": v, "action_seed": a,
               "path": str(path), "sha256": sha_file(path), "written_video_latent_sha256": written,
               "returned_actions_sha256": sha_bytes(np.ascontiguousarray(phys).tobytes()) if phys is not None else None,
               "time": time.time(), **extra}
        self.log(rec)
        return rec

    def run_unit(self, obs, cond, v, a, replay=None, extra=None):
        unit = f"{obs}|{cond}|{v}|{a}"
        if unit in self.done:
            r = self.done[unit]
            if sha_file(r["path"]) != r["sha256"]:
                raise ValueError(f"stored unit changed on disk: {unit}")
            return None, r
        _, tr = self.source(obs)
        t0 = time.time()
        res = self.runner.rerun(tr, video_seed=v, action_seed=a, replay=replay, record=True)
        torch.cuda.synchronize()
        return res, {"unit": unit, "elapsed_s": time.time() - t0, **(extra or {})}

    def c0(self, obs, v, a):
        """C0 + identity for one (obs, seed); returns the C0 trace (kept on disk for the clamp's epsilon)."""
        tpath = self.out / "traces" / obs / f"C0_v{v}_a{a}.pt"
        res, info = self.run_unit(obs, "C0", v, a)
        if res is None:
            c0t = torch.load(tpath, map_location="cpu", weights_only=False)
        else:
            c0t = res["probe_trace"]
            atomic_save(tpath, lambda p: torch.save(c0t, p))
            self.save_unit(info["unit"], obs, "C0", v, a, res, {"elapsed_s": info["elapsed_s"], "trace": str(tpath),
                                                               "trace_sha256": sha_file(tpath)})
        ident_unit = f"{obs}|identity|{v}|{a}"
        if ident_unit not in self.done:
            rep = {"meta": {k: c0t["meta"][k] for k in ("schedule", "num_frames", "action_num_frames", "active_action_mask")},
                   "steps": c0t["steps"], "final": c0t["final"]}
            res2, info2 = self.run_unit(obs, "identity", v, a, replay=rep)
            t2 = res2["probe_trace"]
            same_steps = all(torch.equal(x[k], y[k]) for x, y in zip(c0t["steps"], t2["steps"])
                             for k in ("video_latents_in", "action_latents_in", "video_velocity", "action_velocity"))
            same_final = all(torch.equal(torch.as_tensor(c0t["final"][k]), torch.as_tensor(t2["final"][k]))
                             for k in ("action_latents", "actions_normalized", "actions_physical"))
            if not (same_steps and same_final and len(c0t["steps"]) == len(t2["steps"])):
                self.fail(ident_unit, f"identity replay differs (steps {same_steps}, final {same_final})")
                raise SystemExit(f"identity replay failed for {ident_unit}; nothing further for this unit")
            self.save_unit(ident_unit, obs, "identity", v, a, res2, {"elapsed_s": info2["elapsed_s"], "identity_bitwise": True})
        return c0t

    def conditions(self, obs, v, a, c0t, conds):
        pairs = next(p for p in self.manifest["pairs"] if p["receiver"] == obs)
        tgt = {}
        lat, first_ok = self.target(obs, "exec")
        if "C1" in conds:
            if not first_ok:
                raise SystemExit(f"C1 target first latent != receiver first-frame latent for {obs}")
            tgt["C1"] = (lat, {"donor": obs})
        for c, src in (("C2", pairs["C2"]["donor"]), ("C3", pairs["C3"]["donor"])):
            dl, _ = self.target(src, "exec")
            if c in conds:
                tgt[c] = (dl, {"donor": src})
            sc = "C7_2" if c == "C2" else "C7_3"
            if sc in conds:
                perm = scramble_permutation(src, 2 * 24 * 20)
                tgt[sc] = (scramble_future(dl, perm), {"donor": src, "permutation_sha256": sha_bytes(perm.astype(np.int64).tobytes())})
        if "C5" in conds:
            tgt["C5"] = (self.target(obs, "static")[0], {"donor": obs, "static": True})
        for c in conds:
            target, extra = tgt[c]
            ep = endpoint_checks(c0t, target, self.ntt)
            if not all(ep.values()):
                raise SystemExit(f"endpoint check failed {obs} {c}: {ep}")
            rep = clamp_replay(c0t, target, self.ntt)
            res, info = self.run_unit(obs, c, v, a, replay=rep)
            if res is None:
                continue
            got = [s["video_latents_in"] for s in res["probe_trace"]["steps"]]
            want = [s["video_latents_in"] for s in rep["steps"]]
            if not all(torch.equal(g[:, :, 1:], w[:, :, 1:]) for g, w in zip(got, want)):
                self.fail(info["unit"], "written future latents differ from the constructed clamp")
                raise SystemExit(f"clamp not applied as constructed: {info['unit']}")
            self.save_unit(info["unit"], obs, c, v, a, res, {"elapsed_s": info["elapsed_s"], "endpoints": ep,
                                                            "target_sha256": tensor_sha(target), **extra})

    def endpoint_only(self, obs, c0t):
        """Stage E: build every condition's clean target and check the clamp endpoints (no condition forward)."""
        pairs = next(p for p in self.manifest["pairs"] if p["receiver"] == obs)
        lat, first_ok = self.target(obs, "exec")
        out = {"C1_first_latent_equals_trace": bool(first_ok)}
        cands = {"C1": lat, "C5": self.target(obs, "static")[0]}
        for c, src in (("C2", pairs["C2"]["donor"]), ("C3", pairs["C3"]["donor"])):
            dl, _ = self.target(src, "exec")
            cands[c] = dl
            cands["C7_2" if c == "C2" else "C7_3"] = scramble_future(dl, scramble_permutation(src, 2 * 24 * 20))
        for c, t in cands.items():
            out[c] = endpoint_checks(c0t, t, self.ntt)
        return out

    def run(self, obs_ids, stage, max_runs):
        n = 0
        for obs in obs_ids:
            self.runner.precompute_prompts([self.source(obs)[1]["cond"]["prompt"]], offload_text_encoder=False)
            for v in VIDEO_SEEDS:
                for a in ACTION_SEEDS:
                    c0t = self.c0(obs, v, a)
                    if stage == "E":
                        unit = f"{obs}|endpoints|{v}|{a}"
                        if unit not in self.done:
                            ep = self.endpoint_only(obs, c0t)
                            ok = ep["C1_first_latent_equals_trace"] and all(all(x.values()) for k, x in ep.items() if isinstance(x, dict))
                            self.log({"format": FORMAT, "unit": unit, "obs": obs, "cond": "endpoints", "video_seed": v,
                                      "action_seed": a, "endpoints": ep, "all_passed": bool(ok), "time": time.time()})
                            if not ok:
                                self.fail(unit, f"endpoint checks failed: {ep}")
                                raise SystemExit(f"endpoint checks failed for {unit}")
                    if stage == "P":
                        self.conditions(obs, v, a, c0t, CONDS)
                    n += 1
                    if max_runs and n >= max_runs:
                        return
            if stage == "P":
                for a in SCATTER[1]:
                    res, info = self.run_unit(obs, "scatter", SCATTER[0], a)
                    if res is not None:
                        self.save_unit(info["unit"], obs, "scatter", SCATTER[0], a, res, {"elapsed_s": info["elapsed_s"]})


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("describe"); d.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--manifest", required=True); r.add_argument("--out", required=True)
    r.add_argument("--stage", choices=("E", "P"), required=True)
    r.add_argument("--obs", nargs="*"); r.add_argument("--shard"); r.add_argument("--max-runs", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "describe":
        info = {"repo": repo_state(), "checkpoint": str(CKPT), "config": str(CONFIG), "config_sha256": sha_file(CONFIG),
                "torch": torch.__version__, "cuda": torch.version.cuda, "python": sys.version.split()[0]}
        Path(a.out).write_text(json.dumps(info, indent=1))
        print(json.dumps(info))
        return
    mpath = Path(a.manifest)
    manifest = json.loads(mpath.read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rs = repo_state()
    if not rs["tracked_clean"]:
        raise SystemExit("repository has uncommitted tracked changes; refusing (bind a committed tree)")
    run_meta = out / "run_meta.json"
    meta = {"manifest": str(mpath.resolve()), "manifest_sha256": sha_file(mpath), "repo": rs, "stage": a.stage,
            "checkpoint": str(CKPT), "config_sha256": sha_file(CONFIG)}
    if run_meta.exists():
        old = json.loads(run_meta.read_text())
        if {k: old[k] for k in ("manifest_sha256", "repo", "stage", "config_sha256")} != {k: meta[k] for k in ("manifest_sha256", "repo", "stage", "config_sha256")}:
            raise SystemExit("resume with a different manifest / code / stage / config is refused")
    else:
        run_meta.write_text(json.dumps(meta, indent=1))
    obs_ids = a.obs or (manifest["engineering_receivers"] if a.stage == "E" else sorted(p["receiver"] for p in manifest["pairs"]))
    if a.shard:
        k, n = map(int, a.shard.split("/"))
        obs_ids = [o for o in obs_ids if int(hashlib.sha256(o.encode()).hexdigest(), 16) % n == k]
    drv = P1A(out, manifest, meta["manifest_sha256"])
    drv.run(obs_ids, a.stage, a.max_runs)
    print(json.dumps({"done_units": len(drv.done), "obs": obs_ids}))


if __name__ == "__main__":
    main()
