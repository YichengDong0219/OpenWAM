"""P1A (Claude): open-loop future-content intervention on OpenWAM-alpha (R75 protocol; R76 fixes).

For each receiver observation (a recorded G0 chunk: conditioning canvas, prompt, proprio) and receiver seeds (v, a):
  C0        native joint generation (record=True). The NATIVE probe_trace (top-level seeds / schedule /
            active_action_mask / cond / init / steps / final, no meta) is adapted by ``native_trace`` and bound to
            the requested seeds, the receiver's schedule / frame counts and the live active mask.
  identity  replay of the receiver's OWN adapted C0 trace -> must reproduce C0 exactly: every step's i / t_v / t_a /
            sigma / video_latents_in / action_latents_in / video_velocity / action_velocity, n_steps, schedule,
            active mask, the first-frame latents and the complete final (video_latents, action_latents,
            actions_normalized, actions_physical); per-key SHA-256 of both sides is journalled
  C1/C2/C3  clamp the FUTURE latent frames at every step:
            z_i[future] = (1 - sigma_i) * target[future] + sigma_i * eps[future],  sigma_i = t_v,i / num_train_timesteps
            (flow matching add_noise = (1-sigma)*x0 + sigma*noise; t_v=1000 is the noise end); eps = the receiver's
            own initial video noise; frame 0 = the receiver's. C1 own record (self-execution future), C2 same-task
            other-episode donor, C3 other-task donor (fixed by the frozen manifest).
  C7_2/C7_3 the C2 / C3 target with its future (T,H,W) cells permuted by one permutation shared by all channels
            (seed = first 8 bytes big-endian of SHA256("P1A-R75|scramble|<donor_obs_id>"), NumPy PCG64)
  C5        the receiver's current canvas repeated 9 times, encoded by the same path (static future)
  scatter   native generations at v=1001, a=2003..2008
Stage E (engineering): C0 + identity for the binding's fixed observations and seeds, plus CPU endpoint checks of every
condition's clamp; no condition forward. Stage P: all conditions and scatter.

Binding (``--binding``): a jointly frozen p1a-runtime-binding/1 file naming the base runtime manifest (G2
ALPHA_RUNTIME_MANIFEST_Q5_v4: checkpoint / config / repo artifacts + runtime description), extra artifacts (this script
etc.), the source manifest, stage, observations, seeds and the hard rerun cap. Before the model loads: every artifact
SHA, the resolved config SHA, the source manifest SHA and every source's bytes / eligibility. After loading, before the
first forward: the live runtime description equals the bound one and every imported repo file is bound.

Outputs (new dir): units/<obs>/<cond>_v<v>_a<a>.npz, traces/<obs>/{C0,identity}_v<v>_a<a>.pt, targets/<src>__<kind>.pt,
journal.jsonl (append-only SUCCESS records only, each with every artifact path + SHA), failures.jsonl.
Resume: any failure record, unparsable / duplicate journal line, missing or changed referenced artifact, or orphan
file under units/ traces/ targets/ refuses the run (use a new directory). No metric is computed here.

Usage:
  p1a_claude.py bind-extra --out F             write the extra-artifact list (paths + SHA) for the binding draft
  p1a_claude.py run --binding B --out D        run the binding's stage
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
CONDS = ("C1", "C2", "C3", "C7_2", "C7_3", "C5")
FORMAT = "p1a-claude-unit/2"
BINDING_FORMAT = "p1a-runtime-binding/1"
EXTRA = ("scripts/probe/p1a_claude.py",)
CAMERAS = ("head", "left", "right")
OFFSETS = tuple(range(4, 33, 4))
STEP_KEYS = ("video_latents_in", "action_latents_in", "video_velocity", "action_velocity")
STEP_SCALARS = ("i", "t_v", "t_a", "sigma_v", "sigma_a")
FINAL_KEYS = ("video_latents", "action_latents", "actions_normalized", "actions_physical")


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def tensor_sha(t) -> str:
    t = torch.as_tensor(t).detach().cpu().contiguous()
    raw = t.view(torch.int16) if t.dtype == torch.bfloat16 else t
    return sha_bytes(str(t.dtype).encode() + str(tuple(t.shape)).encode() + raw.numpy().tobytes())


# ---------------------------------------------------------------- pure (CPU-testable) construction
def scramble_permutation(donor_obs_id: str, n_cells: int) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(f"P1A-R75|scramble|{donor_obs_id}".encode()).digest()[:8], "big")
    return np.random.Generator(np.random.PCG64(seed)).permutation(n_cells)


def scramble_future(target: torch.Tensor, perm: np.ndarray) -> torch.Tensor:
    if target.ndim != 5 or target.shape[0] != 1 or target.shape[2] < 2:
        raise ValueError(f"target shape {tuple(target.shape)}")
    fut = target[:, :, 1:]
    c, t, h, w = fut.shape[1:]
    if perm.shape != (t * h * w,) or sorted(perm.tolist()) != list(range(t * h * w)):
        raise ValueError("permutation does not match the future cells")
    out = target.clone()
    out[:, :, 1:] = fut.reshape(1, c, t * h * w)[:, :, torch.as_tensor(perm, dtype=torch.long)].reshape(1, c, t, h, w)
    return out


def clamp_latents(step_latents: torch.Tensor, target: torch.Tensor, eps: torch.Tensor, sigma: float) -> torch.Tensor:
    if not (0.0 <= sigma <= 1.0):
        raise ValueError(f"sigma {sigma} outside [0,1]")
    if step_latents.shape != target.shape or eps.shape != target.shape:
        raise ValueError("shape mismatch between step latents, target and eps")
    z = step_latents.clone()
    z[:, :, 1:] = ((1.0 - sigma) * target[:, :, 1:].float() + sigma * eps[:, :, 1:].float()).to(step_latents.dtype)
    return z


def native_trace(pt: dict, receiver: dict, video_seed: int, action_seed: int) -> dict:
    """Adapt a NATIVE probe_trace (as returned by architecture.generate / OfflineRunner.rerun) to the replay format,
    binding it to the requested seeds and the receiver's schedule / frame counts. Raises on any mismatch."""
    for k in ("seeds", "schedule", "active_action_mask", "cond", "init", "steps", "final"):
        if k not in pt:
            raise ValueError(f"native probe trace lacks {k!r}")
    if "meta" in pt:
        raise ValueError("expected a native probe trace (no meta)")
    if (int(pt["seeds"]["video_seed"]), int(pt["seeds"]["action_seed"])) != (int(video_seed), int(action_seed)):
        raise ValueError(f"trace seeds {pt['seeds']} != requested ({video_seed}, {action_seed})")
    sched = [tuple(map(float, x)) for x in pt["schedule"]]
    if sched != [tuple(map(float, x)) for x in receiver["meta"]["schedule"]]:
        raise ValueError("native trace schedule differs from the receiver's")
    if len(pt["steps"]) != len(sched) - 1 or [int(s["i"]) for s in pt["steps"]] != list(range(len(sched) - 1)):
        raise ValueError("native trace steps do not cover the schedule")
    if any(float(s["t_v"]) != sched[int(s["i"])][0] or float(s["t_a"]) != sched[int(s["i"])][1] for s in pt["steps"]):
        raise ValueError("native step timesteps differ from the schedule")
    if "video_noise" not in pt["init"]:
        raise ValueError("native trace lacks init.video_noise")
    meta = {"schedule": sched, "num_frames": int(receiver["meta"]["num_frames"]),
            "action_num_frames": int(receiver["meta"]["action_num_frames"]),
            "active_action_mask": torch.as_tensor(pt["active_action_mask"], dtype=torch.bool),
            "seed": int(pt["seeds"]["seed"]), "video_seed": int(video_seed), "action_seed": int(action_seed)}
    return {"meta": meta, "cond": pt["cond"], "init": pt["init"], "steps": pt["steps"], "final": pt["final"]}


def replay_of(trace: dict) -> dict:
    return {"meta": {k: trace["meta"][k] for k in ("schedule", "num_frames", "action_num_frames", "active_action_mask")},
            "steps": trace["steps"], "final": trace["final"]}


def clamp_replay(c0: dict, target: torch.Tensor, num_train_timesteps: float) -> dict:
    eps = c0["init"]["video_noise"]
    steps = []
    for s in c0["steps"]:
        dt = s["video_latents_in"].dtype
        steps.append({"i": int(s["i"]), "t_v": float(s["t_v"]),
                      "video_latents_in": clamp_latents(s["video_latents_in"], target.to(dt), eps.to(dt),
                                                        float(s["t_v"]) / float(num_train_timesteps))})
    final = target.to(c0["final"]["video_latents"].dtype).clone()
    final[:, :, :1] = c0["final"]["video_latents"][:, :, :1]
    return {"meta": replay_of(c0)["meta"], "steps": steps, "final": {"video_latents": final}}


def endpoint_checks(c0: dict, target: torch.Tensor, num_train_timesteps: float) -> dict:
    rep = clamp_replay(c0, target, num_train_timesteps)
    s0 = next(s for s in c0["steps"] if int(s["i"]) == 0)
    r0 = next(s for s in rep["steps"] if s["i"] == 0)
    dt = s0["video_latents_in"].dtype
    zero = clamp_latents(s0["video_latents_in"], target.to(dt), c0["init"]["video_noise"].to(dt), 0.0)
    return {"sigma0_step_is_noise_end": float(s0["t_v"]) / num_train_timesteps == 1.0,
            "sigma1_equals_receiver_step0": bool(torch.equal(r0["video_latents_in"], s0["video_latents_in"])),
            "sigma0_cpu_equals_target_future": bool(torch.equal(zero[:, :, 1:], target[:, :, 1:].to(dt))),
            "frame0_kept": all(torch.equal(r["video_latents_in"][:, :, :1], s["video_latents_in"][:, :, :1])
                               for r, s in zip(rep["steps"], c0["steps"]))}


def identity_compare(c0: dict, rep: dict) -> dict:
    """Full identity of a self-replay against its C0 (adapted traces). Returns per-key results + both SHAs."""
    out = {"n_steps": len(c0["steps"]) == len(rep["steps"]),
           "schedule": c0["meta"]["schedule"] == rep["meta"]["schedule"],
           "active_action_mask": torch.equal(c0["meta"]["active_action_mask"], rep["meta"]["active_action_mask"]),
           "first_frame_latents": torch.equal(c0["cond"]["first_frame_latents"], rep["cond"]["first_frame_latents"])}
    for x, y in zip(c0["steps"], rep["steps"]):
        for k in STEP_SCALARS:
            out[f"step{int(x['i'])}.{k}"] = x[k] == y[k]
        for k in STEP_KEYS:
            out[f"step{int(x['i'])}.{k}"] = torch.equal(x[k], y[k])
    for k in FINAL_KEYS:
        out[f"final.{k}"] = torch.equal(torch.as_tensor(c0["final"][k]), torch.as_tensor(rep["final"][k]))
    shas = {k: [tensor_sha(c0["final"][k]), tensor_sha(rep["final"][k])] for k in FINAL_KEYS}
    return {"all_equal": bool(all(out.values())), "checks": {k: bool(v) for k, v in out.items()}, "final_sha256": shas}


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
    ex = rec.get("actions_executed") or []
    po = rec.get("post_obs") or []
    ok = (rec.get("executed_count") == 32 and rec.get("inference_horizon") == 32 and rec.get("termination") in (None, "none")
          and rec.get("complete") is True and acts.shape == (32, 20) and np.isfinite(acts).all()
          and [int(e.get("offset", -1)) for e in ex] == list(range(1, 33))
          and all(e.get("count_delta") == 1 and e.get("exception") is None for e in ex)
          and [int(p["offset"]) for p in po] == list(OFFSETS)
          and set(rec.get("pngs") or {}) == set(CAMERAS) and all(set(p.get("pngs") or {}) == set(CAMERAS) for p in po))
    if not ok:
        raise ValueError(f"source {src['record']} not eligible (32 executed steps with count_delta 1 and no exception, "
                         f"offsets 1..32 / 4..32 exactly once, horizon 32, three cameras, [32,20] finite actions)")
    declared = {(int(p["offset"]), p["camera"], p["path"], p["sha256"]) for p in src["pngs"]}
    used = {(0, cam, e["path"], e["sha256"]) for cam, e in rec["pngs"].items()}
    used |= {(int(p["offset"]), cam, e["path"], e["sha256"]) for p in po for cam, e in p["pngs"].items()}
    if declared != used or len(declared) != 27:
        raise ValueError(f"declared PNG set differs from the record's (or is not 9 offsets x 3 cameras) for {src['record']}")
    return rec


def atomic_save(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    write(tmp)
    os.replace(tmp, path)


def save_npz(path: Path, arrays: dict) -> None:
    def w(p):
        with open(p, "wb") as f:
            np.savez(f, **arrays)
    atomic_save(path, w)


def audit_outdir(out: Path) -> dict:
    """Resume audit: success-only journal, every referenced artifact present with its SHA, no orphans, no failures."""
    fails = out / "failures.jsonl"
    if fails.exists() and fails.read_text().strip():
        raise SystemExit(f"{out} holds failure records; keep them and use a new directory after the fix")
    done, referenced = {}, set()
    j = out / "journal.jsonl"
    if j.exists():
        for n, line in enumerate(j.read_text().splitlines()):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                raise SystemExit(f"journal line {n} is not complete JSON; refusing to resume")
            if r.get("format") != FORMAT or r.get("ok") is not True or not r.get("unit"):
                raise SystemExit(f"journal line {n} is not a success record of {FORMAT}")
            if r["unit"] in done:
                raise SystemExit(f"duplicate journal unit {r['unit']}")
            for a in r.get("artifacts", []):
                p = Path(a["path"])
                if not p.exists() or sha_file(p) != a["sha256"]:
                    raise SystemExit(f"referenced artifact missing or changed: {p} (unit {r['unit']})")
                referenced.add(str(p.resolve()))
            done[r["unit"]] = r
    for sub in ("units", "traces", "targets"):
        d = out / sub
        if d.exists():
            for p in d.rglob("*"):
                if p.is_file() and str(p.resolve()) not in referenced:
                    raise SystemExit(f"orphan file not referenced by a success record: {p}")
    return done


def bound_artifacts(binding: dict) -> list:
    base = json.loads(Path(binding["base_manifest"]["path"]).read_text())
    return base["artifacts"] + binding["extra_artifacts"]


def check_binding(bpath: Path) -> tuple[dict, dict, dict]:
    """Everything that needs no model; SystemExit before any GPU work."""
    b = json.loads(bpath.read_text())
    if b.get("format") != BINDING_FORMAT or b.get("status") != "jointly_frozen":
        raise SystemExit("binding is not a jointly frozen p1a-runtime-binding/1")
    if not all(isinstance((b.get("signatures") or {}).get(k), str) and b["signatures"][k].strip() for k in ("claude", "codex")):
        raise SystemExit("binding lacks both signatures")
    bm = b["base_manifest"]
    if sha_file(bm["path"]) != bm["sha256"]:
        raise SystemExit("base runtime manifest changed")
    base = json.loads(Path(bm["path"]).read_text())
    if base.get("checkpoint_dir") != str(CKPT.resolve()) or base.get("deploy_config") != str(CONFIG.resolve()):
        raise SystemExit("base manifest names another checkpoint / config")
    arts = bound_artifacts(b)
    paths = [a["path"] for a in arts]
    if len(set(paths)) != len(paths):
        raise SystemExit("duplicated artifact path")
    for a in arts:
        if sha_file(a["path"]) != a["sha256"]:
            raise SystemExit(f"artifact changed: {a['path']}")
    from g2_common import cfg_sha, resolve_config
    if cfg_sha(resolve_config(str(CONFIG), str(CKPT))) != base["resolved_config_sha256"]:
        raise SystemExit("resolved deploy config differs from the bound snapshot")
    sm = b["source_manifest"]
    if sha_file(sm["path"]) != sm["sha256"]:
        raise SystemExit("source manifest changed")
    manifest = json.loads(Path(sm["path"]).read_text())
    for k in ("stage", "observations", "video_seeds", "action_seeds", "max_reruns"):
        if k not in b:
            raise SystemExit(f"binding lacks {k}")
    if b["stage"] not in ("E", "P"):
        raise SystemExit("binding stage must be E or P")
    return b, base, manifest


def check_runtime_live(runner, base: dict, binding: dict) -> None:
    from g2_common import describe_runtime
    live = describe_runtime(runner)
    want = base["runtime_description"]
    diff = sorted(k for k in set(live) | set(want) if k != "loaded_repo_files" and live.get(k) != want.get(k))
    if diff:
        raise SystemExit(f"live runtime differs from the bound description: {diff}")
    bound = {os.path.relpath(a["path"], REPO) for a in bound_artifacts(binding)}
    missing = [f for f in live["loaded_repo_files"] if f not in bound]
    if missing:
        raise SystemExit(f"imported repo files not bound: {missing[:8]}")


# ---------------------------------------------------------------- GPU driver
class P1A:
    def __init__(self, out: Path, binding: dict, base: dict, manifest: dict):
        from offline_runner import OfflineRunner, load_trace
        self.out, self.b, self.manifest = out, binding, manifest
        self.done = audit_outdir(out)
        self.sources = manifest["sources"]
        self.records, self.traces, self.targets = {}, {}, {}
        needed = set(binding["observations"])
        for o in binding["observations"]:
            p = next(x for x in manifest["pairs"] if x["receiver"] == o)
            needed |= {p["C2"]["donor"], p["C3"]["donor"]}
        for o in sorted(needed):  # every source verified before the model loads
            self.records[o] = verify_source(self.sources[o])
            self.traces[o] = load_trace(self.sources[o]["trace"])
        self.runner = OfflineRunner(str(CKPT), str(CONFIG))
        check_runtime_live(self.runner, base, binding)
        self.ntt = float(self.runner.arch.video_scheduler.num_train_timesteps)
        # the hard cap counts every forward already journalled in this directory (resumes cannot exceed it)
        self.reruns = sum(1 for r in self.done.values() if r.get("cond") in ("C0", "identity", "scatter") + CONDS)

    def fail(self, unit, why):
        with open(self.out / "failures.jsonl", "a") as f:
            f.write(json.dumps({"unit": unit, "error": why, "time": time.time()}) + "\n")
        raise SystemExit(f"FAILED {unit}: {why}")

    def log(self, rec):
        rec = {"format": FORMAT, "ok": True, "time": time.time(), **rec}
        with open(self.out / "journal.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.done[rec["unit"]] = rec

    def art(self, path: Path) -> dict:
        return {"path": str(path.resolve()), "sha256": sha_file(path)}

    def rerun(self, obs, v, a, replay=None):
        if self.reruns >= int(self.b["max_reruns"]):
            raise SystemExit(f"hard rerun cap {self.b['max_reruns']} reached")
        self.reruns += 1
        t0 = time.time()
        res = self.runner.rerun(self.traces[obs], video_seed=v, action_seed=a, replay=replay, record=True)
        torch.cuda.synchronize()
        return native_trace(res["probe_trace"], self.traces[obs], v, a), time.time() - t0, res

    def target(self, obs_id, kind="exec"):
        key = (obs_id, kind)
        if key in self.targets:
            return self.targets[key]
        rec, tr = self.records[obs_id], self.traces[obs_id]
        path = self.out / "targets" / f"{obs_id}__{kind}.pt"
        frames = self.runner.gt_future_frames(rec)
        if kind == "static":
            frames = [frames[0]] * len(frames)
        lat = self.runner.encode_frames(frames).detach().cpu()
        if tuple(lat.shape) != (1, 48, 3, 24, 20):
            raise SystemExit(f"target latent shape {tuple(lat.shape)}")
        first_ok = bool(torch.equal(lat[:, :, :1].to(tr["cond"]["first_frame_latents"].dtype), tr["cond"]["first_frame_latents"]))
        unit = f"target|{obs_id}|{kind}"
        if unit in self.done:
            old = torch.load(path, map_location="cpu", weights_only=False)
            if not torch.equal(old["latents"], lat):
                self.fail(unit, "re-encoded target differs from the stored one")
        else:
            atomic_save(path, lambda p: torch.save({"latents": lat, "first_latent_equals_trace": first_ok}, p))
            self.log({"unit": unit, "obs": obs_id, "kind": kind, "latent_sha256": tensor_sha(lat),
                      "first_latent_equals_trace": first_ok, "artifacts": [self.art(path)]})
        self.targets[key] = (lat, first_ok)
        return self.targets[key]

    def save_run(self, unit, obs, cond, v, a, tr, elapsed, trace_path=None, extra=None):
        upath = self.out / "units" / obs / f"{cond}_v{v}_a{a}.npz"
        save_npz(upath, {"actions_physical": np.asarray(tr["final"]["actions_physical"], dtype=np.float64),
                         "actions_normalized": np.asarray(tr["final"]["actions_normalized"], dtype=np.float64)})
        arts = [self.art(upath)]
        if trace_path is not None:
            atomic_save(trace_path, lambda p: torch.save(tr, p))
            arts.append(self.art(trace_path))
        self.log({"unit": unit, "obs": obs, "cond": cond, "video_seed": v, "action_seed": a, "elapsed_s": elapsed,
                  "written_video_latent_sha256": [tensor_sha(s["video_latents_in"]) for s in tr["steps"]],
                  "artifacts": arts, **(extra or {})})

    def c0(self, obs, v, a):
        unit, tpath = f"{obs}|C0|{v}|{a}", self.out / "traces" / obs / f"C0_v{v}_a{a}.pt"
        if unit in self.done:
            c0 = torch.load(tpath, map_location="cpu", weights_only=False)
        else:
            c0, el, _ = self.rerun(obs, v, a)
            self.save_run(unit, obs, "C0", v, a, c0, el, trace_path=tpath)
        iu = f"{obs}|identity|{v}|{a}"
        if iu not in self.done:
            rep, el, _ = self.rerun(obs, v, a, replay=replay_of(c0))
            cmp = identity_compare(c0, rep)
            if not cmp["all_equal"]:
                self.fail(iu, "identity replay differs: " + ", ".join(k for k, ok in cmp["checks"].items() if not ok)[:2000])
            self.save_run(iu, obs, "identity", v, a, rep, el, trace_path=self.out / "traces" / obs / f"identity_v{v}_a{a}.pt",
                          extra={"identity": cmp})
        return c0

    def condition_targets(self, obs):
        p = next(x for x in self.manifest["pairs"] if x["receiver"] == obs)
        lat, first_ok = self.target(obs, "exec")
        out = {"C1": (lat, {"donor": obs, "first_latent_equals_trace": first_ok}),
               "C5": (self.target(obs, "static")[0], {"donor": obs, "static": True})}
        for c, sc in (("C2", "C7_2"), ("C3", "C7_3")):
            src = p[c]["donor"]
            dl, _ = self.target(src, "exec")
            perm = scramble_permutation(src, 2 * 24 * 20)
            out[c] = (dl, {"donor": src})
            out[sc] = (scramble_future(dl, perm), {"donor": src, "permutation_sha256": sha_bytes(perm.astype(np.int64).tobytes())})
        return out

    def endpoints(self, obs, v, a, c0, tg):
        unit = f"{obs}|endpoints|{v}|{a}"
        if unit in self.done:
            return
        res = {c: endpoint_checks(c0, t, self.ntt) for c, (t, _) in tg.items()}
        ok = tg["C1"][1]["first_latent_equals_trace"] and all(all(r.values()) for r in res.values())
        if not ok:
            self.fail(unit, f"endpoint checks failed: {res}")
        self.log({"unit": unit, "obs": obs, "cond": "endpoints", "video_seed": v, "action_seed": a, "endpoints": res,
                  "C1_first_latent_equals_trace": True, "artifacts": []})

    def conditions(self, obs, v, a, c0, tg):
        for c in CONDS:
            unit = f"{obs}|{c}|{v}|{a}"
            if unit in self.done:
                continue
            target, extra = tg[c]
            rep = clamp_replay(c0, target, self.ntt)
            tr, el, _ = self.rerun(obs, v, a, replay=rep)
            if not all(torch.equal(g["video_latents_in"][:, :, 1:], w["video_latents_in"][:, :, 1:])
                       for g, w in zip(tr["steps"], rep["steps"])):
                self.fail(unit, "written future latents differ from the constructed clamp")
            self.save_run(unit, obs, c, v, a, tr, el, extra={"target_sha256": tensor_sha(target), **extra})

    def run(self):
        b = self.b
        for obs in b["observations"]:
            self.runner.precompute_prompts([self.traces[obs]["cond"]["prompt"]], offload_text_encoder=False)
            tg = self.condition_targets(obs)
            for v in b["video_seeds"]:
                for a in b["action_seeds"]:
                    c0 = self.c0(obs, v, a)
                    self.endpoints(obs, v, a, c0, tg)
                    if b["stage"] == "P":
                        self.conditions(obs, v, a, c0, tg)
            if b["stage"] == "P":
                sv, sas = b["scatter"]["video_seed"], b["scatter"]["action_seeds"]
                for a in sas:
                    unit = f"{obs}|scatter|{sv}|{a}"
                    if unit not in self.done:
                        tr, el, _ = self.rerun(obs, sv, a)
                        self.save_run(unit, obs, "scatter", sv, a, tr, el)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("bind-extra"); e.add_argument("--out", required=True)
    r = sub.add_parser("run"); r.add_argument("--binding", required=True); r.add_argument("--out", required=True)
    r.add_argument("--shard")
    a = ap.parse_args(argv)
    if a.cmd == "bind-extra":
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        arts = [{"path": str((REPO / f).resolve()), "sha256": sha_file(REPO / f)} for f in EXTRA]
        Path(a.out).write_text(json.dumps({"commit": commit, "extra_artifacts": arts}, indent=1))
        print(json.dumps(arts))
        return
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit("repository has uncommitted tracked changes; refusing")
    bpath = Path(a.binding).resolve()
    binding, base, manifest = check_binding(bpath)
    if a.shard:
        k, n = map(int, a.shard.split("/"))
        binding = dict(binding, observations=[o for o in binding["observations"]
                                              if int(hashlib.sha256(o.encode()).hexdigest(), 16) % n == k])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"binding": str(bpath), "binding_sha256": sha_file(bpath), "shard": a.shard,
            "commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()}
    rm = out / "run_meta.json"
    if rm.exists():
        if json.loads(rm.read_text()) != meta:
            raise SystemExit("resume with another binding / shard / commit is refused")
    else:
        rm.write_text(json.dumps(meta, indent=1))
    drv = P1A(out, binding, base, manifest)
    drv.run()
    print(json.dumps({"done_units": len(drv.done), "reruns_this_invocation": drv.reruns, "obs": binding["observations"]}))


if __name__ == "__main__":
    main()
