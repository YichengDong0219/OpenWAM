"""P1B (Claude): denoising-window localisation of the P1A future clamp on OpenWAM-alpha (Codex R80 design brief).

Per receiver observation (the 26 reused P1A sources), video seed 1001, action seeds 2001/2002:
  C0 x2        native generation (record=True), adapted by p1a_claude.native_trace (seeds/schedule bound)
  identity x2  full self-replay of the C0 trace (all executed steps) -> exact equality on every step and the final
  scatter x6   native at v=1001, a=2003..2008
  windowed x40 for each condition C1/C2/C3/C7_2/C7_3 (targets exactly as P1A) and window W in E=0..3, M=4..6, L=7..9,
               A=0..9 (forward indices of the 10 executed steps): the future latent frames are overwritten ONLY at the
               steps of W with (1 - sigma_i)*target + sigma_i*eps (eps = the receiver's own initial video noise,
               sigma_i = t_v,i / num_train_timesteps), via the opt-in base.generate probe key ``replay_window``; outside W
               the stream evolves normally from whatever state the previous step produced (no C0 back-fill, no reset);
               no replay_video_final. Frame 0 stays the receiver's.
Checks per windowed run: the recorded video input at every step in W equals the constructed clamp (future frames) and
the trace's replay_window equals W; at every executed step outside W the input is NOT the clamp object (it is the
normally evolved state; for steps before the first window step it must equal C0's recorded input bit for bit, since
nothing has been changed yet). For window A the actions must equal the P1A production actions of the same (obs,
condition, seed) bit for bit when ``--p1a-check`` points at the sealed P1A shard dirs (all-step windowed replay is the
P1A clamp; P1A also replayed the final video, which does not touch actions).
Outputs: units/<obs>/<cond>_<W>_v<v>_a<a>.npz + traces/<obs>/<cond>_<W>_v<v>_a<a>.pt (full trace kept for audit),
C0/identity traces, targets/, journal.jsonl (success-only, p1b-claude-unit/1), failures.jsonl; resume refused on any
failure / changed artifact / orphan (as p1a). No metric is computed here.
Usage: p1b_claude.py run --binding B --out D [--p1a-check DIR ...]
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
import p1a_claude as A  # noqa: E402  (pure helpers only; p1a_claude.py bytes unchanged)

FORMAT = "p1b-claude-unit/1"
BINDING_FORMAT = "p1b-runtime-binding/1"
CONDS = ("C1", "C2", "C3", "C7_2", "C7_3")
WINDOWS = {"E": [0, 1, 2, 3], "M": [4, 5, 6], "L": [7, 8, 9], "A": list(range(10))}
FORWARD_CONDS = ("C0", "identity", "scatter", "window")


# ---------------------------------------------------------------- pure (CPU-testable)
def window_replay(c0: dict, target: torch.Tensor, window: list, ntt: float) -> dict:
    """Probe kwargs for a windowed clamp: overwrite only the steps in ``window`` (sorted subset of executed steps)."""
    executed = [int(s["i"]) for s in c0["steps"]]
    if not window or window != sorted(set(window)) or not set(window) <= set(executed):
        raise ValueError(f"window {window} must be a sorted non-empty subset of {executed}")
    eps = c0["init"]["video_noise"]
    steps, t_v = {}, {}
    for s in c0["steps"]:
        i = int(s["i"])
        if i in window:
            dt = s["video_latents_in"].dtype
            steps[i] = A.clamp_latents(s["video_latents_in"], target.to(dt), eps.to(dt), float(s["t_v"]) / float(ntt))
            t_v[i] = float(s["t_v"])
    meta = c0["meta"]
    return {"replay_video_steps": steps, "replay_t_v": t_v, "replay_window": list(window),
            "replay_schedule": [tuple(x) for x in meta["schedule"]],
            "replay_meta": {"num_frames": meta["num_frames"], "action_num_frames": meta["action_num_frames"]},
            "replay_active_mask": meta["active_action_mask"]}


def window_checks(c0: dict, tr: dict, probe: dict, window: list) -> dict:
    """In-window inputs equal the clamp; pre-window inputs equal C0; the recorded window equals W."""
    first = min(window)
    out = {"recorded_window": tr.get("replay_window") == list(window), "n_steps": len(tr["steps"]) == len(c0["steps"])}
    for s, c in zip(tr["steps"], c0["steps"]):
        i = int(s["i"])
        if i in window:
            out[f"step{i}.clamped"] = bool(torch.equal(s["video_latents_in"][:, :, 1:], probe["replay_video_steps"][i][:, :, 1:]))
        elif i < first:
            out[f"step{i}.pre_window_equals_C0"] = bool(torch.equal(s["video_latents_in"], c["video_latents_in"]))
        out[f"step{i}.frame0_receiver"] = bool(torch.equal(s["video_latents_in"][:, :, :1], c["video_latents_in"][:, :, :1]))
    return {"all_passed": all(out.values()), "checks": out}


# ---------------------------------------------------------------- binding
def check_binding(bpath: Path):
    b = json.loads(bpath.read_text())
    if b.get("format") != BINDING_FORMAT or b.get("status") != "jointly_frozen":
        raise SystemExit("binding is not a jointly frozen p1b-runtime-binding/1")
    if not all(isinstance((b.get("signatures") or {}).get(k), str) and b["signatures"][k].strip() for k in ("claude", "codex")):
        raise SystemExit("binding lacks both signatures")
    if b.get("windows") != WINDOWS or b.get("conditions") != list(CONDS):
        raise SystemExit("binding windows / conditions differ from the producer constants")
    man = json.loads(Path(b["runtime_manifest"]["path"]).read_text())
    if A.sha_file(b["runtime_manifest"]["path"]) != b["runtime_manifest"]["sha256"]:
        raise SystemExit("runtime manifest changed")
    arts = man["artifacts"]
    if len({a["path"] for a in arts}) != len(arts):
        raise SystemExit("duplicated artifact path")
    for a in arts:
        if A.sha_file(a["path"]) != a["sha256"]:
            raise SystemExit(f"artifact changed: {a['path']}")
    from g2_common import cfg_sha, resolve_config
    if cfg_sha(resolve_config(str(A.CONFIG), str(A.CKPT))) != man["resolved_config_sha256"]:
        raise SystemExit("resolved deploy config differs from the bound snapshot")
    sm = b["source_manifest"]
    if A.sha_file(sm["path"]) != sm["sha256"]:
        raise SystemExit("source manifest changed")
    for k in ("observations", "video_seeds", "action_seeds", "scatter", "max_reruns"):
        if k not in b:
            raise SystemExit(f"binding lacks {k}")
    return b, man, json.loads(Path(sm["path"]).read_text())


def check_runtime_live(runner, man: dict):
    from g2_common import describe_runtime
    live = describe_runtime(runner)
    diff = sorted(k for k in set(live) | set(man["runtime_description"]) if k != "loaded_repo_files"
                  and live.get(k) != man["runtime_description"].get(k))
    if diff:
        raise SystemExit(f"live runtime differs from the bound description: {diff}")
    bound = {os.path.relpath(a["path"], A.REPO) for a in man["artifacts"]}
    missing = [f for f in live["loaded_repo_files"] if f not in bound]
    if missing:
        raise SystemExit(f"imported repo files not bound: {missing[:8]}")


def audit_outdir(out: Path) -> dict:
    fails = out / "failures.jsonl"
    if fails.exists() and fails.read_text().strip():
        raise SystemExit(f"{out} holds failure records; keep them and use a new directory")
    done, referenced = {}, set()
    j = out / "journal.jsonl"
    if j.exists():
        for n, line in enumerate(j.read_text().splitlines()):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                raise SystemExit(f"journal line {n} is not complete JSON")
            if r.get("format") != FORMAT or r.get("ok") is not True or not r.get("unit"):
                raise SystemExit(f"journal line {n} is not a {FORMAT} success record")
            if r["unit"] in done:
                raise SystemExit(f"duplicate journal unit {r['unit']}")
            for a in r.get("artifacts", []):
                p = Path(a["path"])
                if not p.exists() or A.sha_file(p) != a["sha256"]:
                    raise SystemExit(f"referenced artifact missing or changed: {p}")
                referenced.add(str(p.resolve()))
            done[r["unit"]] = r
    for sub in ("units", "traces", "targets"):
        d = out / sub
        if d.exists():
            for p in d.rglob("*"):
                if p.is_file() and str(p.resolve()) not in referenced:
                    raise SystemExit(f"orphan file: {p}")
    return done


def p1a_actions(dirs) -> dict:
    """(obs, cond, v, a) -> P1A production actions_physical npz path, from the sealed shard journals."""
    out = {}
    for d in dirs:
        for line in (Path(d) / "journal.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r.get("cond") in CONDS:
                arts = [x for x in r["artifacts"] if x["path"].endswith(".npz")]
                if len(arts) != 1 or A.sha_file(arts[0]["path"]) != arts[0]["sha256"]:
                    raise SystemExit(f"P1A reference artifact missing or changed: {r['unit']}")
                out[(r["obs"], r["cond"], r["video_seed"], r["action_seed"])] = arts[0]["path"]
    return out


# ---------------------------------------------------------------- GPU driver
class P1B:
    def __init__(self, out: Path, b: dict, man: dict, manifest: dict, p1a_dirs):
        from offline_runner import OfflineRunner, load_trace
        self.out, self.b, self.manifest = out, b, manifest
        self.done = audit_outdir(out)
        self.p1a = p1a_actions(p1a_dirs) if p1a_dirs else None
        self.records, self.traces, self.targets = {}, {}, {}
        needed = set(b["observations"])
        for o in b["observations"]:
            p = next(x for x in manifest["pairs"] if x["receiver"] == o)
            needed |= {p["C2"]["donor"], p["C3"]["donor"]}
        for o in sorted(needed):
            self.records[o] = A.verify_source(manifest["sources"][o])
            self.traces[o] = load_trace(manifest["sources"][o]["trace"])
        self.runner = OfflineRunner(str(A.CKPT), str(A.CONFIG))
        check_runtime_live(self.runner, man)
        self.ntt = float(self.runner.arch.video_scheduler.num_train_timesteps)
        self.reruns = sum(1 for r in self.done.values() if r.get("kind") in FORWARD_CONDS)

    def fail(self, unit, why):
        with open(self.out / "failures.jsonl", "a") as f:
            f.write(json.dumps({"unit": unit, "error": why, "time": time.time()}) + "\n")
        raise SystemExit(f"FAILED {unit}: {why}")

    def log(self, rec):
        rec = {"format": FORMAT, "ok": True, "time": time.time(), **rec}
        with open(self.out / "journal.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.done[rec["unit"]] = rec

    def art(self, p: Path):
        return {"path": str(p.resolve()), "sha256": A.sha_file(p)}

    def generate(self, obs, v, a, probe_extra=None):
        if self.reruns >= int(self.b["max_reruns"]):
            raise SystemExit(f"hard rerun cap {self.b['max_reruns']} reached")
        self.reruns += 1
        tr0 = self.traces[obs]
        kw = self.runner.generate_kwargs(tr0, video_seed=v, action_seed=a)
        t0 = time.time()
        res = self.runner.arch.generate(**kw, probe={"record": True, **(probe_extra or {})})
        torch.cuda.synchronize()
        pt = res["probe_trace"]
        rw = pt.get("replay_window")
        tr = A.native_trace({k: x for k, x in pt.items() if k not in ("replay_window",)}, tr0, v, a)
        tr["replay_window"] = rw
        return tr, time.time() - t0

    def save(self, unit, obs, kind, cond, window, v, a, tr, el, extra=None):
        name = f"{cond}_{window}_v{v}_a{a}" if window else f"{cond}_v{v}_a{a}"
        up, tp = self.out / "units" / obs / f"{name}.npz", self.out / "traces" / obs / f"{name}.pt"
        A.save_npz(up, {"actions_physical": np.asarray(tr["final"]["actions_physical"], dtype=np.float64),
                        "actions_normalized": np.asarray(tr["final"]["actions_normalized"], dtype=np.float64)})
        A.atomic_save(tp, lambda p: torch.save(tr, p))
        self.log({"unit": unit, "obs": obs, "kind": kind, "cond": cond, "window": window, "video_seed": v, "action_seed": a,
                  "elapsed_s": el, "written_video_latent_sha256": [A.tensor_sha(s["video_latents_in"]) for s in tr["steps"]],
                  "artifacts": [self.art(up), self.art(tp)], **(extra or {})})

    def target(self, obs_id, kind="exec"):
        key = (obs_id, kind)
        if key in self.targets:
            return self.targets[key]
        rec, tr = self.records[obs_id], self.traces[obs_id]
        frames = self.runner.gt_future_frames(rec)
        if kind == "static":
            frames = [frames[0]] * len(frames)
        lat = self.runner.encode_frames(frames).detach().cpu()
        if tuple(lat.shape) != (1, 48, 3, 24, 20):
            raise SystemExit(f"target latent shape {tuple(lat.shape)}")
        first_ok = bool(torch.equal(lat[:, :, :1].to(tr["cond"]["first_frame_latents"].dtype), tr["cond"]["first_frame_latents"]))
        unit, path = f"target|{obs_id}|{kind}", self.out / "targets" / f"{obs_id}__{kind}.pt"
        if unit in self.done:
            if not torch.equal(torch.load(path, map_location="cpu", weights_only=False)["latents"], lat):
                self.fail(unit, "re-encoded target differs from the stored one")
        else:
            A.atomic_save(path, lambda p: torch.save({"latents": lat, "first_latent_equals_trace": first_ok}, p))
            self.log({"unit": unit, "obs": obs_id, "kind": "target", "target_kind": kind, "latent_sha256": A.tensor_sha(lat),
                      "first_latent_equals_trace": first_ok, "artifacts": [self.art(path)]})
        self.targets[key] = (lat, first_ok)
        return self.targets[key]

    def cond_targets(self, obs):
        p = next(x for x in self.manifest["pairs"] if x["receiver"] == obs)
        lat, first_ok = self.target(obs, "exec")
        if not first_ok:
            self.fail(f"target|{obs}|exec", "C1 target first latent != receiver first-frame latent")
        out = {"C1": (lat, {"donor": obs})}
        for c, sc in (("C2", "C7_2"), ("C3", "C7_3")):
            src = p[c]["donor"]
            dl, _ = self.target(src, "exec")
            perm = A.scramble_permutation(src, 2 * 24 * 20)
            out[c] = (dl, {"donor": src})
            out[sc] = (A.scramble_future(dl, perm), {"donor": src, "permutation_sha256": A.sha_bytes(perm.astype(np.int64).tobytes())})
        return out

    def run(self):
        b = self.b
        for obs in b["observations"]:
            self.runner.precompute_prompts([self.traces[obs]["cond"]["prompt"]], offload_text_encoder=False)
            tg = self.cond_targets(obs)
            for v in b["video_seeds"]:
                for a in b["action_seeds"]:
                    u = f"{obs}|C0|{v}|{a}"
                    tp = self.out / "traces" / obs / f"C0_v{v}_a{a}.pt"
                    if u in self.done:
                        c0 = torch.load(tp, map_location="cpu", weights_only=False)
                    else:
                        c0, el = self.generate(obs, v, a)
                        self.save(u, obs, "C0", "C0", None, v, a, c0, el)
                    iu = f"{obs}|identity|{v}|{a}"
                    if iu not in self.done:
                        rep = A.replay_of(c0)
                        probe = {"replay_video_steps": {int(s["i"]): s["video_latents_in"] for s in rep["steps"]},
                                 "replay_t_v": {int(s["i"]): float(s["t_v"]) for s in rep["steps"]},
                                 "replay_video_final": rep["final"]["video_latents"],
                                 "replay_schedule": [tuple(x) for x in rep["meta"]["schedule"]],
                                 "replay_meta": {k: rep["meta"][k] for k in ("num_frames", "action_num_frames")},
                                 "replay_active_mask": rep["meta"]["active_action_mask"]}
                        tr, el = self.generate(obs, v, a, probe)
                        cmp = A.identity_compare(c0, tr)
                        if not cmp["all_equal"]:
                            self.fail(iu, "identity differs: " + ",".join(k for k, ok in cmp["checks"].items() if not ok)[:1500])
                        self.save(iu, obs, "identity", "identity", None, v, a, tr, el, {"identity": cmp})
                    for c in CONDS:
                        target, extra = tg[c]
                        for w, steps in WINDOWS.items():
                            wu = f"{obs}|{c}|{w}|{v}|{a}"
                            if wu in self.done:
                                continue
                            probe = window_replay(c0, target, steps, self.ntt)
                            tr, el = self.generate(obs, v, a, probe)
                            wc = window_checks(c0, tr, probe, steps)
                            if not wc["all_passed"]:
                                self.fail(wu, "window checks failed: " + ",".join(k for k, ok in wc["checks"].items() if not ok))
                            ex = {"window_checks": wc, "target_sha256": A.tensor_sha(target),
                                  "target_coefficients": {int(s["i"]): 1.0 - float(s["t_v"]) / self.ntt for s in c0["steps"] if int(s["i"]) in steps},
                                  **extra}
                            if w == "A" and self.p1a is not None:
                                with np.load(self.p1a[(obs, c, v, a)]) as z:
                                    same = bool(np.array_equal(z["actions_physical"], np.asarray(tr["final"]["actions_physical"], dtype=np.float64)))
                                if not same:
                                    self.fail(wu, "window A actions differ from the P1A production actions of the same unit")
                                ex["equals_p1a_production"] = True
                            self.save(wu, obs, "window", c, w, v, a, tr, el, ex)
            sv, sas = b["scatter"]["video_seed"], b["scatter"]["action_seeds"]
            for a in sas:
                u = f"{obs}|scatter|{sv}|{a}"
                if u not in self.done:
                    tr, el = self.generate(obs, sv, a)
                    self.save(u, obs, "scatter", "scatter", None, sv, a, tr, el)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--binding", required=True); r.add_argument("--out", required=True)
    r.add_argument("--p1a-check", nargs="*", default=[])
    d = sub.add_parser("dry-run"); d.add_argument("--binding", required=True)
    a = ap.parse_args(argv)
    bpath = Path(a.binding).resolve()
    if a.cmd == "dry-run":   # binding + sources only; no model, no output
        b, man, manifest = check_binding(bpath)
        for o in b["observations"]:
            A.verify_source(manifest["sources"][o])
        n = len(b["observations"]) * (len(b["video_seeds"]) * len(b["action_seeds"]) * (2 + len(CONDS) * len(WINDOWS)) + len(b["scatter"]["action_seeds"]))
        print(json.dumps({"binding_sha256": A.sha_file(bpath), "observations": b["observations"], "planned_reruns": n,
                          "max_reruns": b["max_reruns"], "artifacts": len(man["artifacts"])}))
        return
    if subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip():
        raise SystemExit("repository has uncommitted tracked changes; refusing")
    b, man, manifest = check_binding(bpath)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"binding": str(bpath), "binding_sha256": A.sha_file(bpath), "p1a_check": a.p1a_check,
            "commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()}
    rm = out / "run_meta.json"
    if rm.exists():
        if json.loads(rm.read_text()) != meta:
            raise SystemExit("resume with another binding / commit is refused")
    else:
        rm.write_text(json.dumps(meta, indent=1))
    drv = P1B(out, b, man, manifest, a.p1a_check)
    drv.run()
    print(json.dumps({"done_units": len(drv.done), "reruns_this_invocation_total": drv.reruns}))


if __name__ == "__main__":
    main()
