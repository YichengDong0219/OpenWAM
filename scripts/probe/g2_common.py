"""Shared protections for the G2 wrappers (g2_q5.py, g2_q6.py) and the runtime manifest (R24 review).

* ``check_binding``  - before any GPU work or output write: binding/manifest format, scope, status,
  both signatures, prereg, review records, every artifact sha, the deploy config and checkpoint the
  run will actually use, and the resolved (pre-load) deploy config hash.
* ``describe_runtime`` / ``check_runtime`` - after the model is loaded and before any generation:
  engine config hash, architecture, dtype, action dim, attention modes, schedule, inactive dims,
  library versions and every imported repo source file must equal the manifest.
* ``Ledger`` - the output directory: RUN_STAMP, an index keyed by semantic unit, no duplicate keys,
  no orphan files, failures kept apart from completed units, and a final exact key-set check.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
PROBE_ROOT = (REPO / "outputs" / "probe").resolve()
BINDING_FORMAT = "openwam-runtime-binding/2"
MANIFEST_FORMAT = "openwam-runtime-manifest/2"
LEDGER_FILES = {"RUN_STAMP.json", "index.jsonl", "failures.jsonl", "source_checks.jsonl"}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def same(a, b) -> bool:
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, (float, int)) or isinstance(b, (float, int)):
        return float(a) == float(b)
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)


def append_jsonl(path: Path, row: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def atomic_torch_save(obj, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    if path.exists():
        tmp.unlink()
        raise FileExistsError(f"refusing to overwrite {path}")
    os.replace(tmp, path)
    return sha256_file(path)


# -- configuration -------------------------------------------------------------------------------
def resolve_config(config: str, ckpt_dir: str):
    """The deploy config exactly as the server resolves it, without loading a model."""
    from openwam.deploy.server import _build_argparser, resolve_deploy_args

    args = _build_argparser().parse_args(["--config", str(config), "--ckpt-dir", str(ckpt_dir)])
    cfg, _ = resolve_deploy_args(args)
    return cfg


def cfg_text(cfg) -> str:
    """Resolved YAML; unresolved YAML when an interpolation cannot be resolved (e.g. ``${now:...}``)."""
    from omegaconf import OmegaConf

    try:
        return OmegaConf.to_yaml(cfg, resolve=True)
    except Exception:
        return OmegaConf.to_yaml(cfg, resolve=False)


def cfg_sha(cfg) -> str:
    return hashlib.sha256(cfg_text(cfg).encode("utf-8")).hexdigest()


def is_hex64(x) -> bool:
    return isinstance(x, str) and len(x) == 64 and all(c in "0123456789abcdef" for c in x)


def check_binding(binding_path, prereg_path, scope: str, config: str, ckpt_dir: str) -> dict:
    """All binding checks that need no model. Raises SystemExit before any GPU or output work."""
    prereg_sha = sha256_file(prereg_path)
    bind_path = Path(binding_path).resolve()
    bind = json.loads(bind_path.read_text())
    if bind.get("format") != BINDING_FORMAT or bind.get("scope") != scope or bind.get("status") != "jointly_bound":
        raise SystemExit(f"runtime binding is not a jointly bound {scope} binding")
    if not all(isinstance((bind.get("signatures") or {}).get(k), str) and bind["signatures"][k].strip()
               for k in ("claude", "codex")):
        raise SystemExit("runtime binding lacks both signatures")
    if bind.get("prereg_sha256") != prereg_sha:
        raise SystemExit("runtime binding names another prereg")
    reviews = bind.get("reviews")
    if not isinstance(reviews, list) or not reviews:
        raise SystemExit("runtime binding cites no review records")
    for r in reviews:
        if not is_hex64(r.get("sha256")) or sha256_file(r["path"]) != r["sha256"]:
            raise SystemExit(f"review record changed or missing: {r.get('path')}")
    man_path = Path(bind["manifest_path"]).resolve()
    if not is_hex64(bind.get("manifest_sha256")) or sha256_file(man_path) != bind["manifest_sha256"]:
        raise SystemExit("runtime manifest differs from binding")
    man = json.loads(man_path.read_text())
    if man.get("format") != MANIFEST_FORMAT or man.get("scope") != scope or man.get("prereg_sha256") != prereg_sha:
        raise SystemExit("runtime manifest format/scope/prereg mismatch")
    arts = man.get("artifacts") or []
    paths = [a["path"] for a in arts]
    if not arts or len(set(paths)) != len(paths):
        raise SystemExit("runtime manifest artifacts empty or duplicated")
    cfg_real, ckpt_real = str(Path(config).resolve()), str(Path(ckpt_dir).resolve())
    if man.get("deploy_config") != cfg_real or cfg_real not in paths:
        raise SystemExit(f"selected deploy config {cfg_real} is not the bound one")
    if man.get("checkpoint_dir") != ckpt_real:
        raise SystemExit(f"selected checkpoint {ckpt_real} is not the bound one")
    for a in arts:
        if not is_hex64(a.get("sha256")) or sha256_file(a["path"]) != a["sha256"]:
            raise SystemExit(f"runtime artifact changed: {a['path']}")
    if cfg_sha(resolve_config(config, ckpt_dir)) != man.get("resolved_config_sha256"):
        raise SystemExit("resolved deploy config differs from the bound snapshot")
    print(f"[binding] {scope}: {len(arts)} artifacts, resolved config and {len(reviews)} reviews verified", flush=True)
    return {"runtime_binding": {"path": str(bind_path), "sha256": sha256_file(bind_path)},
            "runtime_manifest": {"path": str(man_path), "sha256": bind["manifest_sha256"], "commit": man["runtime_commit"]},
            "manifest": man}


HOLDOUT_MANIFEST_FORMAT = "openwam-g2-holdout-manifest-receipt/1"
HOLDOUT_SELECTION_FORMAT = "openwam-g2-holdout-selection/1"


def _frozen_jointly(path, fmt: str, prereg_sha: str) -> dict:
    r = json.loads(Path(path).read_text())
    if r.get("format") != fmt or r.get("status") != "frozen" or r.get("prereg_sha256") != prereg_sha:
        raise SystemExit(f"{Path(path).name}: not a frozen {fmt} for this prereg")
    approvals = r.get("approvals") or []
    if {a.get("by") for a in approvals} != {"codex", "claude"}:
        raise SystemExit(f"{Path(path).name}: joint approvals required")
    for a in approvals:
        if not is_hex64(a.get("sha256")) or sha256_file(a["path"]) != a["sha256"]:
            raise SystemExit(f"{Path(path).name}: approval artifact changed: {a.get('path')}")
    return r


def check_holdout_gate(prereg_path, freeze_receipt, explore_freeze, holdout_manifest) -> dict:
    """Before any holdout work: G2 freeze, exploration freeze and the jointly frozen holdout manifest,
    whose selection must name this prereg and these two receipts. Returns observations and references."""
    prereg_sha = sha256_file(prereg_path)
    _frozen_jointly(freeze_receipt, "openwam-g2-freeze-receipt/1", prereg_sha)
    ex = _frozen_jointly(explore_freeze, "openwam-g2-explore-freeze-receipt/1", prereg_sha)
    roles = {a.get("role") for a in ex.get("artifacts", [])}
    if not {"analysis_script", "interpretation"} <= roles:
        raise SystemExit("exploration freeze lacks analysis/interpretation artifacts")
    for a in ex["artifacts"]:
        if sha256_file(a["path"]) != a["sha256"]:
            raise SystemExit(f"exploration freeze artifact changed: {a['path']}")
    hm = _frozen_jointly(holdout_manifest, HOLDOUT_MANIFEST_FORMAT, prereg_sha)
    sel_ref = hm.get("selection") or {}
    if not is_hex64(sel_ref.get("sha256")) or sha256_file(sel_ref["path"]) != sel_ref["sha256"]:
        raise SystemExit("holdout selection differs from the frozen manifest")
    sel = json.loads(Path(sel_ref["path"]).read_text())
    if (sel.get("format") != HOLDOUT_SELECTION_FORMAT or sel.get("prereg_sha256") != prereg_sha
            or sel.get("freeze_sha256") != sha256_file(freeze_receipt)
            or sel.get("explore_freeze_sha256") != sha256_file(explore_freeze)):
        raise SystemExit("holdout selection does not name this prereg and these freeze receipts")
    obs = sel.get("observations") or []
    ids = [o["obs_id"] for o in obs]
    if not obs or len(set(ids)) != len(ids) or any(o.get("split") != "holdout" for o in obs):
        raise SystemExit("holdout selection observations empty, duplicated or not split=holdout")
    expected = hm.get("n_observations")
    if not isinstance(expected, int) or expected != len(obs):
        raise SystemExit("holdout manifest observation count does not match the selection")
    print(f"[holdout] gate verified: {len(obs)} observations, selection {sel_ref['sha256'][:12]}", flush=True)
    return {"observations": obs,
            "refs": {"explore_freeze": {"path": str(Path(explore_freeze).resolve()), "sha256": sha256_file(explore_freeze)},
                     "holdout_manifest": {"path": str(Path(holdout_manifest).resolve()), "sha256": sha256_file(holdout_manifest)},
                     "holdout_selection": {"path": sel_ref["path"], "sha256": sel_ref["sha256"]}}}


def holdout_observations(args, mode: str, prereg_path) -> tuple[list, dict]:
    """Observations for --split holdout: gated in run mode; smoke reads an unfrozen selection file."""
    if mode == "run":
        if not (args.explore_freeze and args.holdout_manifest):
            raise SystemExit("holdout run needs --explore-freeze and --holdout-manifest")
        g = check_holdout_gate(prereg_path, args.freeze_receipt, args.explore_freeze, args.holdout_manifest)
        return g["observations"], g["refs"]
    if not args.selection:
        raise SystemExit("holdout smoke needs --selection")
    sel = json.loads(Path(args.selection).read_text())
    return sel["observations"][: args.n_obs], {"smoke_selection": {"path": str(Path(args.selection).resolve()),
                                                                     "sha256": sha256_file(args.selection)}}


# -- runtime description --------------------------------------------------------------------------
def engine_schedule(engine) -> list:
    """The schedule ``DeployEngine.generate`` builds for a request without overrides."""
    from openwam.deploy.denoise_schedule import make_schedule

    inf, arch = engine.cfg.inference, engine.architecture
    ab, vb = getattr(arch, "action_backbone", None), getattr(arch, "video_backbone", None)
    shift = getattr(ab, "shift_action", None) if ab is not None else None
    sched = make_schedule(getattr(inf, "denoise_mode", "sync"), video_scheduler=arch.video_scheduler,
                          action_scheduler=arch.action_scheduler, num_steps=inf.denoise_steps,
                          shift=5.0 if shift is None else shift,
                          shift_video=getattr(vb, "shift_video", None) if vb is not None else None,
                          lead=getattr(inf, "lead_modality", "video"), alpha=getattr(inf, "variance_shift_alpha", 1.0),
                          offset=getattr(inf, "linear_offset", 0.0))
    return [[float(tv), float(ta)] for tv, ta in sched]


def loaded_repo_files() -> list:
    out = set()
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        # some extension modules carry a bare relative __file__ (torch._classes -> "_classes.py")
        if f and os.path.isabs(f) and str(Path(f).resolve()).startswith(str(REPO) + os.sep) and f.endswith(".py"):
            out.add(os.path.relpath(Path(f).resolve(), REPO))
    return sorted(out)


def describe_runtime(runner) -> dict:
    import PIL
    import omegaconf

    eng, arch = runner.engine, runner.arch
    kw = getattr(arch, "_mot_driver_kwargs", None) or {}
    inactive = arch._resolve_inactive_action_dims(None, arch.device) if hasattr(arch, "_resolve_inactive_action_dims") else None
    from openwam.deploy.engine import _cfg_sha256

    # the engine's own hash: it is what every probe trace records as deploy_config_sha256
    return {"engine_config_sha256": _cfg_sha256(eng.cfg), "architecture": type(arch).__name__, "dtype": str(arch.dtype),
            "action_dim": int(getattr(arch, "action_dim", -1)),
            "attention": {k: str(kw[k]) for k in sorted(kw) if "mask" in k},
            "schedule": engine_schedule(eng),
            "inactive_action_dims": None if inactive is None else torch.as_tensor(inactive).cpu().int().tolist(),
            "dit_cache": eng._dit_cache is not None,
            "versions": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                         "cudnn": torch.backends.cudnn.version(), "numpy": np.__version__, "pillow": PIL.__version__,
                         "omegaconf": omegaconf.__version__},
            "loaded_repo_files": loaded_repo_files()}


def check_runtime(runner, manifest: dict) -> dict:
    """After model load, before any generation: the live runtime must equal the bound description."""
    live = describe_runtime(runner)
    want = manifest["runtime_description"]
    diff = sorted(k for k in set(live) | set(want) if k != "loaded_repo_files" and live.get(k) != want.get(k))
    if diff:
        raise SystemExit(f"live runtime differs from the bound description: {diff}")
    shas = {os.path.relpath(a["path"], REPO): a["sha256"] for a in manifest["artifacts"]}
    missing = [f for f in live["loaded_repo_files"] if f not in shas]
    if missing:
        raise SystemExit(f"imported repo files not bound by the manifest: {missing[:8]}")
    print(f"[runtime] live description matches; {len(live['loaded_repo_files'])} imported repo files bound", flush=True)
    return live


def check_imports_bound(manifest: dict) -> None:
    """At the end of a formal run: every repo file imported at any point must be bound."""
    shas = {os.path.relpath(a["path"], REPO) for a in manifest["artifacts"]}
    missing = [f for f in loaded_repo_files() if f not in shas]
    if missing:
        raise SystemExit(f"repo files imported during the run are not bound by the manifest: {missing[:8]}")


def write_description(runner, out) -> None:
    out = Path(out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    desc = describe_runtime(runner)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(desc, indent=1, sort_keys=True))
    os.replace(tmp, out)
    print(json.dumps({"description": str(out), "sha256": sha256_file(out)}), flush=True)


# -- output ledger --------------------------------------------------------------------------------
class Ledger:
    def __init__(self, out, stamp: dict):
        self.out = Path(out).resolve()
        if not str(self.out).startswith(str(PROBE_ROOT) + os.sep):
            raise SystemExit(f"--out must be below {PROBE_ROOT}")
        self.stamp_path, self.index, self.failures = (self.out / n for n in ("RUN_STAMP.json", "index.jsonl", "failures.jsonl"))
        existing = [p for p in self.out.rglob("*") if p.is_file()] if self.out.exists() else []
        if existing and not self.stamp_path.exists():
            raise SystemExit("output dir has files but no RUN_STAMP; refusing to mix runs")
        if self.stamp_path.exists() and json.loads(self.stamp_path.read_text()) != stamp:
            raise SystemExit("output dir belongs to a different prereg/mode/binding; refusing to mix runs")
        self.rows = {}
        if self.index.exists():
            for line in self.index.read_text().splitlines():
                row = json.loads(line)
                if row["key"] in self.rows:
                    raise SystemExit(f"duplicate index key {row['key']}")
                p = Path(row["path"])
                if p.parent != self.out and self.out not in p.parents:
                    raise SystemExit(f"index row outside the output dir: {p}")
                if not p.exists() or sha256_file(p) != row["sha256"]:
                    raise SystemExit(f"indexed output missing or changed: {p}")
                self.rows[row["key"]] = row
        indexed = {str(Path(r["path"])) for r in self.rows.values()}
        orphans = [str(p) for p in existing if p.name not in LEDGER_FILES and str(p) not in indexed]
        if orphans:
            raise SystemExit(f"unindexed files in output dir (failed or foreign units): {orphans[:5]}")
        self.out.mkdir(parents=True, exist_ok=True)
        if not self.stamp_path.exists():
            self.stamp_path.write_text(json.dumps(stamp, sort_keys=True))

    def get(self, key: str):
        return self.rows.get(key)

    def load(self, key: str) -> dict:
        row = self.rows[key]
        if sha256_file(row["path"]) != row["sha256"]:
            raise SystemExit(f"output changed after indexing: {row['path']}")
        return torch.load(row["path"], map_location="cpu", weights_only=False)

    def commit(self, key: str, path: Path, obj: dict, **fields) -> dict:
        if key in self.rows:
            raise SystemExit(f"unit already committed: {key}")
        sha = atomic_torch_save(obj, path)
        row = {"key": key, "path": str(path), "sha256": sha, **fields}
        append_jsonl(self.index, row)
        self.rows[key] = row
        return row

    def adopt(self, key: str, path: Path, **fields) -> dict:
        """Index a file another component wrote (the engine's source trace) after the caller checked it."""
        if key in self.rows:
            raise SystemExit(f"unit already committed: {key}")
        row = {"key": key, "path": str(path), "sha256": sha256_file(path), **fields}
        append_jsonl(self.index, row)
        self.rows[key] = row
        return row

    def fail(self, key: str, error: str) -> None:
        append_jsonl(self.failures, {"key": key, "error": error})

    def note(self, row: dict) -> None:
        """Journal of passed engineering checks that are not output units (e.g. holdout source checks)."""
        append_jsonl(self.out / "source_checks.jsonl", row)

    def require_exact(self, expected_keys) -> None:
        got, want = set(self.rows), set(expected_keys)
        if got != want:
            raise SystemExit(f"output keys differ from the expected set: missing {sorted(want - got)[:5]}, "
                             f"extra {sorted(got - want)[:5]}")


def check_meta(tr: dict, expect: dict, where: str) -> None:
    meta = tr.get("meta") or {}
    bad = sorted(k for k, v in expect.items() if meta.get(k) != v)
    if bad:
        raise SystemExit(f"stored unit does not match its expected identity ({where}): {bad}")
