"""Write a runtime manifest (format 2) for a G2 wrapper. Read-only apart from the two new output files.

Artifacts: every checkpoint file (download caches and partial ``.range_parts`` excluded), the deploy
config, all tracked python under ``openwam/`` and ``benchmarks/utils`` (+ package init), the named probe scripts, and every
repo file the model load actually imported (from the wrapper's ``describe`` output). Also written: a
snapshot of the deploy config as the server resolves it (``<out>.resolved_config.yaml``) whose sha is
bound as ``resolved_config_sha256``, and the runtime description itself.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from g2_common import MANIFEST_FORMAT, cfg_text, resolve_config, sha256_file  # noqa: E402

SKIP_DIRS = {".cache", ".range_parts"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--deploy-config", default=str(REPO / "configs/deploy_probe.yaml"))
    ap.add_argument("--scripts", nargs="+", required=True)
    ap.add_argument("--description", required=True, help="JSON written by '<wrapper> describe'")
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--scope", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    git = lambda *c: subprocess.check_output(["git", "-C", str(REPO), *c], text=True).strip()
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("tracked files are dirty")
    out = Path(a.out)
    snap = out.with_name(out.stem + ".resolved_config.yaml")
    if out.exists() or snap.exists():
        raise SystemExit("refusing to overwrite " + str(out))
    desc = json.loads(Path(a.description).read_text())
    ckpt, config = Path(a.ckpt_dir).resolve(), Path(a.deploy_config).resolve()
    files = {p for p in ckpt.rglob("*") if p.is_file() and not (SKIP_DIRS & set(p.relative_to(ckpt).parts))}
    files.add(config)
    tracked = git("ls-files", "openwam", "benchmarks/__init__.py", "benchmarks/utils").splitlines()
    files |= {REPO / f for f in tracked if f.endswith(".py")}  # benchmarks.utils.client builds the payload
    files |= {(REPO / "scripts/probe" / s).resolve() for s in a.scripts}
    files |= {REPO / f for f in desc["loaded_repo_files"]}
    for f in files:
        rel = os.path.relpath(f, REPO)
        if not rel.startswith("assets") and git("ls-files", rel) != rel:
            raise SystemExit(f"untracked runtime file: {rel}")
    text = cfg_text(resolve_config(str(config), str(ckpt)))
    snap.write_text(text)
    man = {"format": MANIFEST_FORMAT, "scope": a.scope, "prereg_sha256": sha256_file(a.prereg),
           "runtime_commit": git("rev-parse", "HEAD"), "checkpoint_dir": str(ckpt), "deploy_config": str(config),
           "resolved_config_snapshot": str(snap.resolve()),
           "resolved_config_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
           "runtime_description": desc, "runtime_description_sha256": sha256_file(a.description),
           "excluded_checkpoint_dirs": sorted(SKIP_DIRS),
           "artifacts": [{"path": str(f), "bytes": f.stat().st_size, "sha256": sha256_file(f)} for f in sorted(files)]}
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(man, indent=1))
    os.replace(tmp, out)
    print(json.dumps({"out": str(out), "sha256": sha256_file(out), "n_artifacts": len(files),
                      "commit": man["runtime_commit"], "resolved_config_sha256": man["resolved_config_sha256"]}))


if __name__ == "__main__":
    main()
