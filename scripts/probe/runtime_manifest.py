"""Write a runtime manifest: sha256 of every runtime-relevant file (checkpoint dir, deploy config, all
tracked python under openwam/, and the named probe scripts) plus the repo commit. Read-only."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--deploy-config", default=str(REPO / "configs/deploy_probe.yaml"))
    ap.add_argument("--scripts", nargs="+", required=True)
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--scope", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    git = lambda *c: subprocess.check_output(["git", "-C", str(REPO), *c], text=True).strip()
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("tracked files are dirty")
    ckpt = Path(a.ckpt_dir).resolve()
    files = sorted(p for p in ckpt.rglob("*") if p.is_file() and ".cache" not in p.parts)
    files += [Path(a.deploy_config).resolve()]
    files += sorted(REPO / f for f in git("ls-files", "openwam").splitlines() if f.endswith(".py"))
    files += [(REPO / "scripts/probe" / s).resolve() for s in a.scripts]
    for f in files:
        rel = os.path.relpath(f, REPO)
        if not rel.startswith("assets") and git("ls-files", rel) != rel:
            raise SystemExit(f"untracked runtime file: {rel}")
    man = {"format": "openwam-runtime-manifest/1", "scope": a.scope, "prereg_sha256": sha(a.prereg),
           "runtime_commit": git("rev-parse", "HEAD"), "checkpoint_dir": str(ckpt),
           "artifacts": [{"path": str(f), "bytes": f.stat().st_size, "sha256": sha(f)} for f in files]}
    out = Path(a.out)
    if out.exists():
        raise SystemExit("refusing to overwrite " + str(out))
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(man, indent=1))
    os.replace(tmp, out)
    print(json.dumps({"out": str(out), "sha256": sha(out), "n_artifacts": len(files), "commit": man["runtime_commit"]}))


if __name__ == "__main__":
    main()
