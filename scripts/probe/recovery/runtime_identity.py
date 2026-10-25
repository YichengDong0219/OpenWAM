"""Host-side runtime identity (CPU only). Prints one JSON object; exit 1 on any failed probe.

--code-only : SHA-256 over the ON-DISK bytes of every git-tracked file of the OpenWAM worktree and the RoboTwin
              checkout (not the git index: assume-unchanged / skip-worktree edits are caught), plus HEADs.
(default)   : the above plus stat identities (relpath, size, mtime_ns, inode, device) of every file under the large
              asset roots (alpha checkpoint, Wan/T5/VAE dir, RoboTwin assets) and the two Python environments'
              site-packages + interpreter realpaths. Content SHAs of the model assets are bound separately in the
              backend manifest; a changed identity means the asset or environment was touched after that.
Usage: runtime_identity.py [--code-only]
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
errors: list[str] = []


def git(repo: Path, *args) -> str:
    try:
        r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        errors.append(f"git {args[0]} timeout in {repo}"); return ""
    if r.returncode != 0:
        errors.append(f"git {args[0]} rc={r.returncode} in {repo}")
    return r.stdout


def code_digest(repo: Path) -> dict:
    files = [f for f in git(repo, "ls-files", "-z").split("\0") if f]
    if not files:
        errors.append(f"no tracked files in {repo}")
    h = hashlib.sha256()
    for f in sorted(files):
        p = repo / f
        if p.is_symlink():
            data = ("symlink:" + os.readlink(p)).encode()
        elif p.is_file():
            data = p.read_bytes()
        else:
            data = b"<missing>"
        h.update(f.encode() + b"\0" + hashlib.sha256(data).hexdigest().encode() + b"\n")
    return {"head": git(repo, "rev-parse", "HEAD").strip(), "n_files": len(files), "bytes_digest": h.hexdigest()}


def stat_digest(root: Path) -> dict:
    root = root.resolve()
    if not root.is_dir():
        errors.append(f"missing dir {root}"); return {"root": str(root), "n_files": 0, "digest": None}
    h, n = hashlib.sha256(), 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")  # bytecode caches are rewritten by imports
        for fn in sorted(f for f in filenames if not f.endswith(".pyc")):
            p = Path(dirpath) / fn
            try:
                st = p.stat()
            except OSError:
                continue
            rel = str(p.relative_to(root))
            h.update(f"{rel}\0{st.st_size}\0{st.st_mtime_ns}\0{st.st_ino}\0{st.st_dev}\n".encode())
            n += 1
    if n == 0:
        errors.append(f"empty dir {root}")
    return {"root": str(root), "n_files": n, "digest": h.hexdigest()}


def main() -> int:
    rt = Path(os.environ["ROBOTWIN_PATH"])
    out = {"format": "recovery-runtime-identity/1", "openwam_code": code_digest(ROOT), "robotwin_code": code_digest(rt)}
    if "--code-only" not in sys.argv:
        out["assets"] = {k: stat_digest(ROOT / v) for k, v in {
            "checkpoint": "assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboTwin-Full",
            "video_backbone_ckpt": "assets/video_backbone_ckpt"}.items()}
        out["assets"]["robotwin_assets"] = stat_digest(rt / "assets")
        for name in ("OPENWAM_PYTHON", "ROBOTWIN_PYTHON"):
            py = Path(os.environ[name])
            prefix = py.resolve().parent.parent
            sp = sorted(prefix.glob("lib/python3*/site-packages"))
            out[name] = {"interpreter": str(py.resolve()), **(stat_digest(sp[0]) if sp else {"digest": None})}
            if not sp:
                errors.append(f"no site-packages under {prefix}")
    out["errors"] = errors
    out["ok"] = not errors
    print(json.dumps(out, sort_keys=True))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
