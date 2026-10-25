"""Host-side runtime dependency manifest (CPU only). Fails (rc 1) on a dirty tracked tree or any failed probe.

Records: worktree HEAD + digest of `git ls-files -s` (+ untracked-ignored symlink targets), SHA-256 of EVERY file under
the alpha checkpoint dir (weights, config, normalizer stats) and under assets/video_backbone_ckpt (Wan/T5/VAE/
tokenizer), the RoboTwin checkout (HEAD, status, ls-files digest, hanging_mug task source, mug/rack asset files),
both Python environments (`pip freeze`), torch/CUDA versions and the driver. Usage: backend_manifest.py OUT.json
"""
import hashlib, json, os, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
out = {"format": "recovery-backend-manifest/1", "errors": []}


def run(cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        out["errors"].append({"cmd": cmd[:4], "rc": r.returncode, "stderr": r.stderr[-300:]})
    return r.stdout


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def tree(d):
    d = Path(d).resolve()
    return {str(p.relative_to(d)): sha(p) for p in sorted(d.rglob("*")) if p.is_file()}


out["openwam"] = {"head": run(["git", "rev-parse", "HEAD"], ROOT).strip(),
                  "dirty": run(["git", "status", "--porcelain", "--untracked-files=no"], ROOT).strip(),
                  "tracked_digest": hashlib.sha256(run(["git", "ls-files", "-s"], ROOT).encode()).hexdigest(),
                  "assets_link": os.path.realpath(ROOT / "assets")}
for key, rel in (("checkpoint", "assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboTwin-Full"),
                 ("video_backbone_ckpt", "assets/video_backbone_ckpt")):
    out[key] = tree(ROOT / rel)
    out[key + "_realpath"] = os.path.realpath(ROOT / rel)
    if not out[key]:
        out["errors"].append({"cmd": [key], "rc": "empty", "stderr": f"no files under {rel}"})
rt = Path(os.environ["ROBOTWIN_PATH"])
out["robotwin"] = {"path": str(rt), "head": run(["git", "rev-parse", "HEAD"], rt).strip(),
                   "dirty": run(["git", "status", "--porcelain", "--untracked-files=no"], rt).strip(),
                   "tracked_digest": hashlib.sha256(run(["git", "ls-files", "-s"], rt).encode()).hexdigest(),
                   "hanging_mug_py": sha(rt / "envs/hanging_mug.py"),
                   "assets": {k: tree(v) for k, v in {a.name: a for a in sorted((rt / "assets/objects").glob("*"))
                                                       if a.name.split("_", 1)[-1] in ("cup_with_handle", "rack", "mug")
                                                       or "rack" in a.name or "mug" in a.name}.items()}}
for name in ("OPENWAM_PYTHON", "ROBOTWIN_PYTHON"):
    py = os.environ[name]
    out[name] = {"path": py, "freeze_sha256": hashlib.sha256(run([py, "-m", "pip", "freeze"]).encode()).hexdigest(),
                 "torch": run([py, "-c", "import torch;print(torch.__version__, torch.version.cuda)"]).strip()}
out["driver"] = run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader"]).strip()
ok = not out["errors"] and not out["openwam"]["dirty"] and not out["robotwin"]["dirty"]
out["ok"] = ok
Path(sys.argv[1]).write_text(json.dumps(out, indent=1, sort_keys=True))
print(json.dumps({"ok": ok, "errors": out["errors"][:3], "openwam_head": out["openwam"]["head"]}))
raise SystemExit(0 if ok else 1)
