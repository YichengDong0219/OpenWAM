"""CPU tests for P1B: window replay construction / checks (p1b_claude) and the patched
BaseWAMArchitecture._check_probe_replay (replay_window opt-in). No model, no GPU. Usage: test_p1b_claude.py"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import p1b_claude as B  # noqa: E402
from openwam.model.architectures.base import BaseWAMArchitecture  # noqa: E402

checks = {}
SHAPE = (1, 48, 3, 24, 20)
SCHED = [(1000.0, 1000.0), (978.2608642578125,) * 2, (952.3810424804688,) * 2, (921.0526123046875,) * 2,
         (882.3529052734375,) * 2, (833.3333129882812,) * 2, (769.2307739257812,) * 2, (681.8181762695312,) * 2,
         (555.5555419921875,) * 2, (357.1428527832031,) * 2, (0.0, 0.0)]
g = torch.Generator().manual_seed(0)
eps = torch.randn(SHAPE, generator=g).to(torch.bfloat16)
first = eps[:, :, :1].clone()
steps = []
for i, (tv, ta) in enumerate(SCHED[:-1]):
    z = eps.clone() if i == 0 else torch.randn(SHAPE, generator=g).to(torch.bfloat16)
    z[:, :, :1] = first
    steps.append({"i": i, "t_v": tv, "t_a": ta, "sigma_v": tv / 1000, "sigma_a": ta / 1000, "video_latents_in": z,
                  "action_latents_in": torch.randn((1, 32, 80), generator=g).to(torch.bfloat16)})
c0 = {"meta": {"schedule": SCHED, "num_frames": 9, "action_num_frames": 33, "active_action_mask": torch.ones(80, dtype=torch.bool)},
      "init": {"video_noise": eps}, "steps": steps, "final": {"video_latents": eps.clone()}}
target = torch.randn(SHAPE, generator=g).to(torch.bfloat16)


def refuses(fn):
    try:
        fn(); return False
    except (ValueError, SystemExit):
        return True


# windows constants
checks["windows: E=0..3, M=4..6, L=7..9, A=0..9 partition / cover the 10 executed steps"] = \
    B.WINDOWS["E"] + B.WINDOWS["M"] + B.WINDOWS["L"] == list(range(10)) and B.WINDOWS["A"] == list(range(10))

for w, idx in B.WINDOWS.items():
    pr = B.window_replay(c0, target, idx, 1000.0)
    checks[f"window {w}: replay steps exactly the window"] = sorted(pr["replay_video_steps"]) == idx and pr["replay_window"] == idx \
        and sorted(pr["replay_t_v"]) == idx
    checks[f"window {w}: each step = (1-sigma)*target + sigma*eps on future frames, frame 0 = receiver"] = all(
        torch.equal(pr["replay_video_steps"][i][:, :, 1:],
                    ((1 - SCHED[i][0] / 1000) * target[:, :, 1:].float() + (SCHED[i][0] / 1000) * eps[:, :, 1:].float()).to(torch.bfloat16))
        and torch.equal(pr["replay_video_steps"][i][:, :, :1], first) for i in idx)
pa = B.window_replay(c0, target, B.WINDOWS["A"], 1000.0)
from p1a_claude import clamp_replay  # noqa: E402
old = clamp_replay(c0, target, 1000.0)
checks["window A equals the P1A all-step clamp construction bit for bit"] = all(
    torch.equal(pa["replay_video_steps"][s["i"]], s["video_latents_in"]) for s in old["steps"])
pe = B.window_replay(c0, target, B.WINDOWS["E"], 1000.0)
checks["window E step 0 (sigma=1): target coefficient 0 -> equals receiver step-0 input"] = torch.equal(pe["replay_video_steps"][0], steps[0]["video_latents_in"])
checks["no extra forward: replay only names executed step indices (no new timestep)"] = set(pa["replay_t_v"].values()) <= {t for t, _ in SCHED[:-1]}
for bad, name in (([], "empty"), ([3, 1], "unsorted"), ([1, 1, 2], "duplicate"), ([9, 10], "index outside executed steps"), ([-1, 0], "negative")):
    checks[f"window_replay refuses {name}"] = refuses(lambda: B.window_replay(c0, target, bad, 1000.0))

# window checks on synthetic recorded traces
def recorded(probe, window, tamper=None):
    st = []
    for s in steps:
        i = s["i"]
        z = probe["replay_video_steps"][i].clone() if i in window else (s["video_latents_in"].clone() if i < min(window)
                                                                       else torch.randn(SHAPE, generator=g).to(torch.bfloat16))
        z[:, :, :1] = first
        st.append({**{k: s[k] for k in ("i", "t_v", "t_a", "sigma_v", "sigma_a")}, "video_latents_in": z,
                   "action_latents_in": s["action_latents_in"].clone() if i < min(window) else torch.randn((1, 32, 80), generator=g).to(torch.bfloat16)})
    tr = {"steps": st, "replay_window": list(window)}
    if tamper:
        tamper(tr)
    return tr


for w in ("E", "M", "L", "A"):
    idx = B.WINDOWS[w]; pr = B.window_replay(c0, target, idx, 1000.0)
    checks[f"window_checks pass on a faithful window-{w} trace"] = B.window_checks(c0, recorded(pr, idx), pr, idx)["all_passed"]
pm = B.window_replay(c0, target, B.WINDOWS["M"], 1000.0)
checks["window_checks: in-window input not clamped -> fail"] = not B.window_checks(c0, recorded(pm, B.WINDOWS["M"], lambda t: t["steps"][5].__setitem__("video_latents_in", steps[5]["video_latents_in"])), pm, B.WINDOWS["M"])["all_passed"]
checks["window_checks: pre-window input differs from C0 -> fail"] = not B.window_checks(c0, recorded(pm, B.WINDOWS["M"], lambda t: t["steps"][2].__setitem__("video_latents_in", target.clone())), pm, B.WINDOWS["M"])["all_passed"]
checks["window_checks: recorded window differs -> fail"] = not B.window_checks(c0, recorded(pm, B.WINDOWS["M"], lambda t: t.__setitem__("replay_window", [4, 5])), pm, B.WINDOWS["M"])["all_passed"]
checks["window_checks: frame 0 changed -> fail"] = not B.window_checks(c0, recorded(pm, B.WINDOWS["M"], lambda t: t["steps"][8]["video_latents_in"].__setitem__((slice(None), slice(None), slice(0, 1)), target[:, :, :1])), pm, B.WINDOWS["M"])["all_passed"]

# patched base: _check_probe_replay with replay_window
live = torch.zeros(SHAPE, dtype=torch.bfloat16)
def probe_for(window, final=None, steps_override=None):
    st = steps_override if steps_override is not None else {i: live.clone() for i in window}
    return {"replay_video_steps": st, "replay_t_v": {i: SCHED[i][0] for i in st}, "replay_schedule": SCHED,
            "replay_window": window, "replay_video_final": final}
ok = lambda pr: not refuses(lambda: BaseWAMArchitecture._check_probe_replay(SCHED, pr, live))
checks["base: windowed replay with a sorted subset accepted (E, M, L, A)"] = all(ok(probe_for(B.WINDOWS[w])) for w in "EMLA")
checks["base: window with steps outside it refused"] = not ok(probe_for([4, 5], steps_override={4: live, 5: live, 6: live}))
checks["base: unsorted window refused"] = not ok(probe_for([5, 4]))
checks["base: empty window refused"] = not ok(probe_for([]))
checks["base: window index 10 (not executed) refused"] = not ok(probe_for([9, 10]))
checks["base: replay_window with replay_video_final refused"] = not ok(probe_for([4, 5], final=live))
full = {"replay_video_steps": {i: live for i in range(10)}, "replay_t_v": {i: SCHED[i][0] for i in range(10)}, "replay_schedule": SCHED}
checks["base: no replay_window -> original all-step rule (accepted with all steps)"] = ok(full)
checks["base: no replay_window and a missing step -> refused (original behaviour)"] = not ok({**full, "replay_video_steps": {i: live for i in range(9)}})
bad_dtype = probe_for([4], steps_override={4: live.float()})
checks["base: windowed step with wrong dtype refused"] = not ok(bad_dtype)

# R81 counterexamples
def dup6(t):
    t["steps"][6] = dict(t["steps"][5])
checks["window_checks: step 6 replaced by a duplicate of step 5 (index 5 twice, 6 missing) -> fail"] = not B.window_checks(c0, recorded(pm, B.WINDOWS["M"], dup6), pm, B.WINDOWS["M"])["all_passed"]
checks["window_checks: a step index relabelled -> fail"] = not B.window_checks(c0, recorded(pm, B.WINDOWS["M"], lambda t: t["steps"][7].__setitem__("i", 8)), pm, B.WINDOWS["M"])["all_passed"]
checks["window_checks: a step t_a differs from C0/schedule -> fail"] = not B.window_checks(c0, recorded(pm, B.WINDOWS["M"], lambda t: t["steps"][2].__setitem__("t_a", 1.0)), pm, B.WINDOWS["M"])["all_passed"]
checks["window_checks: a step missing -> fail"] = not B.window_checks(c0, recorded(pm, B.WINDOWS["M"], lambda t: t["steps"].pop()), pm, B.WINDOWS["M"])["all_passed"]
checks["window_checks: pre-window ACTION state differs from C0 -> fail"] = not B.window_checks(c0, recorded(pm, B.WINDOWS["M"], lambda t: t["steps"][1].__setitem__("action_latents_in", torch.zeros((1, 32, 80), dtype=torch.bfloat16))), pm, B.WINDOWS["M"])["all_passed"]
checks["base: float window index 4.9 refused (no silent truncation)"] = not ok(probe_for([4.9], steps_override={4: live}))
checks["base: bool window index refused"] = not ok(probe_for([True], steps_override={1: live}))

# seal-anchored P1A reference (synthetic sealed shards)
import hashlib, shutil, tempfile
import numpy as np


def make_seal(units_per_shard):
    root = Path(tempfile.mkdtemp()); shards = []
    for k, units in enumerate(units_per_shard):
        d = root / f"s{k}"; (d / "units").mkdir(parents=True); rows = []
        for (o, c, v, a) in units:
            p = d / "units" / f"{o}_{c}_{v}_{a}.npz"
            with open(p, "wb") as f:
                np.savez(f, actions_physical=np.full((32, 20), float(v + a)), actions_normalized=np.zeros((32, 80)))
            rows.append({"ok": True, "unit": f"{o}|{c}|{v}|{a}", "obs": o, "cond": c, "video_seed": v, "action_seed": a,
                         "artifacts": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}]})
        (d / "journal.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        dig, n = B.dir_file_map_digest(d)
        shards.append({"k": k, "output_dir": str(d), "directory_file_map_sha256": dig, "files": n})
    seal = root / "seal.json"; seal.write_text(json.dumps({"format": "p1a-production-seal/1", "shards": shards}))
    return {"path": str(seal), "sha256": hashlib.sha256(seal.read_bytes()).hexdigest()}, shards


bind = {"observations": ["o1"], "video_seeds": [1001], "action_seeds": [2001, 2002],
        "scatter": {"video_seed": 1001, "action_seeds": [2003]}}
need = B.needed_reference_units(bind)
units = sorted(need)
seal_ref, shards = make_seal([units[:5], units[5:9], units[9:], []])
checks["p1a_reference: accepts an intact seal with every needed unit"] = not refuses(lambda: B.p1a_reference(seal_ref, need))
checks["p1a_reference: empty / missing seal reference refused (never skipped)"] = refuses(lambda: B.p1a_reference({}, need)) and refuses(lambda: B.p1a_reference(None, need))


def tampered(fn):
    sr, sh = make_seal([units[:5], units[5:9], units[9:], []]); fn(sh); return refuses(lambda: B.p1a_reference(sr, need))


def rewrite_npz_and_journal(sh):   # rewrite an npz AND its journal hash consistently: the sealed dir digest catches it
    d = Path(sh[0]["output_dir"]); rows = [json.loads(x) for x in (d / "journal.jsonl").read_text().splitlines()]
    p = Path(rows[0]["artifacts"][0]["path"])
    with open(p, "wb") as f:
        np.savez(f, actions_physical=np.zeros((32, 20)), actions_normalized=np.zeros((32, 80)))
    rows[0]["artifacts"][0]["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    (d / "journal.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


checks["p1a_reference: npz and journal hash rewritten together -> refused (sealed directory digest)"] = tampered(rewrite_npz_and_journal)
sr2, sh2 = make_seal([units[:5] + units[:1], units[5:9], units[9:], []])
checks["p1a_reference: duplicate reference unit -> refused"] = refuses(lambda: B.p1a_reference(sr2, need))
sr3, sh3 = make_seal([units[:5], units[5:9], units[9:-1], []])
checks["p1a_reference: a needed unit missing -> refused"] = refuses(lambda: B.p1a_reference(sr3, need))
sr4 = dict(seal_ref, sha256="0" * 64)
checks["p1a_reference: seal file SHA mismatch -> refused"] = refuses(lambda: B.p1a_reference(sr4, need))

failed = [k for k, v in checks.items() if not v]
print("RESULT " + json.dumps({"n": len(checks), "failed": failed, "checks": checks}))
sys.exit(1 if failed else 0)
