"""CPU tests for p1a_claude (R76 fixes). No model, no GPU: the driver runs end to end on a MOCK runner that returns
NATIVE probe traces (top-level seeds/schedule/mask/cond/init/steps/final, no meta), exactly the structure of
base.generate. Usage: test_p1a_claude.py"""
import copy
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p1a_claude as P  # noqa: E402

checks = {}
SHAPE = (1, 48, 3, 24, 20)
SCHED = [(1000.0, 1000.0), (978.2608642578125,) * 2, (952.3810424804688,) * 2, (921.0526123046875,) * 2,
         (882.3529052734375,) * 2, (833.3333129882812,) * 2, (769.2307739257812,) * 2, (681.8181762695312,) * 2,
         (555.5555419921875,) * 2, (357.1428527832031,) * 2, (0.0, 0.0)]
FIRST = torch.randn((1, 48, 1, 24, 20), generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)


def refuses(fn, exc=(ValueError, SystemExit)):
    try:
        fn(); return False
    except exc:
        return True


def native(v, a, replay=None, tamper=None):
    """Deterministic native probe trace for seeds (v, a); with replay, the video stream follows the replay."""
    g = torch.Generator().manual_seed(v * 10007 + a)
    eps = torch.randn(SHAPE, generator=g).to(torch.bfloat16); eps[:, :, :1] = FIRST
    steps = []
    for i, (tv, ta) in enumerate(SCHED[:-1]):
        if replay is not None:
            z = replay["steps"][i]["video_latents_in"].clone(); z[:, :, :1] = FIRST
        else:
            z = eps.clone() if i == 0 else torch.randn(SHAPE, generator=g).to(torch.bfloat16)
            z[:, :, :1] = FIRST
        h = int(hashlib.sha256(P.tensor_sha(z).encode()).hexdigest()[:8], 16)
        gg = torch.Generator().manual_seed(h + a)
        steps.append({"i": i, "t_v": tv, "t_a": ta, "sigma_v": tv / 1000, "sigma_a": ta / 1000, "video_latents_in": z,
                      "action_latents_in": torch.randn((1, 32, 80), generator=gg).to(torch.bfloat16),
                      "video_velocity": torch.randn(SHAPE, generator=gg).to(torch.bfloat16),
                      "action_velocity": torch.randn((1, 32, 80), generator=gg).to(torch.bfloat16)})
    fin_v = torch.randn(SHAPE, generator=g).to(torch.bfloat16) if replay is None else replay["final"]["video_latents"].clone()
    fin_v[:, :, :1] = FIRST
    h = int(hashlib.sha256("".join(P.tensor_sha(s["video_latents_in"]) for s in steps).encode()).hexdigest()[:8], 16)
    gg = torch.Generator().manual_seed(h + a)
    pt = {"seeds": {"seed": 42, "video_seed": v, "action_seed": a}, "schedule": SCHED,
          "active_action_mask": torch.ones(80, dtype=torch.bool), "active_action_mask_source": "x", "inactive_action_dims": None,
          "cond": {"first_frame_latents": FIRST.clone()}, "init": {"video_noise": eps.clone()}, "steps": steps,
          "final": {"video_latents": fin_v, "action_latents": torch.randn((1, 32, 80), generator=gg),
                    "actions_normalized": np.asarray(torch.randn((32, 80), generator=gg)),
                    "actions_physical": np.asarray(torch.randn((32, 20), generator=gg))}, "replay": replay is not None}
    if tamper:
        tamper(pt)
    return pt


RECEIVER = {"meta": {"schedule": SCHED, "num_frames": 9, "action_num_frames": 33}, "cond": {"prompt": "p", "first_frame_latents": FIRST}}

# ---------------- native adapter
nt = P.native_trace(native(1001, 2001), RECEIVER, 1001, 2001)
checks["native_trace: adapts a meta-less native trace and binds seeds/schedule/frames/mask"] = nt["meta"]["video_seed"] == 1001 \
    and nt["meta"]["action_seed"] == 2001 and nt["meta"]["num_frames"] == 9 and nt["meta"]["action_num_frames"] == 33 \
    and nt["meta"]["schedule"] == [tuple(map(float, x)) for x in SCHED] and nt["meta"]["active_action_mask"].dtype == torch.bool
checks["native_trace refuses: seeds differ from the request"] = refuses(lambda: P.native_trace(native(1001, 2001), RECEIVER, 1002, 2001))
checks["native_trace refuses: schedule differs from the receiver"] = refuses(lambda: P.native_trace(native(1001, 2001, tamper=lambda p: p.update(schedule=SCHED[:-2] + [(1.0, 1.0), (0.0, 0.0)])), RECEIVER, 1001, 2001))
checks["native_trace refuses: a step timestep differs from the schedule"] = refuses(lambda: P.native_trace(native(1001, 2001, tamper=lambda p: p["steps"][3].update(t_v=1.0)), RECEIVER, 1001, 2001))
checks["native_trace refuses: missing init.video_noise"] = refuses(lambda: P.native_trace(native(1001, 2001, tamper=lambda p: p["init"].pop("video_noise")), RECEIVER, 1001, 2001))
checks["native_trace refuses: an old G0 file with meta (no silent seed-42 meta)"] = refuses(lambda: P.native_trace(dict(native(1001, 2001), meta={}), RECEIVER, 1001, 2001))

# ---------------- clamp / scramble (kept from R79)
g = torch.Generator().manual_seed(0)
target = torch.randn(SHAPE, generator=g).to(torch.bfloat16)
eps = nt["init"]["video_noise"]
z = P.clamp_latents(nt["steps"][5]["video_latents_in"], target, eps, 0.4)
checks["clamp: future = (1-sigma)*target + sigma*eps (fp32 then bf16), frame 0 untouched"] = torch.equal(
    z[:, :, 1:], (0.6 * target[:, :, 1:].float() + 0.4 * eps[:, :, 1:].float()).to(torch.bfloat16)) and torch.equal(z[:, :, :1], FIRST)
checks["clamp: sigma bounds / shape refused"] = refuses(lambda: P.clamp_latents(z, target, eps, 1.2)) and refuses(lambda: P.clamp_latents(z, target, eps, -.1)) \
    and refuses(lambda: P.clamp_latents(z, target[:, :, :2], eps, .5))
rep = P.clamp_replay(nt, target, 1000.0)
checks["clamp_replay on an ADAPTED native trace: every step at sigma=t_v/1000; final = target future + receiver frame 0"] = all(
    torch.equal(r["video_latents_in"][:, :, 1:], ((1 - t / 1000) * target[:, :, 1:].float() + (t / 1000) * eps[:, :, 1:].float()).to(torch.bfloat16))
    for r, (t, _) in zip(rep["steps"], SCHED)) and torch.equal(rep["final"]["video_latents"][:, :, 1:], target[:, :, 1:]) \
    and torch.equal(rep["final"]["video_latents"][:, :, :1], FIRST)
ep = P.endpoint_checks(nt, target, 1000.0)
checks["endpoints pass on an adapted native trace"] = all(ep.values())
perm = P.scramble_permutation("d", 960)
sc = P.scramble_future(target, perm)
checks["scramble: one shared permutation, frame 0 untouched, multiset kept"] = torch.equal(sc[0, :, 1:].reshape(48, -1), target[0, :, 1:].reshape(48, -1)[:, torch.as_tensor(perm)]) \
    and torch.equal(sc[:, :, :1], target[:, :, :1])

# ---------------- identity comparison (complete)
ident = P.native_trace(native(1001, 2001, replay=P.replay_of(nt)), RECEIVER, 1001, 2001)
c = P.identity_compare(nt, ident)
checks["identity: a faithful self-replay is equal on every key incl. final video"] = c["all_equal"] and "final.video_latents" in c["checks"]
for name, mut in [("final video latents only", lambda t: t["final"].__setitem__("video_latents", t["final"]["video_latents"] + 1)),
                  ("one step t_a", lambda t: t["steps"][4].__setitem__("t_a", 1.0)),
                  ("active mask", lambda t: t["meta"].__setitem__("active_action_mask", torch.zeros(80, dtype=torch.bool))),
                  ("first-frame latents", lambda t: t["cond"].__setitem__("first_frame_latents", FIRST + 1)),
                  ("step count", lambda t: t["steps"].pop())]:
    bad = copy.deepcopy(ident); mut(bad)
    checks[f"identity detects a difference in {name}"] = P.identity_compare(nt, bad)["all_equal"] is False
checks["identity journals both sides' final SHAs"] = set(c["final_sha256"]) == set(P.FINAL_KEYS)

# ---------------- source verification
d0 = Path(tempfile.mkdtemp())
pngs, rec_pngs, post = [], {}, []
for off in range(0, 33, 4):
    cams = {}
    for cam in P.CAMERAS:
        p = d0 / f"{off}_{cam}.png"; p.write_bytes(f"{off}{cam}".encode())
        e = {"path": str(p), "sha256": P.sha_file(p)}
        pngs.append({"offset": off, "camera": cam, **e}); cams[cam] = e
    (rec_pngs.update(cams) if off == 0 else post.append({"offset": off, "pngs": cams}))
REC = {"executed_count": 32, "inference_horizon": 32, "termination": "none", "complete": True, "actions": np.zeros((32, 20)).tolist(),
       "actions_executed": [{"offset": i, "count_delta": 1, "exception": None} for i in range(1, 33)], "post_obs": post, "pngs": rec_pngs}


def make_src(rec, pngs_decl=None):
    d = Path(tempfile.mkdtemp())
    (d / "record.json").write_text(json.dumps(rec)); (d / "trace.pt").write_bytes(b"trace")
    return {"trace": str(d / "trace.pt"), "trace_sha256": P.sha_file(d / "trace.pt"), "record": str(d / "record.json"),
            "record_sha256": P.sha_file(d / "record.json"), "pngs": pngs if pngs_decl is None else pngs_decl}


checks["verify_source accepts an eligible source"] = not refuses(lambda: P.verify_source(make_src(REC)))


def bad_rec(name, mut, decl=None):
    r = json.loads(json.dumps(REC)); mut(r)
    checks[f"verify_source refuses: {name}"] = refuses(lambda: P.verify_source(make_src(r, decl)))


bad_rec("truncated execution", lambda r: r.update(executed_count=31))
bad_rec("terminated chunk", lambda r: r.update(termination="plan_fail"))
bad_rec("missing post_obs offset 32", lambda r: r["post_obs"].pop())
bad_rec("actions not [32,20]", lambda r: r.update(actions=np.zeros((31, 20)).tolist()))
bad_rec("an executed step with an exception", lambda r: r["actions_executed"][7].update(exception="boom"))
bad_rec("an executed step with count_delta 0", lambda r: r["actions_executed"][7].update(count_delta=0))
bad_rec("a duplicated executed offset", lambda r: r["actions_executed"][8].update(offset=8))
bad_rec("wrong inference horizon", lambda r: r.update(inference_horizon=16))
bad_rec("PNG declared set differs", lambda r: r["post_obs"][0]["pngs"]["head"].update(sha256="0" * 64))
drop_left = [p for p in pngs if not (p["offset"] == 12 and p["camera"] == "left")]
bad_rec("a camera missing in both record and declaration", lambda r: r["post_obs"][2]["pngs"].pop("left"), decl=drop_left)
s3 = make_src(REC); s3["trace_sha256"] = "0" * 64
checks["verify_source refuses: trace sha mismatch"] = refuses(lambda: P.verify_source(s3))

# ---------------- mock-driven driver: c0 -> identity -> endpoints (stage E), failure and resume protections
import offline_runner  # noqa: E402  (real module; its OfflineRunner is replaced by the mock below)


class MockArch:
    class video_scheduler:
        num_train_timesteps = 1000


class MockRunner:
    calls = []
    identity_tamper = None

    def __init__(self, *a, **k):
        self.arch = MockArch()

    def precompute_prompts(self, *a, **k):
        pass

    def rerun(self, trace, *, video_seed, action_seed, replay=None, record=True):
        MockRunner.calls.append((video_seed, action_seed, replay is not None))
        tamper = MockRunner.identity_tamper if replay is not None else None
        return {"probe_trace": native(video_seed, action_seed, replay=replay, tamper=tamper)}

    def gt_future_frames(self, rec):
        return ["f"] * 9

    def encode_frames(self, frames):
        lat = torch.randn(SHAPE, generator=torch.Generator().manual_seed(len(set(map(str, frames))))).to(torch.bfloat16)
        lat[:, :, :1] = FIRST
        return lat


offline_runner.OfflineRunner = MockRunner
P.check_runtime_live = lambda *a, **k: None
torch.cuda.synchronize = lambda *a, **k: None
srcs = {o: make_src(REC) for o in ("R", "D2", "D3")}
for o in srcs:
    torch.save({"meta": RECEIVER["meta"], "cond": RECEIVER["cond"]}, srcs[o]["trace"])
    srcs[o]["trace_sha256"] = P.sha_file(srcs[o]["trace"])
offline_runner.load_trace = lambda p: torch.load(p, weights_only=False)
MAN = {"sources": srcs, "pairs": [{"receiver": "R", "C2": {"donor": "D2"}, "C3": {"donor": "D3"}}]}
BIND = {"stage": "E", "observations": ["R"], "video_seeds": [1001, 1002], "action_seeds": [2001], "max_reruns": 4}


def driver(out, bind=BIND):
    return P.P1A(out, bind, {}, MAN)


out = Path(tempfile.mkdtemp()) / "e"; out.mkdir()
MockRunner.calls = []
drv = driver(out); drv.run()
j = [json.loads(x) for x in (out / "journal.jsonl").read_text().splitlines()]
units = [r["unit"] for r in j]
checks["driver E (native traces): C0 + identity + endpoints for each seed, 4 forwards, all success records"] = \
    len(MockRunner.calls) == 4 and sum(u.split("|")[1] == "C0" for u in units) == 2 and sum(u.split("|")[1] == "identity" for u in units) == 2 \
    and sum(u.split("|")[1] == "endpoints" for u in units) == 2 and all(r["ok"] is True for r in j)
checks["driver E: identity records carry the full comparison incl. final video"] = all(
    r["identity"]["all_equal"] and r["identity"]["checks"]["final.video_latents"] for r in j if r.get("cond") == "identity")
checks["driver E: no condition forward in stage E"] = all(not (rp and False) for _, _, rp in MockRunner.calls) and \
    sum(1 for _, _, rp in MockRunner.calls if rp) == 2   # the two replays are the identity runs only
MockRunner.calls = []
drv2 = driver(out); drv2.run()
checks["resume of a complete E dir: re-audits artifacts, no new forward"] = MockRunner.calls == []

# cap
out_c = Path(tempfile.mkdtemp()) / "c"; out_c.mkdir()
checks["hard rerun cap stops at max_reruns (3 < 4 needed)"] = refuses(lambda: driver(out_c, dict(BIND, max_reruns=3)).run())
checks["cap counts journalled forwards on resume (still refused)"] = refuses(lambda: driver(out_c, dict(BIND, max_reruns=3)).run())

# identity failure -> failure record, no success record, resume refused
out_f = Path(tempfile.mkdtemp()) / "f"; out_f.mkdir()
MockRunner.identity_tamper = lambda p: p["final"].__setitem__("video_latents", p["final"]["video_latents"] + 1)
checks["identity differing only in final video -> run fails"] = refuses(lambda: driver(out_f).run())
MockRunner.identity_tamper = None
jf = [json.loads(x) for x in (out_f / "journal.jsonl").read_text().splitlines()] if (out_f / "journal.jsonl").exists() else []
checks["failed identity is not journalled as success"] = not any(r.get("cond") == "identity" for r in jf)
checks["a directory with failure records refuses to resume (no silent laundering)"] = refuses(lambda: driver(out_f))

# endpoint failure: same protections
out_p = Path(tempfile.mkdtemp()) / "p"; out_p.mkdir()
orig_ep = P.endpoint_checks
P.endpoint_checks = lambda *a, **k: {"sigma1_equals_receiver_step0": False}
checks["endpoint failure -> run fails"] = refuses(lambda: driver(out_p).run())
P.endpoint_checks = orig_ep
checks["endpoint failure not journalled; resume refused"] = not any(json.loads(x).get("cond") == "endpoints" for x in (out_p / "journal.jsonl").read_text().splitlines()) \
    and refuses(lambda: driver(out_p))


# artifact integrity on resume
def corrupt(name, fn):
    o = Path(tempfile.mkdtemp()) / name
    shutil.copytree(out, o)
    jj = (o / "journal.jsonl").read_text().replace(str(out.resolve()), str(o.resolve()))
    (o / "journal.jsonl").write_text(jj)
    fn(o)
    checks[f"resume refused: {name}"] = refuses(lambda: driver(o))


corrupt("C0 trace changed", lambda o: next((o / "traces").rglob("C0_*.pt")).write_bytes(b"x"))
corrupt("identity npz changed", lambda o: next((o / "units").rglob("identity_*.npz")).write_bytes(b"x"))
corrupt("identity trace deleted", lambda o: next((o / "traces").rglob("identity_*.pt")).unlink())
corrupt("orphan unit file", lambda o: (o / "units" / "R" / "C9_v1_a1.npz").write_bytes(b"x"))
corrupt("duplicate journal unit", lambda o: (o / "journal.jsonl").write_text((o / "journal.jsonl").read_text() + (o / "journal.jsonl").read_text().splitlines()[0] + "\n"))
corrupt("truncated journal line", lambda o: (o / "journal.jsonl").write_text((o / "journal.jsonl").read_text() + '{"format": "p1a'))
corrupt("a non-success record", lambda o: (o / "journal.jsonl").write_text((o / "journal.jsonl").read_text() + json.dumps({"format": P.FORMAT, "unit": "x", "ok": False}) + "\n"))

failed = [k for k, v in checks.items() if not v]
print("RESULT " + json.dumps({"n": len(checks), "failed": failed, "checks": checks}))
sys.exit(1 if failed else 0)
