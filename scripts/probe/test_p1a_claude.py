"""CPU tests for p1a_claude's pure construction (clamp, scramble, replay dict, endpoints, source verification).
No model, no GPU. Usage: test_p1a_claude.py"""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p1a_claude as P  # noqa: E402

checks = {}
g = torch.Generator().manual_seed(0)
shape = (1, 48, 3, 24, 20)
target = torch.randn(shape, generator=g).to(torch.bfloat16)
eps = torch.randn(shape, generator=g).to(torch.bfloat16)
first = torch.randn((1, 48, 1, 24, 20), generator=g).to(torch.bfloat16)
sched = [1000.0, 978.2608642578125, 952.3810424804688, 921.0526123046875, 882.3529663085938, 833.3333740234375,
         769.2307739257812, 681.8181762695312, 555.5555419921875, 357.1428527832031]


def fake_trace():
    steps = []
    for i, t in enumerate(sched):
        z = torch.randn(shape, generator=g).to(torch.bfloat16)
        if i == 0:
            z = eps.clone()
        z[:, :, :1] = first
        steps.append({"i": i, "t_v": t, "video_latents_in": z})
    fin = torch.randn(shape, generator=g).to(torch.bfloat16); fin[:, :, :1] = first
    return {"init": {"video_noise": eps.clone()}, "steps": steps, "final": {"video_latents": fin},
            "meta": {"schedule": [(t, t) for t in sched], "num_frames": 9, "action_num_frames": 33, "active_action_mask": None}}


tr = fake_trace()
# clamp formula
z = P.clamp_latents(tr["steps"][5]["video_latents_in"], target, eps, 0.4)
want = (0.6 * target[:, :, 1:].float() + 0.4 * eps[:, :, 1:].float()).to(torch.bfloat16)
checks["clamp: future = (1-sigma)*target + sigma*eps (fp32 then bf16)"] = torch.equal(z[:, :, 1:], want)
checks["clamp: frame 0 untouched"] = torch.equal(z[:, :, :1], tr["steps"][5]["video_latents_in"][:, :, :1])
checks["clamp: sigma=1 -> eps exactly"] = torch.equal(P.clamp_latents(tr["steps"][3]["video_latents_in"], target, eps, 1.0)[:, :, 1:], eps[:, :, 1:])
checks["clamp: sigma=0 -> target exactly"] = torch.equal(P.clamp_latents(tr["steps"][3]["video_latents_in"], target, eps, 0.0)[:, :, 1:], target[:, :, 1:])
for bad, name in ((1.2, "sigma>1"), (-0.1, "sigma<0")):
    try:
        P.clamp_latents(z, target, eps, bad); checks[f"clamp refuses {name}"] = False
    except ValueError:
        checks[f"clamp refuses {name}"] = True
try:
    P.clamp_latents(z, target[:, :, :2], eps, 0.5); checks["clamp refuses shape mismatch"] = False
except ValueError:
    checks["clamp refuses shape mismatch"] = True
# NOT the reversed (R78) convention
rev = (0.4 * target[:, :, 1:].float() + 0.6 * eps[:, :, 1:].float()).to(torch.bfloat16)
checks["clamp: not the reversed R78 convention"] = not torch.equal(z[:, :, 1:], rev)

# replay dict
rep = P.clamp_replay(tr, target, 1000.0)
checks["replay: one step per receiver step with the receiver's t_v"] = [s["t_v"] for s in rep["steps"]] == sched
checks["replay: every step's future = clamp at sigma=t_v/1000"] = all(
    torch.equal(r["video_latents_in"][:, :, 1:], ((1 - t / 1000) * target[:, :, 1:].float() + (t / 1000) * eps[:, :, 1:].float()).to(torch.bfloat16))
    for r, t in zip(rep["steps"], sched))
checks["replay: final = clean target future + receiver first frame"] = torch.equal(rep["final"]["video_latents"][:, :, 1:], target[:, :, 1:]) \
    and torch.equal(rep["final"]["video_latents"][:, :, :1], first)
checks["replay: meta carries schedule / frames / mask"] = set(rep["meta"]) == {"schedule", "num_frames", "action_num_frames", "active_action_mask"}

# endpoints
ep = P.endpoint_checks(tr, target, 1000.0)
checks["endpoints all pass on a consistent trace"] = all(ep.values())
tr_bad = fake_trace(); tr_bad["init"]["video_noise"] = torch.randn(shape, generator=g).to(torch.bfloat16)
checks["endpoints: wrong epsilon source -> sigma1 check fails"] = P.endpoint_checks(tr_bad, target, 1000.0)["sigma1_equals_receiver_step0"] is False
other = torch.randn(shape, generator=g).to(torch.bfloat16)
checks["endpoints: sigma=1 step independent of target"] = torch.equal(P.clamp_replay(tr, other, 1000.0)["steps"][0]["video_latents_in"],
                                                                       rep["steps"][0]["video_latents_in"])

# scramble
perm = P.scramble_permutation("hanging_mug_ep0003_chunk0006", 960)
seed = int.from_bytes(hashlib.sha256(b"P1A-R75|scramble|hanging_mug_ep0003_chunk0006").digest()[:8], "big")
checks["scramble: permutation = PCG64(SHA256 first 8 bytes big-endian)"] = np.array_equal(perm, np.random.Generator(np.random.PCG64(seed)).permutation(960))
checks["scramble: deterministic per donor, differs across donors"] = np.array_equal(perm, P.scramble_permutation("hanging_mug_ep0003_chunk0006", 960)) \
    and not np.array_equal(perm, P.scramble_permutation("turn_switch_ep0001_chunk0003", 960))
s = P.scramble_future(target, perm)
checks["scramble: frame 0 untouched"] = torch.equal(s[:, :, :1], target[:, :, :1])
checks["scramble: per-channel multiset preserved"] = all(
    torch.equal(torch.sort(s[0, c, 1:].flatten().float())[0], torch.sort(target[0, c, 1:].flatten().float())[0]) for c in range(48))
fl_t, fl_s = target[0, :, 1:].reshape(48, -1), s[0, :, 1:].reshape(48, -1)
checks["scramble: one permutation shared by all channels (cell vectors move together)"] = torch.equal(fl_s, fl_t[:, torch.as_tensor(perm)])
checks["scramble: actually changes the future"] = not torch.equal(s, target)
try:
    P.scramble_future(target, perm[:10]); checks["scramble refuses wrong-length permutation"] = False
except ValueError:
    checks["scramble refuses wrong-length permutation"] = True

# source verification on a synthetic record
d = Path(tempfile.mkdtemp())
pngs = []
rec_pngs = {}
post = []
for off in range(0, 33, 4):
    cams = {}
    for cam in ("head", "left", "right"):
        p = d / f"{off}_{cam}.png"; p.write_bytes(f"{off}{cam}".encode())
        e = {"path": str(p), "sha256": P.sha_file(p)}
        pngs.append({"offset": off, "camera": cam, **e}); cams[cam] = e
    if off == 0:
        rec_pngs = cams
    else:
        post.append({"offset": off, "pngs": cams})
rec = {"executed_count": 32, "termination": "none", "complete": True, "actions": np.zeros((32, 20)).tolist(),
       "actions_executed": [{}] * 32, "post_obs": post, "pngs": rec_pngs}
(d / "record.json").write_text(json.dumps(rec)); (d / "trace.pt").write_bytes(b"trace")
src = {"trace": str(d / "trace.pt"), "trace_sha256": P.sha_file(d / "trace.pt"), "record": str(d / "record.json"),
       "record_sha256": P.sha_file(d / "record.json"), "pngs": pngs}
try:
    P.verify_source(src); checks["verify_source accepts an eligible source"] = True
except ValueError as e:
    checks["verify_source accepts an eligible source"] = False


def refused_src(mut, name):
    d2 = Path(tempfile.mkdtemp()); r2 = json.loads(json.dumps(rec)); mut(r2)
    (d2 / "record.json").write_text(json.dumps(r2))
    s2 = dict(src, record=str(d2 / "record.json"), record_sha256=P.sha_file(d2 / "record.json"))
    try:
        P.verify_source(s2); checks[f"verify_source refuses: {name}"] = False
    except ValueError:
        checks[f"verify_source refuses: {name}"] = True


refused_src(lambda r: r.update(executed_count=31), "truncated execution")
refused_src(lambda r: r.update(termination="plan_fail"), "terminated chunk")
refused_src(lambda r: r["post_obs"].pop(), "missing offset 32")
refused_src(lambda r: r.update(actions=np.zeros((31, 20)).tolist()), "actions not [32,20]")
refused_src(lambda r: r["post_obs"][0]["pngs"]["head"].update(sha256="0" * 64), "PNG set differs from the declared set")
s3 = dict(src, trace_sha256="0" * 64)
try:
    P.verify_source(s3); checks["verify_source refuses: trace sha mismatch"] = False
except ValueError:
    checks["verify_source refuses: trace sha mismatch"] = True
(d / pngs[4]["path"].split("/")[-1]).write_bytes(b"tampered")
try:
    P.verify_source(src); checks["verify_source refuses: tampered PNG bytes"] = False
except ValueError:
    checks["verify_source refuses: tampered PNG bytes"] = True

failed = [k for k, v in checks.items() if not v]
print("RESULT " + json.dumps({"n": len(checks), "failed": failed, "checks": checks}))
sys.exit(1 if failed else 0)
