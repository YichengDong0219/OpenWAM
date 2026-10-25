"""CPU unit tests for the recovery hooks (executor orchestration; B2 clamp == P1A clamp_replay semantics)."""
import ast, sys, types
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_function(path, cls, name):
    """Extract one method from a big module without importing its heavy dependencies."""
    tree = ast.parse(Path(path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name == name:
                    mod = ast.Module(body=[f], type_ignores=[])
                    ns = {"Tensor": torch.Tensor, "torch": torch}
                    exec(compile(mod, str(path), "exec"), ns)
                    return ns[name]
    raise KeyError(name)


def p1a_clamp_latents(step_latents, target, eps, sigma):  # verbatim semantics of scripts/probe/p1a_claude.clamp_latents
    z = step_latents.clone()
    z[:, :, 1:] = ((1.0 - sigma) * target[:, :, 1:].float() + sigma * eps[:, :, 1:].float()).to(step_latents.dtype)
    return z


def test_clamp_matches_p1a():
    tree = ast.parse((ROOT / "openwam/model/architectures/base.py").read_text())
    owner = next(n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
                 and any(isinstance(f, ast.FunctionDef) and f.name == "_clamp_target_probe" for f in n.body))
    clamp = load_function(ROOT / "openwam/model/architectures/base.py", owner, "_clamp_target_probe")
    self = types.SimpleNamespace(video_scheduler=types.SimpleNamespace(num_train_timesteps=1000))
    g = torch.Generator().manual_seed(0)
    raw = torch.randn(1, 4, 3, 2, 2, generator=g).to(torch.bfloat16)
    target = torch.randn(1, 4, 3, 2, 2, generator=g).to(torch.bfloat16)
    schedule = [(1000.0, 1000.0), (900.0, 900.0), (900.0, 900.0), (500.0, 400.0), (357.0, 300.0), (0.0, 0.0)]
    out = clamp(self, {"clamp_target": target, "record": True}, schedule, raw, 9, 33)
    assert "clamp_target" not in out and out["record"] is True
    assert sorted(out["replay_video_steps"]) == [0, 2, 3, 4]  # step 1 is a plateau (no update)
    for i, z in out["replay_video_steps"].items():
        ref = p1a_clamp_latents(raw, target, raw, schedule[i][0] / 1000.0)
        assert torch.equal(z, ref), i
        assert out["replay_t_v"][i] == schedule[i][0]
    assert torch.equal(out["replay_video_steps"][0][:, :, 1:], raw[:, :, 1:])  # sigma=1: pure own noise
    assert torch.equal(out["replay_video_final"], target)
    assert out["replay_meta"] == {"num_frames": 9, "action_num_frames": 33}
    try:
        clamp(self, {"clamp_target": target[:, :, :2]}, schedule, raw, 9, 33); raise AssertionError("shape not checked")
    except ValueError:
        pass
    try:
        clamp(self, {"clamp_target": target, "replay_video_steps": {}}, schedule, raw, 9, 33); raise AssertionError
    except ValueError:
        pass


class StubEngine:
    def __init__(self):
        self.calls = []

    def generate(self, conditions):
        self.calls.append(conditions)
        p = conditions.get("probe") or {}
        tag = len(self.calls)
        res = {"actions": np.full((33, 20), float(tag)), "video": None}
        if p.get("trace_path"):
            res["probe_receipt"] = {"trace_path": p["trace_path"], "sha256": f"s{tag}", "n_steps": 10}
        return res


def load_executor():
    sys.path.insert(0, str(ROOT))
    src = (ROOT / "openwam/deploy/executors/sync_executor.py").read_text()
    ns = {}
    # drop package-relative imports; only deque/np/Optional are needed
    lines = [l for l in src.splitlines() if not (l.startswith("from .") or l.startswith("from openwam"))]
    code = "\n".join(lines).replace("engine: BaseInferenceEngine", "engine")
    exec(compile(code, "sync_executor.py", "exec"), ns)
    return next(v for k, v in ns.items() if isinstance(v, type) and hasattr(v, "_recovery_generate"))


def rec(arm):
    return {"arm": arm, "ref_trace_path": "/p/R.pt", "ref_video_seed": 7, "ref_action_seed": 8,
            "cand_trace_path": "/p/C.pt", "cand_video_seed": 9, "cand_action_seed": 10}


def test_executor_b1_b2():
    Ex = load_executor()
    for arm in ("B1", "B2"):
        eng = StubEngine(); ex = Ex(eng, inference_horizon=32)
        cond = {"observation": {}, "prompt": "x", "probe": {"recovery": rec(arm)}}
        a, gen = ex.predict_action_with_generation(cond)
        assert len(eng.calls) == 2
        r, c = eng.calls
        assert r["probe"] == {"trace_path": "/p/R.pt", "video_seed": 7, "action_seed": 8}
        exp = {"trace_path": "/p/C.pt", "video_seed": 9, "action_seed": 10}
        if arm == "B2":
            exp["clamp_target_trace_path"] = "/p/R.pt"
        assert c["probe"] == exp
        assert {k: v for k, v in r.items() if k != "probe"} == {k: v for k, v in c.items() if k != "probe"}
        assert float(a[0]) == 2.0  # candidate actions executed, never R's
        assert len(ex._action_buffer) == 31 and all(float(x[0]) == 2.0 for x in ex._action_buffer)
        assert gen["recovery_receipt"]["n_generations"] == 2 and gen["probe_receipt"]["trace_path"] == "/p/C.pt"
        # buffered steps do not regenerate
        ex.predict_action_with_generation({"observation": {}})
        assert len(eng.calls) == 2


def test_executor_rejects_bad():
    Ex = load_executor()
    ex = Ex(StubEngine(), inference_horizon=32)
    bad = rec("B3")
    for probe in ({"recovery": bad}, {"recovery": {**rec("B1"), "extra": 1}}, {"recovery": rec("B1"), "trace_path": "/x"}):
        try:
            ex._generate_and_enqueue({"probe": probe}); raise AssertionError(probe)
        except ValueError:
            pass


def test_plain_path_unchanged():
    Ex = load_executor()
    eng = StubEngine(); ex = Ex(eng, inference_horizon=32)
    ex.predict_action({"observation": {}, "probe": {"trace_path": "/p/N.pt"}})
    assert len(eng.calls) == 1 and eng.calls[0]["probe"] == {"trace_path": "/p/N.pt"}


def load_trigger():
    import importlib.util, types as T
    pkg = T.ModuleType("benchmarks"); sub = T.ModuleType("benchmarks.robotwin")
    pc = T.ModuleType("benchmarks.robotwin.probe_capture")
    pc.ProbeCapture = object; pc.task_fields = lambda env: {}
    sys.modules.update({"benchmarks": pkg, "benchmarks.robotwin": sub, "benchmarks.robotwin.probe_capture": pc})
    spec = importlib.util.spec_from_file_location("rc", ROOT / "benchmarks/robotwin/recovery_capture.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_trigger_reproduces_m0_on_g0():
    import json
    m = load_trigger()
    recs = sorted((ROOT.parent / "g0_mug_records").rglob("record.json"))
    by_ep = {}
    for p in recs:
        d = json.loads(p.read_text())
        by_ep.setdefault(d["episode"], []).append(d)
    m0 = json.loads((Path("/newhome/dongyicheng/robotwin_probe_dialogue/evidence/recovery_M0_claude/m0_mug_states.json")).read_text())
    for ep, rs in by_ep.items():
        rs.sort(key=lambda d: d["chunk"])
        online = [n for n in range(1, len(rs) + 1) if m.mug_stall_trigger(rs[:n])["trigger"]]
        m0_trig = [r["chunk"] for r in m0["episodes"][f"hanging_mug_ep{ep}"]["rows"] if r["trigger_candidate"]]
        # online decision before chunk n uses completed chunks < n, i.e. M0's trigger at chunk n-1
        assert online == [c + 1 for c in m0_trig if c + 1 <= len(rs)], (ep, online, m0_trig)
    assert not any(m.mug_stall_trigger(by_ep[e][:n])["trigger"] for e in (0, 3, 4) for n in range(1, len(by_ep[e]) + 1))


def test_request_flow_cooldown_and_cap():
    import importlib.util, json, types as T, tempfile
    for k in [k for k in sys.modules if k.startswith("benchmarks")]:
        del sys.modules[k]
    sys.path.insert(0, str(ROOT))
    from benchmarks.robotwin.recovery_capture import RecoveryCapture
    recs = sorted((ROOT.parent / "g0_mug_records").rglob("record.json"))
    ep1 = sorted([json.loads(p.read_text()) for p in recs if json.loads(p.read_text())["episode"] == 1], key=lambda d: d["chunk"])
    env = T.SimpleNamespace(task_name="hanging_mug", take_action_cnt=0)
    for arm in ("B0", "B1", "B2"):
        cap = RecoveryCapture(tempfile.mkdtemp(), arm=arm)
        cap.reset(1)
        kinds = []
        for n in range(len(ep1)):
            cap.chunk = n
            cap.completed = ep1[:n]
            req = cap.request(env)
            assert cap.request(env) is req  # repeated call for the same chunk is idempotent
            kinds.append("rec" if "recovery" in req else "norm")
            if "recovery" in req:
                r = req["recovery"]
                assert r["arm"] == arm and r["cand_trace_path"].endswith(f"chunk{n:04d}.pt")
                assert r["ref_trace_path"].endswith(f"chunk{n:04d}_ref.pt")
        trig = [e["chunk"] for e in cap.events if e["eligible_trigger"]]
        assert trig == [12, 15], (arm, trig)  # first at 12; cooldown 2 normal chunks; capped at 2 corrections
        if arm == "B0":
            assert "rec" not in kinds
        else:
            assert [i for i, k in enumerate(kinds) if k == "rec"] == [12, 15]
        idx = [e.get("correction_index") for e in cap.events if e["eligible_trigger"]]
        assert idx == [0, 1]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
