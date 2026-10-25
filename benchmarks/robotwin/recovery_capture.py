"""Test-time failure-recovery arms on top of the G0 probe capture (hanging_mug, privileged stall trigger).

Arms (one per run, selected by ROBOTWIN_RECOVERY_ARM):
  B0  original policy; the trigger is evaluated and logged, nothing is changed;
  B1  at a trigger, the next chunk is generated as reference R + ordinary candidate (two generations);
  B2  same R, candidate future frames clamped to R's final video latents (two generations).
Only the candidate chunk is executed; afterwards the ordinary policy resumes.

The trigger is a SIMULATOR-PRIVILEGED diagnostic (task_fields), not a deployable visual detector. It is evaluated
only when a new chunk is about to be generated, from fully executed earlier chunks (causal). Parameters are frozen
in TRIGGER_DEFAULTS / the run binding; this module never looks at the episode outcome.
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path

from benchmarks.robotwin.probe_capture import ProbeCapture, task_fields

TRIGGER_DEFAULTS = {"K": 2, "eps_m": 0.01, "closed": 0.1, "open": 0.9, "max_corrections": 2, "cooldown_chunks": 2}
# Seeds of the k-th correction in an episode, shared by all arms (B0 never uses them).
RECOVERY_SEEDS = {0: {"ref": (1001, 2001), "cand": (3001, 4001)}, 1: {"ref": (1002, 2002), "cand": (3002, 4002)}}


def chunk_samples(record: dict) -> list[dict]:
    """task_fields at the chunk start followed by the post-observation samples (4-step cadence)."""
    out = [record["task_fields"]]
    out += [p["task_fields"] for p in record.get("post_obs", []) if p.get("task_fields")]
    return out


def mug_geometric_success(tf: dict) -> bool:
    m, r = tf["mug_functional_point"][:3], tf["rack_functional_point"][:3]
    return abs(m[0] - r[0]) < 0.02 and abs(m[1] - r[1]) < 0.02 and m[2] > 0.86


def mug_stall_trigger(completed: list[dict], params: dict = TRIGGER_DEFAULTS) -> dict:
    """Decide, before generating the next chunk, whether the hang attempt has stalled.

    completed: fully executed chunk records of this episode, in order. With c = the last completed chunk:
      range  = max pairwise distance of the mug functional point over all samples of the last K chunks;
      phase  = the right gripper was <= closed at some sample of chunks before c (the hang attempt has started;
               hanging_mug first moves the mug with the other arm, so a lift-based gate fires in that handoff pause);
      open   = right gripper >= open at the last sample of c;
      not geometric success at the last sample of c.
    """
    k = int(params["K"])
    diag = {"n_completed": len(completed), "range_m": None, "phase": False, "grip_open": False, "geom_success": None}
    if len(completed) < k:
        return {"trigger": False, **diag}
    samples = [s for rec in completed[-k:] for s in chunk_samples(rec)]
    pts = [s["mug_functional_point"][:3] for s in samples]
    rng = max((math.dist(a, b) for a, b in itertools.combinations(pts, 2)), default=0.0)
    earlier = [s for rec in completed[:-1] for s in chunk_samples(rec)]
    phase = any(s["right_gripper"] <= params["closed"] for s in earlier)
    last = chunk_samples(completed[-1])[-1]
    grip_open = last["right_gripper"] >= params["open"]
    geom = mug_geometric_success(last)
    diag.update(range_m=rng, phase=phase, grip_open=grip_open, geom_success=geom)
    return {"trigger": bool((not geom) and phase and rng <= params["eps_m"] and grip_open), **diag}


class RecoveryCapture(ProbeCapture):
    """ProbeCapture that turns the first chunk after a trigger into a recovery request (arms B1/B2)."""

    def __init__(self, root, *, arm: str, video_seed=None, action_seed=None, params: dict | None = None):
        super().__init__(root, video_seed=video_seed, action_seed=action_seed)
        if arm not in ("B0", "B1", "B2"):
            raise ValueError(f"arm must be B0/B1/B2; got {arm!r}")
        self.arm = arm
        self.params = dict(TRIGGER_DEFAULTS if params is None else params)
        self.completed: list[dict] = []
        self.events: list[dict] = []
        self.n_corrections = 0
        self.last_correction_chunk = None
        self._pending = None

    def reset(self, episode):
        super().reset(episode)
        self.completed, self.events = [], []
        self.n_corrections, self.last_correction_chunk, self._pending = 0, None, None

    def _cooldown_ok(self) -> bool:
        if self.last_correction_chunk is None:
            return True
        return self.chunk - self.last_correction_chunk - 1 >= int(self.params["cooldown_chunks"])

    def request(self, env):
        if self.active is not None:  # buffered step: the server will not generate
            return super().request(env)
        if self._pending is not None and self._pending["chunk"] == self.chunk:
            return self._pending["request"]
        decision = mug_stall_trigger(self.completed, self.params)
        eligible = decision["trigger"] and self.n_corrections < int(self.params["max_corrections"]) and self._cooldown_ok()
        event = {"chunk": self.chunk, "take_action_cnt": int(env.take_action_cnt), "arm": self.arm,
                 "eligible_trigger": bool(eligible), **decision}
        req = super().request(env)
        if eligible:
            seeds = RECOVERY_SEEDS[self.n_corrections]
            event["correction_index"] = self.n_corrections
            if self.arm in ("B1", "B2"):
                base = Path(req["trace_path"])
                req = {"recovery": {"arm": self.arm,
                                    "ref_trace_path": str(base.with_name(base.stem + "_ref.pt")),
                                    "ref_video_seed": seeds["ref"][0], "ref_action_seed": seeds["ref"][1],
                                    "cand_trace_path": str(base), "cand_video_seed": seeds["cand"][0],
                                    "cand_action_seed": seeds["cand"][1]}}
            self.n_corrections += 1
            self.last_correction_chunk = self.chunk
        self.events.append(event)
        self._pending = {"chunk": self.chunk, "request": req}
        return req

    def response(self, env, observation, *, instruction, prompt, proprio, request, response):
        rec = request.get("recovery") if isinstance(request, dict) else None
        if rec is None:
            return super().response(env, observation, instruction=instruction, prompt=prompt, proprio=proprio,
                                    request=request, response=response)
        receipt = response.get("recovery")
        if receipt is None or receipt.get("arm") != rec["arm"] or receipt.get("n_generations") != 2:
            raise RuntimeError("recovery request without a matching recovery receipt")
        if receipt["cand"]["trace_path"] != rec["cand_trace_path"] or receipt["ref"]["trace_path"] != rec["ref_trace_path"]:
            raise ValueError("recovery receipt trace paths do not match the request")
        flat = {"trace_path": rec["cand_trace_path"], "video_seed": rec["cand_video_seed"],
                "action_seed": rec["cand_action_seed"], "recovery": rec}
        super().response(env, observation, instruction=instruction, prompt=prompt, proprio=proprio,
                         request=flat, response=response)
        self.active["recovery_receipt"] = receipt
        self.persist()

    def after_action(self, env, action, before_count, planner_events, *, exception=None):
        termination = super().after_action(env, action, before_count, planner_events, exception=exception)
        if termination != "none":
            # Actual termination-step state, whether or not it falls on the 4-step cadence.
            self.active["terminal_task_fields"] = task_fields(env)
            self.active["terminal_take_action_cnt"] = int(env.take_action_cnt)
            self.active["terminal_eval_success"] = bool(env.eval_success)
            self.persist()
        return termination

    def finish(self, termination, **extra):
        active = self.active
        super().finish(termination, **extra)
        if active is not None:
            self.completed.append(active)
            events_path = self.root / active["task"] / f"ep{self.episode:04d}" / "recovery_events.json"
            from benchmarks.robotwin.probe_capture import atomic_json
            atomic_json(events_path, {"arm": self.arm, "params": self.params, "seeds": RECOVERY_SEEDS,
                                      "n_corrections": self.n_corrections, "events": self.events})
