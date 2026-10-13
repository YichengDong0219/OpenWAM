"""Opt-in G0 records. No RNG draws; observations are supplied by the eval loop."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import numpy as np
import cv2


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def pose_dict(pose):
    return {"p": jsonable(pose.p), "q": jsonable(pose.q)}


def task_fields(env):
    fields = {}
    robot = env.robot
    for arm in ("left", "right"):
        fields[f"{arm}_gripper"] = float(getattr(robot, f"get_{arm}_gripper_val")())
    task = env.task_name
    if task == "turn_switch":
        face = -env.switch.get_pose().to_transformation_matrix()[:3, 0]
        fields.update(face_dir=face, arm_tag="right" if face[0] > 0 else "left",
                      switch_qpos=env.switch.get_qpos())
    elif task == "hanging_mug":
        fields.update(mug_functional_point=env.mug.get_functional_point(0),
                      rack_pose=pose_dict(env.rack.get_pose()),
                      rack_functional_point=env.rack.get_functional_point(0))
    elif task == "place_empty_cup":
        fields.update(cup_pose=pose_dict(env.cup.get_pose()),
                      coaster_pose=pose_dict(env.coaster.get_pose()),
                      cup_functional_point=env.cup.get_functional_point(0),
                      coaster_functional_point=env.coaster.get_functional_point(0),
                      arm_tag="right" if env.cup.get_pose().p[0] > 0 else "left")
    else:
        raise ValueError(f"G0 task fields are not defined for {task!r}")
    return jsonable(fields)


def save_cameras(directory, observation):
    directory.mkdir(parents=True, exist_ok=True)
    result = {}
    for label, key in (("head", "head_camera"), ("left", "left_camera"), ("right", "right_camera")):
        image = observation["observation"].get(key, {}).get("rgb")
        if image is None:
            raise ValueError(f"G0 requires camera {key}")
        path = directory / f"{label}.png"
        if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
            raise OSError(f"Could not save {path}")
        result[label] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "shape": list(image.shape), "dtype": str(image.dtype)}
    return result


@contextmanager
def observe_plans(env, events):
    """Observe planner return status without changing planner results or control flow."""
    replaced = []
    try:
        for arm in ("left", "right"):
            name = f"{arm}_plan_path"
            original = getattr(env.robot, name, None)
            if original is None:
                continue
            # Preserve whether the instance originally shadowed its class method.
            existed = name in vars(env.robot)
            previous = vars(env.robot).get(name)
            def wrapped(*args, _method=original, _arm=arm, **kwargs):
                try:
                    result = _method(*args, **kwargs)
                except Exception as exc:
                    events.append({"arm": _arm, "exception": type(exc).__name__, "message": str(exc)})
                    raise
                events.append({"arm": _arm, "status": str(result.get("status"))})
                return result
            setattr(env.robot, name, wrapped)
            replaced.append((name, existed, previous))
        yield
    finally:
        for name, existed, previous in replaced:
            if existed:
                setattr(env.robot, name, previous)
            else:
                delattr(env.robot, name)


class ProbeCapture:
    def __init__(self, root, *, video_seed=None, action_seed=None):
        self.root = Path(root).resolve()
        self.video_seed = video_seed
        self.action_seed = action_seed
        self.active = None
        self.directory = None
        self.episode = -1
        self.chunk = 0

    def reset(self, episode):
        if self.active is not None:
            self.finish("none", incomplete_reason="episode_reset")
        self.episode, self.chunk = episode, 0

    def request(self, env):
        path = self.root / env.task_name / f"ep{self.episode:04d}" / f"chunk{self.chunk:04d}.pt"
        return {"trace_path": str(path), "video_seed": self.video_seed, "action_seed": self.action_seed,
                "replay_trace_path": None, "replay_mode": None}

    def observe(self, env, observation, *, finish_complete=True):
        """Use the next ordinary observation; avoid extra get_obs/RNG calls mid-rollout."""
        if self.active is None:
            return
        offset = int(env.take_action_cnt) - self.active["take_action_cnt"]
        if offset > 0 and offset % 4 == 0 and offset not in self.active["observed_offsets"]:
            pngs = save_cameras(self.directory / f"post{offset:04d}", observation)
            self.active["post_obs"].append({"offset": offset, "pngs": pngs,
                                             "endpose": jsonable(observation.get("endpose")),
                                             "task_fields": task_fields(env)})
            self.active["observed_offsets"].append(offset)
        if finish_complete and self.active["executed_count"] >= self.active["inference_horizon"]:
            self.finish("none")

    def response(self, env, observation, *, instruction, prompt, proprio, request, response):
        generation = response.get("generation")
        receipt = response.get("probe") or (generation or {}).get("probe")
        if receipt is None:
            if self.active is None:
                raise RuntimeError("probe requested a new chunk but server returned no probe receipt")
            return
        if self.active is not None:
            raise RuntimeError("server generated a new chunk before the previous horizon completed")
        if not generation or not generation.get("actions"):
            raise ValueError("probe receipt must accompany full generation actions")
        if receipt.get("trace_path") != request["trace_path"]:
            raise ValueError("server probe trace_path does not match request")
        if not receipt.get("sha256") or int(receipt.get("n_steps", 0)) <= 0:
            raise ValueError("incomplete probe receipt")
        actions = generation["actions"]
        horizon = int(generation["inference_horizon"])
        if horizon <= 0 or horizon > len(actions):
            raise ValueError("invalid generation inference_horizon")
        self.directory = Path(request["trace_path"]).with_suffix("")
        # A run must not quietly overwrite an earlier episode/chunk.
        self.directory.mkdir(parents=True, exist_ok=False)
        pngs = save_cameras(self.directory, observation)
        self.active = {"schema_version": 1, "episode": self.episode,
                       "env_seed": getattr(env, "_probe_env_seed", None), "task": env.task_name,
                       "chunk": self.chunk, "instruction": instruction, "prompt": prompt,
                       "take_action_cnt": int(env.take_action_cnt),
                       "endpose": jsonable(observation.get("endpose")), "proprio": jsonable(proprio),
                       "probe": request, "probe_receipt": receipt, "server_response": response,
                       "actions": actions, "inference_horizon": horizon, "pngs": pngs,
                       "executed_count": 0, "termination": "none", "task_fields": task_fields(env),
                       "termination_semantics": "plan_fail means take_action exception; non-Success planner statuses do not terminate baseline",
                       "post_obs": [], "observed_offsets": [], "actions_executed": [], "complete": False}
        self.chunk += 1
        self.persist()

    def after_action(self, env, action, before_count, planner_events, *, exception=None):
        if self.active is None:
            raise RuntimeError("no active chunk for action execution")
        delta = int(env.take_action_cnt) - int(before_count)
        if delta not in (0, 1):
            raise ValueError(f"unexpected take_action count delta: {delta}")
        self.active["executed_count"] += delta
        self.active["actions_executed"].append({"offset": self.active["executed_count"],
             "env_action": jsonable(action), "count_delta": delta, "planner": planner_events,
             "exception": str(exception) if exception else None})
        termination = ("plan_fail" if exception is not None else "success" if env.eval_success else
                       "step_lim" if env.take_action_cnt >= env.step_lim else "none")
        self.persist()
        return termination

    def persist(self):
        atomic_json(self.directory / "record.json", self.active)

    def finish(self, termination, **extra):
        if self.active is None:
            return
        self.active.update(termination=termination, complete=not bool(extra.get("incomplete_reason")), **extra)
        self.active["missing_offsets"] = [i for i in range(4, len(self.active["actions"]) + 1, 4)
                                          if i not in self.active["observed_offsets"]]
        self.persist()
        self.active = None


def append_seed_audit(root, *, task, seed, eligible, reason, episode=None, error=None):
    path = Path(root) / task / "seed_audit.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"task": task, "seed": int(seed), "accepted": bool(eligible), "reason": reason,
           "episode": episode, "error": error}
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
