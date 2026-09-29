"""OpenWAM policy client for canonical native-action LIBERO checkpoints.

The checkpoint returns raw 10-D native-action EEF values::

    [delta_xyz3, rot6d(Exp(delta_axis_angle3)), open_scale_gripper1]

The adapter converts only that representation to LIBERO's runtime 7-D OSC
command.  It does not compose with the current EEF pose and does not apply the
controller's 0.05 m / 0.5 rad output scales.  The live observation sent to the
server is achieved EEF10, matching the canonical training reader.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from benchmarks.utils import (
    WSPolicyClient,
    build_payload,
    encode_numpy_b64,
    libero_obs_to_eef10,
    libero_open_scale_to_gripper_cmd,
    resize_for_lshape_slot,
)
from benchmarks.utils.action_conversion import rot6d_to_axis_angle

LIBERO_ACTION_MODE = "eef"
LIBERO_EEF10_DIM = 10
LIBERO_ACTION7_DIM = 7
PROPRIO_OBS_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")


def native_eef10_to_libero7d(action: np.ndarray, *, clip: bool = True) -> np.ndarray:
    """Convert native-action EEF10 to LIBERO's runtime 7-D OSC command.

    ``action[:3]`` and the decoded rotation vector stay in native normalized
    command space.  The only semantic projection is the gripper sign flip from
    OpenWAM's ``-1 closed / +1 open`` to LIBERO's ``+1 close / -1 open``.
    """
    value = np.asarray(action, dtype=np.float32).reshape(-1)
    if value.shape != (LIBERO_EEF10_DIM,):
        raise ValueError(f"expected a native LIBERO EEF10 action, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("native LIBERO EEF10 action contains NaN or infinity")
    position = value[:3].copy()
    rotation = rot6d_to_axis_angle(value[3:9]).astype(np.float32)
    gripper = np.array([libero_open_scale_to_gripper_cmd(value[9])], dtype=np.float32)
    if clip:
        position = np.clip(position, -1.0, 1.0)
        rotation = np.clip(rotation, -1.0, 1.0)
        gripper = np.clip(gripper, -1.0, 1.0)
    return np.concatenate([position, rotation, gripper]).astype(np.float32)


def _as_list(value) -> list[float]:
    return np.asarray(value, dtype=np.float32).reshape(-1).tolist()


def _build_state(obs: dict, keys: Iterable[str]) -> list[float]:
    state: list[float] = []
    missing: list[str] = []
    for key in keys:
        if key not in obs:
            missing.append(key)
            continue
        state.extend(_as_list(obs[key]))
    if missing:
        raise KeyError(f"LIBERO obs missing state key(s): {missing}")
    return state


_DEBUG_IMAGE_SUFFIX = "_image"


def _numeric_obs(obs: dict) -> dict:
    """Flatten the numeric (non-image) entries of a LIBERO observation.

    Camera frames are skipped -- they are already captured in the rollout video --
    but object poses (``*_pos`` / ``*_quat``), the EEF pose, gripper joints and
    ``object-state`` are all kept, so object trajectories can be lined up against
    the per-step action stream. Stored per step by ``OpenWAMLiberoPolicy._maybe_debug``.
    """
    collected = {}
    for key, value in obs.items():
        if key.endswith(_DEBUG_IMAGE_SUFFIX) or value is None:
            continue
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if array.ndim > 2:
            continue
        collected[key] = array.tolist()
    return collected


class OpenWAMLiberoPolicy:
    """WebSocket policy client for the canonical native-action contract."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8848,
        request_timeout: int = 300,
        action_mode: str = LIBERO_ACTION_MODE,
        head_camera_key: str = "agentview_image",
        left_wrist_camera_key: str | None = "robot0_eye_in_hand_image",
        right_wrist_camera_key: str | None = None,
        image_transform: str = "rotate_180",
        send_state: bool = True,
        state_keys: list[str] | None = None,
        state_dim: int | None = LIBERO_EEF10_DIM,
        action_dim: int = LIBERO_ACTION7_DIM,
        action_indices: list[int] | None = None,
        action_clip: float | None = None,
        debug: bool = False,
        debug_dir: str = "./debug_libero",
        paired_capture: dict | None = None,
        _client=None,
    ) -> None:
        if str(action_mode).strip().lower() != LIBERO_ACTION_MODE:
            raise ValueError(
                f"native-delta LIBERO client requires action_mode={LIBERO_ACTION_MODE!r}, got {action_mode!r}"
            )
        if image_transform not in ("none", "rotate_180"):
            raise ValueError("image_transform must be 'none' or 'rotate_180'")
        if int(action_dim) != LIBERO_ACTION7_DIM:
            raise ValueError(f"native-delta LIBERO runtime action_dim must be {LIBERO_ACTION7_DIM}")

        self._client = _client or WSPolicyClient(f"ws://{host}:{port}", timeout=request_timeout)
        self._head_camera_key = head_camera_key
        self._left_wrist_camera_key = left_wrist_camera_key
        self._right_wrist_camera_key = right_wrist_camera_key
        self._image_transform = image_transform
        self._send_state = bool(send_state)
        self._state_keys = state_keys or []
        self._state_dim = state_dim
        self._action_dim = int(action_dim)
        self._action_indices = action_indices
        self._action_clip = action_clip
        self._debug = bool(debug)
        self._debug_dir = Path(debug_dir)
        paired_capture = paired_capture or {}
        self._paired_capture_enabled = paired_capture.get("enabled", False) is True
        self._paired_decode_world = paired_capture.get("decode_world", True) is True
        self._paired_trials = {int(value) for value in paired_capture.get("trials", [])}
        self._paired_steps = {int(value) for value in paired_capture.get("capture_steps", [])}
        self._paired_dir = Path(paired_capture.get("output_dir", "./paired_rollouts"))
        self._last_generation: dict | None = None
        self._episode = -1
        self._step = 0
        if self._debug:
            self._debug_dir.mkdir(parents=True, exist_ok=True)
        if self._paired_capture_enabled:
            self._paired_dir.mkdir(parents=True, exist_ok=True)

        pong = self._client.ping()
        if pong.get("type") != "pong":
            raise RuntimeError(f"OpenWAM server ping returned unexpected response: {pong}")
        if pong.get("representation") != LIBERO_ACTION_MODE:
            raise RuntimeError(
                f"representation mismatch: eval requires {LIBERO_ACTION_MODE!r}, "
                f"server advertises {pong.get('representation')!r}"
            )
        print(
            f"[OpenWAMLiberoPolicy] server=ws://{host}:{port} "
            f"action_mode={LIBERO_ACTION_MODE} model_action_dim={LIBERO_EEF10_DIM} "
            f"runtime_action_dim={self._action_dim} image_transform={image_transform} send_state={send_state}"
        )

    def close(self) -> None:
        self._client.close()

    def reset(self, episode: int | None = None) -> None:
        self._episode = self._episode + 1 if episode is None else int(episode)
        self._step = 0
        self._last_generation = None
        ack = self._client.reset()
        if ack.get("type") != "reset_ack":
            raise RuntimeError(f"OpenWAM server reset returned unexpected response: {ack}")

    def act(self, obs: dict, prompt: str) -> np.ndarray:
        payload = build_payload(
            head=encode_numpy_b64(resize_for_lshape_slot(self._image(obs, self._head_camera_key), "head_camera")),
            left_wrist=self._maybe_encode(obs, self._left_wrist_camera_key, "left_wrist_camera"),
            right_wrist=self._maybe_encode(obs, self._right_wrist_camera_key, "right_wrist_camera"),
            prompt=prompt,
            state=self._state(obs),
        )
        capture_generation = (
            self._paired_capture_enabled
            and self._episode in self._paired_trials
            and self._step in self._paired_steps
        )
        # Main evaluation always uses the exact baseline request. Selected
        # inputs are retained client-side and replayed only after all trials.
        response = self._client.predict(payload)
        self._last_generation = None
        model_action = np.asarray(response["action"], dtype=np.float32).reshape(-1)
        if self._action_indices is not None:
            model_action = model_action[self._action_indices]
        if model_action.shape != (LIBERO_EEF10_DIM,):
            raise ValueError(
                f"OpenWAM returned action dim {model_action.shape[0]}, expected {LIBERO_EEF10_DIM} "
                "raw native-delta EEF10 values"
            )
        action = native_eef10_to_libero7d(model_action)
        if self._action_clip is not None:
            action = np.clip(action, -float(self._action_clip), float(self._action_clip))
        self._maybe_debug(obs, payload, model_action, action)
        if capture_generation:
            directory = self._paired_dir / f"trial_{self._episode:03d}" / f"step_{self._step:04d}"
            (directory / "predicted").mkdir(parents=True, exist_ok=True)
            self._last_generation = {
                "trial": self._episode,
                "chunk_start_step": self._step,
                "directory": str(directory),
                "replay_payload": payload,
                "online_native_eef10_action": model_action.copy(),
                "online_libero_action7": action.copy(),
            }
        self._step += 1
        return action


    def pop_generation(self) -> dict | None:
        """Return and clear the most recently captured joint generation."""
        generation, self._last_generation = self._last_generation, None
        return generation

    def decode_generation(self, generation: dict) -> None:
        """Generate a paired world/action sample from a saved main-run input."""
        payload = dict(generation.pop("replay_payload"))
        payload["capture_generation"] = True
        payload["decode_generation"] = self._paired_decode_world
        ack = self._client.reset()
        if ack.get("type") != "reset_ack":
            raise RuntimeError(f"OpenWAM replay reset returned unexpected response: {ack}")
        replay = self._client.predict(payload).get("generation")
        if replay is None:
            raise RuntimeError("deferred replay did not return a fresh generation")

        native_actions = np.asarray(replay["actions"], dtype=np.float32)
        if native_actions.ndim != 2 or native_actions.shape[1] != LIBERO_EEF10_DIM:
            raise ValueError(f"replayed action chunk has unexpected shape {native_actions.shape}")
        runtime_actions = np.stack([native_eef10_to_libero7d(value) for value in native_actions])
        if self._action_clip is not None:
            runtime_actions = np.clip(runtime_actions, -float(self._action_clip), float(self._action_clip))

        encoded_frames = replay.get("video_jpeg_b64", [])
        directory = Path(generation["directory"])
        predicted_dir = directory / "predicted"
        for index, encoded in enumerate(encoded_frames):
            (predicted_dir / f"frame_{index:03d}.jpg").write_bytes(base64.b64decode(encoded))

        frame_count = len(encoded_frames)
        action_count = len(native_actions)
        if frame_count > 1 and action_count % (frame_count - 1) == 0:
            stride = action_count // (frame_count - 1)
            offsets = list(range(0, action_count + 1, stride))
        else:
            stride = None
            offsets = []
        online_native = np.asarray(generation["online_native_eef10_action"], dtype=np.float32)
        first_action_error = float(np.max(np.abs(native_actions[0] - online_native)))
        first_action_matches = bool(np.array_equal(native_actions[0], online_native))
        generation.update(
            inference_horizon=int(replay["inference_horizon"]),
            native_eef10_actions=native_actions,
            libero_action7=runtime_actions,
            predicted_frame_count=frame_count,
            predicted_action_stride=stride,
            predicted_action_offsets=offsets,
            replay_first_action_matches_online_exactly=first_action_matches,
            replay_first_action_max_abs_error=first_action_error,
        )
        metadata = {
            key: value
            for key, value in generation.items()
            if key not in {
                "directory",
                "native_eef10_actions",
                "libero_action7",
                "online_native_eef10_action",
                "online_libero_action7",
            }
        }
        metadata["native_eef10_actions"] = native_actions.tolist()
        metadata["libero_action7"] = runtime_actions.tolist()
        metadata["online_native_eef10_action"] = online_native.tolist()
        metadata["online_libero_action7"] = np.asarray(
            generation["online_libero_action7"], dtype=np.float32
        ).tolist()
        (directory / "generation.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def _image(self, obs: dict, key: str) -> np.ndarray:
        if key not in obs:
            raise KeyError(f"LIBERO obs missing camera key: {key}")
        image = np.asarray(obs[key])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Camera '{key}' must be HxWx3, got {image.shape}")
        image = image.astype(np.uint8, copy=False)
        return image[::-1, ::-1] if self._image_transform == "rotate_180" else image

    def _maybe_encode(self, obs: dict, key: str | None, slot: str) -> str | None:
        if not key:
            return None
        if key not in obs or obs[key] is None:
            return None
        return encode_numpy_b64(resize_for_lshape_slot(self._image(obs, key), slot))

    def _state(self, obs: dict) -> list[float] | None:
        if not self._send_state:
            return None
        if all(key in obs for key in PROPRIO_OBS_KEYS):
            state = libero_obs_to_eef10(
                obs["robot0_eef_pos"],
                obs["robot0_eef_quat"],
                obs["robot0_gripper_qpos"],
            ).tolist()
        else:
            state = _build_state(obs, self._state_keys)
        if self._state_dim is not None and len(state) != int(self._state_dim):
            raise ValueError(f"LIBERO state dim {len(state)} != expected {self._state_dim}")
        return state

    def _maybe_debug(self, obs: dict, payload: dict, model_action: np.ndarray, action: np.ndarray) -> None:
        if not self._debug:
            return
        step_dir = self._debug_dir / f"episode_{self._episode:03d}" / f"step_{self._step:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "action_mode": LIBERO_ACTION_MODE,
            "episode": self._episode,
            "step": self._step,
            "state_dim": len(payload.get("state", [])) if payload.get("state") is not None else None,
            "state": payload.get("state"),
            "native_eef10_action": model_action.tolist(),
            "libero_action7": action.tolist(),
            "obs_keys": sorted(obs.keys()),
            "obs_numeric": _numeric_obs(obs),
        }
        (step_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


__all__ = [
    "LIBERO_ACTION7_DIM",
    "LIBERO_ACTION_MODE",
    "LIBERO_EEF10_DIM",
    "OpenWAMLiberoPolicy",
    "native_eef10_to_libero7d",
]
