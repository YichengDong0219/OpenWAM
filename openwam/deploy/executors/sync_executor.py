"""Synchronous inference executor: buffer-and-replan over engine chunks.

The default execution mode. Generates an action chunk when the executable
buffer runs out and returns one action per control step. If
``inference_horizon`` is shorter than the generated chunk, the unused tail is
discarded rather than mixed into later predictions.

Mirrors :class:`AsyncInferenceExecutor`'s interface
(``predict_action(conditions)`` / ``reset()`` / ``shutdown()``) so
:class:`~openwam.deploy.policy.WAMPolicy` can pick either executor at
construction time.
"""

from collections import deque
from typing import Optional

import numpy as np

from openwam.deploy.engine import BaseInferenceEngine


class SyncInferenceExecutor:
    """Receding-horizon execution of engine-generated action chunks.

    Args:
        engine: Inference engine that generates action chunks.
        inference_horizon: Number of actions to execute before re-generating.
            ``None`` means consume the full chunk (greedy).
    """

    def __init__(
        self,
        engine: BaseInferenceEngine,
        inference_horizon: Optional[int] = None,
    ):
        self.engine = engine
        if inference_horizon is not None and inference_horizon <= 0:
            raise ValueError("inference_horizon must be positive")
        self.inference_horizon = inference_horizon

        self._action_buffer: deque = deque()
        self._last_generation: Optional[dict] = None

    def predict_action(self, conditions: dict) -> np.ndarray:
        """Pop the next action, regenerating after the configured horizon."""
        if len(self._action_buffer) == 0:
            self._generate_and_enqueue(conditions)

        action = self._action_buffer.popleft()
        return action

    def predict_action_with_generation(self, conditions: dict) -> tuple[np.ndarray, Optional[dict]]:
        """Return the next action and a fresh joint generation when replanning.

        This opt-in research hook leaves predict_action unchanged. The
        generation payload is returned only on the first action of a newly
        generated chunk, so buffered steps do not duplicate video data.
        """
        generated = len(self._action_buffer) == 0
        if generated:
            self._generate_and_enqueue(conditions)
        action = self._action_buffer.popleft()
        return action, self._last_generation if generated else None

    def _generate_and_enqueue(self, conditions: dict):
        """Run inference and keep only the executable action horizon."""
        probe = conditions.get("probe")
        recovery = probe.get("recovery") if isinstance(probe, dict) else None
        if recovery is not None:
            result = self._recovery_generate(conditions, recovery)
        else:
            result = self.engine.generate(conditions)
        actions = result["actions"]
        if hasattr(actions, "cpu"):
            actions = actions.detach().cpu().numpy()

        chunk_len = len(actions)
        if chunk_len <= 0:
            raise RuntimeError("Inference result did not contain any actions")

        inference_horizon = self.inference_horizon if self.inference_horizon is not None else chunk_len
        if inference_horizon > chunk_len:
            raise ValueError(f"inference_horizon ({inference_horizon}) must be <= action horizon ({chunk_len})")

        self._action_buffer.clear()
        self._action_buffer.extend(actions[:inference_horizon])
        self._last_generation = {
            "actions": np.asarray(actions).copy(),
            "video": result.get("video"),
            "inference_horizon": inference_horizon,
        }
        if result.get("probe_receipt") is not None:
            self._last_generation["probe_receipt"] = result["probe_receipt"]
        if result.get("recovery_receipt") is not None:
            self._last_generation["recovery_receipt"] = result["recovery_receipt"]

    _RECOVERY_KEYS = {"arm", "ref_trace_path", "ref_video_seed", "ref_action_seed",
                      "cand_trace_path", "cand_video_seed", "cand_action_seed"}

    def _recovery_generate(self, conditions: dict, recovery: dict) -> dict:
        """Recovery arms B1/B2: two full generations from the SAME conditions, no action executed in between.

        1. reference proposal R (recorded, seeds ref_*); its actions are never executed;
        2. candidate (recorded, seeds cand_*): B1 = ordinary joint generation; B2 = future frames clamped to R's
           final video latents, re-noised with the candidate's own seeded noise at every executed step.
        Only the candidate's actions are enqueued. Both arms spend exactly two generations."""
        if not isinstance(recovery, dict) or set(recovery) != self._RECOVERY_KEYS:
            raise ValueError(f"recovery request must have exactly {sorted(self._RECOVERY_KEYS)}")
        if recovery["arm"] not in ("B1", "B2"):
            raise ValueError(f"recovery arm must be B1 or B2; got {recovery['arm']!r}")
        if set(conditions["probe"]) != {"recovery"}:
            raise ValueError("a recovery probe must not carry other probe fields")
        base = {k: v for k, v in conditions.items() if k != "probe"}
        ref = self.engine.generate({**base, "probe": {"trace_path": recovery["ref_trace_path"],
                                                      "video_seed": int(recovery["ref_video_seed"]),
                                                      "action_seed": int(recovery["ref_action_seed"])}})
        if ref.get("probe_receipt") is None:
            raise RuntimeError("recovery reference generation wrote no trace")
        cand_probe = {"trace_path": recovery["cand_trace_path"], "video_seed": int(recovery["cand_video_seed"]),
                      "action_seed": int(recovery["cand_action_seed"])}
        if recovery["arm"] == "B2":
            cand_probe["clamp_target_trace_path"] = ref["probe_receipt"]["trace_path"]
        cand = self.engine.generate({**base, "probe": cand_probe})
        if cand.get("probe_receipt") is None:
            raise RuntimeError("recovery candidate generation wrote no trace")
        cand = dict(cand)
        cand["recovery_receipt"] = {"arm": recovery["arm"], "ref": ref["probe_receipt"],
                                    "cand": cand["probe_receipt"], "n_generations": 2}
        return cand

    def reset(self):
        """Clear state between episodes."""
        self._action_buffer.clear()
        self._last_generation = None

    def shutdown(self):
        """No background resources; present for executor-interface symmetry."""
