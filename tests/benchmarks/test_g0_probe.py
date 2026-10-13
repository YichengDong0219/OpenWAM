"""G0 data integrity: fake env/transport, no model or simulator needed."""
import json
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.robotwin.probe_capture import ProbeCapture, append_seed_audit, observe_plans
from benchmarks.robotwin import openwam2robotwin_interface as iface


def obs():
    return {"observation": {k: {"rgb": np.full((3, 4, 3), i, dtype=np.uint8)}
            for i, k in enumerate(("head_camera", "left_camera", "right_camera"))}, "endpose": {"test": 1}}


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = SimpleNamespace(task_name="place_empty_cup", take_action_cnt=0,
                                   eval_success=False, step_lim=32, _probe_env_seed=100000)
        self.fields = patch("benchmarks.robotwin.probe_capture.task_fields", return_value={"q": 1})
        self.fields.start()
        self.cap = ProbeCapture(self.temp.name)
        self.cap.reset(0)

    def tearDown(self):
        self.fields.stop()
        self.temp.cleanup()

    def start(self, horizon=8):
        request = self.cap.request(self.env)
        response = {"action": [0.0] * 20, "generation": {"actions": [[0.0] * 20] * horizon,
                     "inference_horizon": horizon}, "probe": {"trace_path": request["trace_path"],
                     "sha256": "0" * 64, "n_steps": 10}}
        self.cap.response(self.env, obs(), instruction="raw", prompt="wrapped", proprio=None,
                          request=request, response=response)
        return request, response

    def test_cached_actions_are_not_new_chunks_and_offsets_are_actual(self):
        self.start()
        first = self.cap.directory
        for i in range(8):
            if i:
                self.cap.observe(self.env, obs())
                self.cap.response(self.env, obs(), instruction="raw", prompt="wrapped", proprio=None,
                                  request=self.cap.request(self.env), response={"action": [0.0] * 20})
            before = self.env.take_action_cnt
            self.env.take_action_cnt += 1
            self.cap.after_action(self.env, [1], before, [])
        self.cap.observe(self.env, obs())
        d = json.loads((first / "record.json").read_text())
        self.assertEqual(d["executed_count"], 8)
        self.assertEqual([p["offset"] for p in d["post_obs"]], [4, 8])
        self.assertEqual(d["missing_offsets"], [])
        self.assertTrue(d["complete"])
        self.assertEqual(self.cap.chunk, 1)
        self.assertEqual(len(list(Path(self.temp.name).glob("*/ep*/chunk*/record.json"))), 1)

    def test_terminal_at_full_horizon_retains_success(self):
        self.start(horizon=4)
        directory = self.cap.directory
        for i in range(4):
            self.env.take_action_cnt = i + 1
            self.env.eval_success = i == 3
            terminal = self.cap.after_action(self.env, [1], i, [])
        self.cap.observe(self.env, obs(), finish_complete=False)
        self.cap.finish(terminal)
        d = json.loads((directory / "record.json").read_text())
        self.assertEqual(d["termination"], "success")
        self.assertEqual(d["executed_count"], 4)

    def test_short_terminal_has_missing_unexecuted_future(self):
        self.start(horizon=8)
        directory = self.cap.directory
        self.env.take_action_cnt = 1
        self.cap.after_action(self.env, [1], 0, [])
        self.cap.finish("success")
        d = json.loads((directory / "record.json").read_text())
        self.assertEqual(d["executed_count"], 1)
        self.assertEqual(d["missing_offsets"], [4, 8])

    def test_missing_receipt_and_duplicate_paths_fail_closed(self):
        with self.assertRaises(RuntimeError):
            self.cap.response(self.env, obs(), instruction="", prompt="", proprio=None,
                              request=self.cap.request(self.env), response={"action": [0]})
        self.start()
        self.cap.finish("step_lim")
        self.cap.reset(0)
        with self.assertRaises(FileExistsError):
            self.start()

    def test_seed_audit_preserves_rejections(self):
        for seed, eligible, reason in ((100000, False, "expert_plan_fail"), (100001, True, "accepted")):
            append_seed_audit(self.temp.name, task="place_empty_cup", seed=seed, eligible=eligible, reason=reason)
        lines = (Path(self.temp.name) / "place_empty_cup/seed_audit.jsonl").read_text().splitlines()
        self.assertEqual([json.loads(x)["accepted"] for x in lines], [False, True])

    def test_plan_observer_restores_methods_and_does_not_change_results(self):
        result = {"status": "Failure"}
        class Robot:
            def left_plan_path(self, _target):
                return result
        self.env.robot = Robot()
        events = []
        with observe_plans(self.env, events):
            self.assertIs(self.env.robot.left_plan_path(None), result)
        self.assertNotIn("left_plan_path", vars(self.env.robot))
        self.assertEqual(events, [{"arm": "left", "status": "Failure"}])

    def test_default_step_payload_has_no_probe_and_probe_disables_retry(self):
        model = iface.ModelClient.__new__(iface.ModelClient)
        model._task_description = "pick"
        model._send_state = False
        model._step, model._debug, model._action_indices = 0, False, None
        calls = []
        class Client:
            def predict(self, payload):
                calls.append(("default", payload))
                return {"action": [0.] * 20}
            def predict_once(self, payload):
                calls.append(("once", payload))
                return {"action": [0.] * 20}
        model._client = Client()
        example = {"cams": {"head": np.zeros((3, 4, 3), np.uint8)}, "lang": "pick"}
        model.step(example)
        model.step({**example, "_probe": {"trace_path": "test.pt"}})
        self.assertNotIn("probe", calls[0][1])
        self.assertEqual(calls[1][0], "once")


class VerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[2] / "scripts/probe/verify_g0.py"
        spec = importlib.util.spec_from_file_location("verify_g0", path)
        cls.verify = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.verify)

    def test_invalid_shape_or_nonfinite_cannot_pass_comparison(self):
        for a, b in (([1], [1, 2]), ([float("nan")], [0]), ([0], [float("inf")])):
            with self.assertRaises(ValueError):
                self.verify.difference(a, b)

    def test_noop_threshold_is_derived_from_repeated_measurements(self):
        def trace(x):
            return {"final": {"actions_physical": [x], "action_latents": [x]},
                    "steps": [{"i": 0, "action_velocity": [x]}]}
        baseline = [trace(x) for x in (1., 1.01, 1.02, 1.01, 1.)]
        tol = self.verify.tolerance_from_repeats(baseline)
        self.assertTrue(self.verify.compare_traces(baseline[0], trace(1.02), tol)["passed"])
        self.assertFalse(self.verify.compare_traces(baseline[0], trace(1.04), tol)["passed"])
        zero = self.verify.tolerance_from_repeats([trace(1.)] * 5)
        self.assertEqual(zero["final.action_latents"]["atol"], 0.)


if __name__ == "__main__":
    unittest.main()
