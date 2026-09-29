# LIBERO spatial/task05: paired world-action diagnostic

**Updated:** 2026-09-29
**Branch:** `research/world-action-paired-rollout`
**Status:** implementation and task05 experiment complete

## Research question

The task is `pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate`. Failed videos suggested that the gripper lifts the bowl together with its ramekin base. The diagnostic asks whether the paired world prediction also moves the base, or whether the world prediction separates the bowl while the paired action chunk fails physically.

## Final protocol

The standard evaluation path remains unchanged while the 50 main trials run. For selected trial/step pairs, the client only retains the exact observation payload, online action, and flattened MuJoCo state.

After all main trials finish:

1. Replay each saved observation once and request both the decoded world frames and complete action chunk from the same generation.
2. Check that the replayed first action equals the online action for that observation.
3. Restore a second LIBERO environment from the saved MuJoCo state.
4. Execute the first 10 actions of the replayed chunk and save images plus numeric object states.

This deferred design matters. Creating a second MuJoCo environment or decoding world frames during the main run changed later trajectories in exploratory experiments.

## Sampling

- Trials: `0, 2, 8, 21, 26, 30, 41, 44`
- Policy steps: `20, 30, 40, 50, 60`
- Paired samples: 40
- Predicted frames per sample: 9
- Generated actions per sample: 32
- Counterfactual execution horizon: 10 actions

## Integrity results

- Main task result: **49/50**; only trial 2 failed.
- Paired generations written: **40/40**.
- Counterfactual rollouts written: **40/40**.
- Replayed first action exactly matched the online first action: **40/40**.
- Maximum first-action absolute error: **0**.

## Main evidence

The target object is `akita_black_bowl_1`; the base is `glazed_rim_porcelain_ramekin_1`.

| Capture | Bowl displacement in 10 actions | Base displacement | Bowl-base distance change |
|---|---:|---:|---:|
| trial 2, step 50 | 0.1002 m | 0.0886 m | +0.0017 m |
| seven successful controls, step 50 mean | 0.0587 m | 0.00007 m | -0.0271 m |
| trial 2, step 60 | 0.1420 m | 0.1400 m | +0.0009 m |
| seven successful controls, step 60 mean | 0.1572 m | 0.00003 m | -0.0653 m |

The largest base displacement among successful controls at these stages is 0.00018 m. In trial 2, the bowl and ramekin move almost equally and retain nearly the same relative distance, confirming that the action chunk moves them as a stack.

Visual inspection at steps 50 and 60 shows the predicted world moving the black bowl while leaving the red ramekin on the table. The actual paired rollout moves the bowl and ramekin together. This supports a world/action alignment failure for trial 2: the world prediction represents the intended separation, while physical execution of the paired action chunk does not realize it.

## Claims suitable for writing

- The paired trial 2 evidence supports the hypothesis that at least one task05 failure comes from a gap between the predicted world transition and the physical result of the paired action chunk.
- The action branch quantitatively reproduces the bowl-plus-base motion seen in the failure video.
- Deferred replay avoids perturbing the measured main rollout, and its first action matches the online action exactly for all 40 samples.

## Claims to avoid

- Do not claim that all task05 failures are caused by world/action misalignment; the final run contains one failure.
- Do not claim that trials 2, 8, 21, and 41 fail deterministically.
- Do not use success-rate differences between separately launched servers as a causal estimate of instrumentation effects.

## Cross-launch nondeterminism

A paired-off run produced 46/50 with failures `{2, 8, 21, 41}`. The final run used baseline-identical requests during the main evaluation but produced 49/50 with failure `{2}`. GPU/model inference is therefore not reproducible across server launches under the current setup.

Exploratory results are retained only as diagnostics:

| Run | Successes | Failures | Interpretation |
|---|---:|---|---|
| paired off | 46/50 | 2, 8, 21, 41 | Reference launch |
| online action plus same-process branch | 48/50 | 2, 21 | Instrumentation perturbed execution |
| online world decode | 49/50 | 2 | Instrumentation perturbed execution |
| final deferred replay | 49/50 | 2 | Valid paired artifact run; separate server launch |

A future statistical comparison should repeat every condition across multiple server launches or capture and restore all PyTorch/CUDA RNG and determinism settings.

## Code and configuration

- `benchmarks/libero/policy_config_paired.yml`: opt-in task05 sampling configuration.
- `benchmarks/libero/openwam2libero_interface.py`: saves inputs during the main run and materializes paired generations afterward.
- `benchmarks/libero/single_eval.py`: saves MuJoCo states and executes deferred counterfactual rollouts.
- `openwam/deploy/{engine,policy,server}.py` and `openwam/deploy/executors/sync_executor.py`: opt-in complete generation return; default responses are unchanged.
- `tests/benchmarks/test_libero_bridge.py` and `tests/test_policy_binary_projection.py`: opt-in and chunk-capture coverage.

## Reproduction command

```bash
SERVER_PYTHON=/home/dongyicheng/miniconda3/envs/openwam/bin/python \
LIBERO_PYTHON=/home/dongyicheng/miniconda3/envs/libero_openwam/bin/python \
LIBERO_PATH=/home/dongyicheng/OpenWAM/third_party/LIBERO \
POLICY_CONFIG_PATH=/home/dongyicheng/OpenWAM/benchmarks/libero/policy_config_paired.yml \
GPUS=0 REPLICAS_PER_GPU=1 \
OUTPUT_DIR=/home/dongyicheng/OpenWAM/outputs/libero/openwam_alpha_sim_libero/paired_world_action_replay_v2_task05 \
bash benchmarks/libero/run_eval.sh \
  /home/dongyicheng/OpenWAM/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-LIBERO \
  checkpoint_step_10690.safetensors \
  --suites libero_spatial --task-ids 5 --num-trials 50
```

## Artifacts

Root:

`/home/dongyicheng/OpenWAM/outputs/libero/openwam_alpha_sim_libero/paired_world_action_replay_v2_task05`

- `summary.json`: evaluation summary.
- `analysis/report.md`: standalone experiment report.
- `analysis/world_actual_contact.jpg`: predicted-world versus actual-rollout contact sheet.
- `videos/libero_spatial/task_05/trials_000_049/attempt_01/paired_rollouts/`: all 40 paired samples.
