"""Result-blind G2 registration and gated future holdout observation selection.

Standard library only; never imports a model or reads G2 action results. Registration
is a candidate until Claude's independent review and a separate joint freeze receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random

TASKS = ('hanging_mug', 'place_empty_cup', 'turn_switch')
VIDEO = [1001, 1002, 1003, 1004]
ACTION = [2001, 2002, 2003, 2004]
STARTS = dict(zip(TASKS, (100007, 100005, 100006)))
STAGES = (('early', 'middle'), ('middle', 'late'), ('early', 'late'))
CENTERS = {'early': 1/6, 'middle': 1/2, 'late': 5/6}
G1_SHA = 'ec333887c4dbc8e6bed2c9348ee1d6448ae72c7eadeef2ef42695a00ca7e4539'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('refusing to overwrite a different protocol: '+str(path))
        return
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def gt_reasons(d):
    """Availability predicates; termination and planner outcome do not gate eligibility."""
    reasons = []
    if d.get('complete') is not True: reasons.append('record_incomplete')
    if d.get('inference_horizon') != 32: reasons.append('horizon_not_32')
    if d.get('executed_count') != 32: reasons.append('executed_count_not_32')
    actions = d.get('actions_executed', [])
    if len(actions) != 32: reasons.append('execution_rows_not_32')
    if any(a.get('offset') != i+1 or a.get('count_delta') != 1 or a.get('exception') is not None
           for i, a in enumerate(actions)): reasons.append('execution_alignment_or_exception')
    if sorted(x.get('offset', -1) for x in d.get('post_obs', [])) != list(range(4, 33, 4)):
        reasons.append('post_offsets_not_full_window')
    return reasons


def source_observation(o, *, g1_source=True):
    """Integrity failure is fatal, never silently recoded as GT unavailability."""
    trace, record = Path(o['trace']), Path(o['record'])
    if sha(trace) != o['trace_sha256'] or sha(record) != o['record_sha256']:
        raise ValueError('source hash mismatch: '+o['obs_id'])
    d = read(record)
    if any(d[k] != o[k] for k in ('task', 'episode', 'chunk')):
        raise ValueError('source identity mismatch: '+o['obs_id'])
    receipt = d['probe_receipt']
    if Path(receipt['trace_path']).resolve() != trace.resolve() or receipt['sha256'] != o['trace_sha256']:
        raise ValueError('trace receipt mismatch: '+o['obs_id'])
    commands = d['actions']
    if len(commands) != 32 or any(len(a) != 20 or any(not math.isfinite(v) for v in a) for a in commands):
        raise ValueError('invalid physical20 commands: '+o['obs_id'])
    if d.get('complete') is not True:
        raise ValueError('incomplete source record: '+o['obs_id'])
    pngs = []
    for entry in [d, *sorted(d['post_obs'], key=lambda x: x['offset'])]:
        if set(entry['pngs']) != {'head', 'left', 'right'}:
            raise ValueError('camera set mismatch: '+o['obs_id'])
        for camera in ('head', 'left', 'right'):
            item = entry['pngs'][camera]
            if sha(item['path']) != item['sha256']:
                raise ValueError('PNG hash mismatch: '+str(item['path']))
            pngs.append(dict(offset=entry.get('offset', 0), camera=camera, **item))
    reasons = gt_reasons(d)
    result = {k: o[k] for k in ('obs_id', 'task', 'episode', 'chunk', 'trace', 'trace_sha256',
                              'record', 'record_sha256', 'selection_slot', 'requested_stage',
                              'relative_stage', 'relative_progress', 'selection_fallback')}
    result.update({
        'split': 'explore', 'source_g1_split': o['split'] if g1_source else None, 'n_chunks': o['n_chunks'],
        'episode_outcome': o['episode_outcome'],
        'gt': {'eligible': not reasons, 'exclusion_reasons': reasons,
               'executed_count': d['executed_count'], 'termination': d['termination'],
               'observed_offsets': sorted(x['offset'] for x in d['post_obs']),
               'planner_failure_count': sum(p.get('status') != 'Success'
                   for a in d['actions_executed'] for p in a.get('planner', [])),
               'command_reference': 'record.actions, physical20 dispatched EEF targets; not realized EEF'},
        'pngs': pngs})
    return result


def select_holdout(rows):
    """Pure deterministic selection from a complete metadata inventory, no effect values."""
    rows = sorted(rows, key=lambda x: (x['task'], x['episode'], x['chunk']))
    expected = {(t, e) for t in TASKS for e in range(3)}
    if {(r['task'], r['episode']) for r in rows} != expected:
        raise ValueError('holdout requires exactly three new episodes per task')
    rng = random.Random(20261017)
    out = []
    for task in TASKS:
        for ep in range(3):
            group = [r for r in rows if (r['task'], r['episode']) == (task, ep)]
            n = len(group)
            if n < 2 or [r['chunk'] for r in group] != list(range(n)):
                raise ValueError(f'{task}/{ep}: insufficient/noncontiguous chunks; no episode replacement')
            selected = set()
            for slot, stage in enumerate(STAGES[ep]):
                def progress(r): return (r['chunk']+0.5)/n
                def stage_of(r):
                    return 'early' if progress(r) < 1/3 else 'middle' if progress(r) < 2/3 else 'late'
                candidates = [r for r in group if r['chunk'] not in selected and stage_of(r) == stage]
                fallback = not candidates
                row = rng.choice(candidates) if candidates else min(
                    (r for r in group if r['chunk'] not in selected),
                    key=lambda r: (abs(progress(r)-CENTERS[stage]), r['chunk']))
                selected.add(row['chunk'])
                out.append(dict(row, split='holdout', n_chunks=n, selection_slot=slot,
                                requested_stage=stage, relative_stage=stage_of(row),
                                relative_progress=progress(row), selection_fallback=fallback))
    return out


def build(g1_path, tasks_path, draft_path, peer_path):
    if sha(g1_path) != G1_SHA: raise ValueError('G1 preregistration hash mismatch')
    g1 = read(g1_path)
    if len(g1['observations']) != 30 or len({o['obs_id'] for o in g1['observations']}) != 30:
        raise ValueError('expected all 30 unique G1 observations')
    observations = [source_observation(o) for o in g1['observations']]
    if len({(o['task'], o['episode']) for o in observations}) != 15:
        raise ValueError('expected 15 source episodes')
    pairs = [{'pair_kind': 'primary_same_action', 'receiver': [v, a], 'donor': [w, a]}
             for v in VIDEO for a in ACTION for w in VIDEO if w != v]
    action_pairs = [{'video_seed': v, 'action_seed_pair': [a, b]}
                    for v in VIDEO for i, a in enumerate(ACTION) for b in ACTION[i+1:]]
    eligible = [o['obs_id'] for o in observations if o['gt']['eligible']]
    if len(eligible) != 26: raise ValueError('existing GT inventory differs from independently reviewed 26')
    baseline_indexes = {}
    for split in ('explore', 'holdout'):
        root = Path(g1_path).resolve().parent.parent / ('g1_'+split+'_20261014')
        if (root/'PREREG_SHA256').read_text().strip() != G1_SHA:
            raise ValueError('G1 baseline source protocol mismatch')
        baseline_indexes[split] = {'root': str(root), 'index_sha256': sha(root/'index.jsonl'),
                                  'policy': 'original G1 split maps to baseline root; all G2 observations now explore; verify per-file indexed hashes before reuse'}
    verification = []
    for task in TASKS:
        group = [o for o in observations if o['task'] == task]
        first = min(group, key=lambda o: (o['episode'], o['selection_slot']))
        gt_first = min((o for o in group if o['gt']['eligible']), key=lambda o: o['obs_id'])
        verification.append({'task': task, 'obs_id': first['obs_id'], 'gt_obs_id': gt_first['obs_id'],
                             'receiver': [1001, 2001], 'donor': [1002, 2001],
                             'stage1_independence_action_seeds': [2001, 2002]})
    return {
        'format': 'openwam-g2-prereg/1',
        'approval': 'candidate_pending_independent_review; separate joint freeze receipt required',
        'provenance': {name: {'path': str(path), 'sha256': sha(path)} for name, path in
            [('g1_prereg', g1_path), ('tasks', tasks_path), ('protocol_candidate', draft_path),
             ('claude_r16', peer_path), ('sampler', Path(__file__).resolve())]},
        'scope': ['Q5_alpha_full_trajectory_swap', 'Q6_IDM_clean_future', 'Q6_GT_command_consistency'],
        'excluded': ['training', 'closed_loop_intervention', 'Q7_unappointed_semantic_raters',
                     'cross_observation_future', 'shifted_action_pairs', 'cross_model_causal_attribution'],
        'video_seeds': VIDEO, 'action_seeds': ACTION, 'observations': observations,
        'pairs': pairs, 'idm_action_scatter_pairs': action_pairs, 'verification': verification,
        'alpha_baseline_sources': baseline_indexes,
        'interventions': {
            'Q5': 'donor video_latents_in at every denoising step; receiver first frame, prompt, proprio, initial action noise and schedule; action state evolves',
            'Q6': 'IDM own stage1 future or same-observation donor future; replace future temporal positions after stage1 BEFORE vb.prepare/KV prefill; receiver first frame fixed; fresh KV per condition',
            'GT': '9 assembled frames from receiver offset0,4,...,32; actual IDM VAE preprocess/scaling; same receiver first-frame latent forced',
            'identity': 'all16 receivers per observation for Q5 and Q6; same-source overwrite; compare all32 actions and step tensors against measured repeatability',
            'stage1_reuse': 'allowed only after independently verified identical per-step video tensors across action seeds; retain16 stage2 baselines regardless'},
        'gt_eligibility': {
            'predicates': ['complete=true', 'inference_horizon=32', 'executed_count=32',
                           '32 execution rows: offsets1..32/count_delta=1/no exception',
                           'post_offsets exactly4,8,...,32', 'three cameras/source hashes verified',
                           '32x20 finite physical commands'],
            'termination': 'full32 terminal chunk eligible; short chunks ineligible; no padding or stitching',
            'planner': 'report failures separately; do not equate dispatch with realized motion',
            'eligible_explore_obs_ids': eligible,
            'paired_subset': 'GT error and self error on exactly same eligible observations/receiver action seeds; report task/episode coverage'},
        'holdout': {
            'new_episodes_per_task': 3, 'expert_seed_starts': STARTS,
            'expert_selection': 'ascending seed, first3 passing original expert validity check; log all rejected seeds; no policy-outcome/GT selection or episode replacement',
            'source_policy': 'frozen original alpha G0 collection; not G2 intervention',
            'native_step_limits': {'hanging_mug': 900, 'place_empty_cup': 500, 'turn_switch': 400},
            'instruction_type': 'unseen, original manifest assignment',
            'observation_rng': 'Python random.Random(20261017)', 'episode_stage_pairs': STAGES,
            'progress': '(chunk+0.5)/n; early<1/3,middle<2/3,else late',
            'selection_order': ['task', 'episode', 'slot'],
            'fallback': 'nearest unused chunk to stage center; lower index tie break; log',
            'insufficient': 'n<2 or incomplete/noncontiguous inventory => stop, record deviation; never replace episode',
            'release_order': ['joint_T1_freeze', 'explore_K5_K8_V2', 'independent_A3_review',
                              'explore_analysis_code_and_interpretation_freeze', 'C4_GPU3',
                              'Claude_C4_audit', 'joint_holdout_manifest_freeze',
                              'Claude_GPU2_holdout', 'Codex_frozen_analysis', 'Claude_independent_review']},
        'metrics': {
            'physical_layout': '[left_xyz0:3,left_rot6d3:9,left_grip9,right_xyz10:13,right_rot6d13:19,right_grip19]',
            'primary_D_xyz': 'mean of64 3D Euclidean EEF target distances; metres internally, mm in report',
            'Q5_primary': 'D_xyz(A_swap,A_receiver)', 'Q6_primary': 'D_xyz(A_alt,A_self)',
            'GT_primary': 'D_xyz(A_GT,A_G0_commands)-D_xyz(A_self,A_G0_commands); negative=command consistency only',
            'secondary': ['max_xyz', 'rotation_geodesic_mean_max_degrees_and_degenerate_counts',
                          'continuous_gripper_mean_max_abs', 'D_xyz(A_GT,A_self)'],
            'rotation': 'first two columns Gram-Schmidt; normalization floor1e-8, degeneracy1e-6; flag invalid rotations',
            'direction': {'space': 'flatten physical32x6 xyz to192 values', 'd': 'A_donor-A_receiver',
                          'delta': 'A_swap-A_receiver', 'r_d': 'norm(d)/sqrt(64)',
                          'eligibility_r_d_metres': 0.001, 'p': 'dot(delta,d)/dot(d,d)',
                          's_metres': 'dot(delta,d)/norm(d)/sqrt(64)',
                          'r_perp_metres': 'norm(delta-p*d)/sqrt(64)',
                          'cosine': 'null if norm(delta)=0; otherwise dot(delta,d)/(norm(delta)*norm(d))',
                          'ineligible': 'direction null; retain primary amplitude; report coverage',
                          'interpretation': 'donor action direction alignment, not semantic following; 1mm descriptive convention',
                          'Q6': 'same formulas secondary, using own IDM donor/self actions; no cross-model vectors'},
            'background': {'Q5': 'natural D_xyz(A_donor,A_receiver) for same48 pairs',
                           'Q6': 'same48 natural donor pairs plus samevideo/actionseed scatter (24 unordered pairs per obs); background not zero reference'},
            'numerics': 'physical actions -> float64 metrics/aggregation; no normalized20/80 cross-model comparison'},
        'statistics': {'aggregation': 'pair mean per obs/condition, equal obs per episode, equal episode per task/split',
                       'eligible_subset': 'GT/direction denominators and eligible episodes reported; empty episode null, never zero or replacement',
                       'bootstrap_unit': 'whole eligible episode within task/split', 'bootstrap_repetitions': 10000,
                       'bootstrap_seed': 20261018, 'interval': '95% percentile; numpy quantile linear interpolation',
                       'no_p_values': True, 'no_cross_task_pooling': True,
                       'missing': 'missing/nonfinite/duplicate expected row is failure; never silently drop',
                       'interpretation': 'pilot descriptive intervals, only3 new episodes/task; no population or utility claim'},
        'engineering_gates': ['clean owned-code tree; runtime/source/config/checkpoint/normalizer hashes recorded',
                              'separate jointly approved freeze receipt bound to exact prereg SHA',
                              'K6/K7 runtime compatibility independently accepted before Q6',
                              'no-op/fullidentity repeatability; firstframe preservation; new KV; shape/dtype/mask checks',
                              'GT frame layout/preprocess/VAE reconstruction and execution command conversion checked'],
        'runtime_freeze': {'required_before_model_effects': True,
                           'artifacts': ['model checkpoint file manifest/SHA', 'resolved model/deploy config/SHA',
                                         'normalizer/SHA', 'preprocess/VAE parameters', 'video/action schedules',
                                         'attention mask and active dimensions', 'source and runtime commits'],
                           'approval': 'both parties bind runtime manifest SHA before respective Q5/Q6 computations; cannot silently alter schedules to improve results',
                           'IDM_status': 'checkpoint still downloading; runtime compatibility pending; protocol freeze alone does not release Q6'},
        'limitations': ['future latent dependence is not semantics', 'Q5 donor history contains action feedback',
                        'IDM training and architecture differ from alpha; no single-factor comparison',
                        'GT is an after-the-fact oracle containing original action information',
                        'offsets are command counts, not verified physical/training frame time',
                        'G2 explore sources already inspected for G1; only new holdout is fresh'],
        'expected_counts': {'explore_observations': 30, 'explore_episodes': 15,
                            'pairs_per_observation': 48, 'explore_Q5_swaps': 1440,
                            'explore_Q5_identity': 480, 'explore_IDM_stage2_baselines': 480,
                            'explore_IDM_alt': 1440, 'explore_IDM_GT': 16*len(eligible),
                            'explore_IDM_identity': 480, 'new_holdout_observations': 18,
                            'new_holdout_episodes': 9, 'holdout_GT': '16 times GT-eligible selected obs; 0..288'}}


def gated_holdout(args):
    protocol = read(args.prereg)
    if protocol.get('format') != 'openwam-g2-prereg/1': raise ValueError('not G2 prereg')
    freeze, explore = read(args.freeze), read(args.explore_freeze)
    if freeze.get('format') != 'openwam-g2-freeze-receipt/1' or explore.get('format') != 'openwam-g2-explore-freeze-receipt/1':
        raise ValueError('incorrect receipt formats')
    for receipt in (freeze, explore):
        if receipt.get('status') != 'frozen' or receipt.get('prereg_sha256') != sha(args.prereg):
            raise ValueError('missing/mismatched G2 freeze receipt')
        if {a.get('by') for a in receipt.get('approvals', [])} != {'codex', 'claude'}:
            raise ValueError('joint approvals required')
        for approval in receipt['approvals']:
            if sha(approval['path']) != approval['sha256']:
                raise ValueError('approval artifact hash mismatch')
    artifacts = explore.get('artifacts', [])
    if not {'analysis_script', 'interpretation'}.issubset({a.get('role') for a in artifacts}):
        raise ValueError('exploration analysis and interpretation hashes required')
    for item in artifacts:
        if sha(item['path']) != item['sha256']: raise ValueError('explore artifact hash mismatch')
    if sha(__file__) != protocol['provenance']['sampler']['sha256']:
        raise ValueError('sampler changed since registration')
    rows = []
    for path in sorted(args.capture_root.glob('*/ep*/chunk*/record.json')):
        d = read(path)
        rows.append({'obs_id': f"{d['task']}_ep{d['episode']:04d}_chunk{d['chunk']:04d}",
                     **{k: d[k] for k in ('task', 'episode', 'chunk')}, 'env_seed': d['env_seed'],
                     'trace': d['probe_receipt']['trace_path'], 'trace_sha256': d['probe_receipt']['sha256'],
                     'record': str(path.resolve()), 'record_sha256': sha(path),
                     'source_termination': d['termination']})
    for task in TASKS:
        seeds = []
        for ep in range(3):
            group = [r for r in rows if (r['task'], r['episode']) == (task, ep)]
            values = {r['env_seed'] for r in group}
            if len(values) != 1: raise ValueError('missing/inconsistent episode seed')
            seeds.append(next(iter(values)))
            ordered = sorted(group, key=lambda r: r['chunk'])
            if ordered[-1]['source_termination'] not in ('success', 'step_lim') or any(
                    r['source_termination'] != 'none' for r in ordered[:-1]):
                raise ValueError('incomplete source episode or chunks after termination')
        if seeds != sorted(set(seeds)) or seeds[0] < STARTS[task]:
            raise ValueError('reused/nonascending holdout seed')
    selected = select_holdout(rows)
    audited = []
    for row in selected:
        row.update(n_chunks=row['n_chunks'], episode_outcome=next(
            r['source_termination'] for r in rows if (r['task'], r['episode'], r['chunk']) ==
            (row['task'], row['episode'], row['n_chunks']-1)))
        checked = source_observation(row, g1_source=False)
        checked.update(split='holdout', env_seed=row['env_seed'])
        audited.append(checked)
    return {'format': 'openwam-g2-holdout-selection/1', 'approval': 'joint manifest freeze required before running',
            'prereg_sha256': sha(args.prereg), 'freeze_sha256': sha(args.freeze),
            'explore_freeze_sha256': sha(args.explore_freeze), 'observations': audited,
            'capture_inventory': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    reg = sub.add_parser('register')
    for name in ('g1-prereg', 'tasks', 'draft', 'peer-reply', 'out'):
        reg.add_argument('--'+name, type=Path, required=True)
    hold = sub.add_parser('select-holdout')
    for name in ('prereg', 'freeze', 'explore-freeze', 'capture-root', 'out'):
        hold.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    value = build(args.g1_prereg, args.tasks, args.draft, args.peer_reply) if args.command == 'register' else gated_holdout(args)
    atomic_json(args.out, value)
    print(json.dumps({'path': str(args.out), 'sha256': sha(args.out),
                      'observations': len(value['observations']),
                      'gt_eligible': sum(o['gt']['eligible'] for o in value['observations'])}))


if __name__ == '__main__':
    main()
