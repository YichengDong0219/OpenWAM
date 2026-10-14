"""Result-blind holdout input adapter; imports only jointly frozen A3 metric/hierarchy functions.
No new metric, weight, seed, threshold or scientific group. Never uses exploration results as holdout.
"""
import argparse, collections, datetime, hashlib, json, os
from pathlib import Path
import numpy as np
import torch
import g2_analysis_codex as F
PREREG = 'be91026ca9bfb0ae8a72f1c7390ea8486952371e4661d2ecc4e82b4a0a44de39'
ANALYSIS = '39260bae0d18987e359c6f4c6c42c4c426455f10a588e28ebf19dd53bb44428b'

def joint(path, fmt):
    d = json.loads(Path(path).read_text())
    if not (d['format'] == fmt and d['status'] == 'frozen' and (d['prereg_sha256'] == PREREG)):
        raise ValueError("Holdout validation failed: d['format'] == fmt and d['status'] == 'frozen' and (d['prereg_sha256'] == PREREG)")
    if not {a['by'] for a in d['approvals']} == {'codex', 'claude'}:
        raise ValueError("Holdout validation failed: {a['by'] for a in d['approvals']} == {'codex', 'claude'}")
    for a in d['approvals']:
        if not F.sha(a['path']) == a['sha256']:
            raise ValueError("Holdout validation failed: F.sha(a['path']) == a['sha256']")
    return d

def load_inputs(prepath, freeze, explore, receipt):
    if not (F.sha(prepath) == PREREG and F.sha(F.__file__) == ANALYSIS):
        raise ValueError('Holdout validation failed: F.sha(prepath) == PREREG and F.sha(F.__file__) == ANALYSIS')
    pre = json.loads(Path(prepath).read_text())
    joint(freeze, 'openwam-g2-freeze-receipt/1')
    ex = joint(explore, 'openwam-g2-explore-freeze-receipt/1')
    if not {'analysis_script', 'interpretation'} <= {a['role'] for a in ex['artifacts']}:
        raise ValueError("Holdout validation failed: {'analysis_script', 'interpretation'} <= {a['role'] for a in ex['artifacts']}")
    for a in ex['artifacts']:
        if not F.sha(a['path']) == a['sha256']:
            raise ValueError("Holdout validation failed: F.sha(a['path']) == a['sha256']")
    hm = joint(receipt, 'openwam-g2-holdout-manifest-receipt/1')
    sr = hm['selection']
    if not F.sha(sr['path']) == sr['sha256']:
        raise ValueError("Holdout validation failed: F.sha(sr['path']) == sr['sha256']")
    sel = json.loads(Path(sr['path']).read_text())
    if not (sel['format'] == 'openwam-g2-holdout-selection/1' and sel['prereg_sha256'] == PREREG and (sel['freeze_sha256'] == F.sha(freeze)) and (sel['explore_freeze_sha256'] == F.sha(explore))):
        raise ValueError("Holdout validation failed: sel['format'] == 'openwam-g2-holdout-selection/1' and sel['prereg_sha256'] == PREREG and (sel['freeze_sha256'] == F.sha(freeze)) and (sel['explore_freeze_sha256'] == F.sha(explore))")
    obs = sel['observations']
    if not hm['n_observations'] == len(obs) == pre['expected_counts']['new_holdout_observations']:
        raise ValueError("Holdout validation failed: hm['n_observations'] == len(obs) == pre['expected_counts']['new_holdout_observations']")
    if not (len({o['obs_id'] for o in obs}) == len(obs) and all((o['split'] == 'holdout' for o in obs))):
        raise ValueError("Holdout validation failed: len({o['obs_id'] for o in obs}) == len(obs) and all((o['split'] == 'holdout' for o in obs))")
    if not len({(o['task'], o['episode']) for o in obs}) == pre['expected_counts']['new_holdout_episodes']:
        raise ValueError("Holdout validation failed: len({(o['task'], o['episode']) for o in obs}) == pre['expected_counts']['new_holdout_episodes']")
    if not collections.Counter((o['task'] for o in obs)) == {t: 6 for t in pre['holdout']['expert_seed_starts']}:
        raise ValueError("Holdout validation failed: collections.Counter((o['task'] for o in obs)) == {t: 6 for t in pre['holdout']['expert_seed_starts']}")
    if not all((n == 2 for n in collections.Counter(((o['task'], o['episode']) for o in obs)).values())):
        raise ValueError("Holdout validation failed: all((n == 2 for n in collections.Counter(((o['task'], o['episode']) for o in obs)).values()))")
    refs = {'explore_freeze': {'path': str(Path(explore).resolve()), 'sha256': F.sha(explore)}, 'holdout_manifest': {'path': str(Path(receipt).resolve()), 'sha256': F.sha(receipt)}, 'holdout_selection': sr}
    return (pre, obs, refs)

def keys(scope, obs, seeds, pairs):
    expected = set()
    for o in obs:
        oid = o['obs_id']
        if scope == 'Q6':
            expected.add('source|' + oid)
        if scope == 'Q6' and o['gt']['eligible']:
            expected.add('gt_future|' + oid)
        for (v, a) in seeds:
            expected |= {f'baseline|{oid}|{v}-{a}', f'identity|{oid}|{v}-{a}'}
            if scope == 'Q6' and o['gt']['eligible']:
                expected.add(f'gt|{oid}|{v}-{a}')
        for p in pairs:
            expected.add(f"{('swap' if scope == 'Q5' else 'alt')}|{oid}|{p['receiver'][0]}-{p['receiver'][1]}|{p['donor'][0]}-{p['donor'][1]}")
    return expected

def main():
    ap = argparse.ArgumentParser()
    for name in ['prereg', 'freeze', 'explore-freeze', 'holdout-manifest', 'q5', 'q6', 'q5-binding', 'q6-binding', 'out']:
        ap.add_argument('--' + name, required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if not not out.exists():
        raise ValueError('Holdout validation failed: not out.exists()')
    (pre, obs, refs) = load_inputs(args.prereg, args.freeze, args.explore_freeze, args.holdout_manifest)
    seeds = [(v, a) for v in pre['video_seeds'] for a in pre['action_seeds']]
    indexes = {}
    bindings = {}
    audit = {}
    for (scope, rootpath, bpath) in [('Q5', args.q5, args.q5_binding), ('Q6', args.q6, args.q6_binding)]:
        root = Path(rootpath).resolve()
        if not root.is_relative_to(Path('/home/dongyicheng/OpenWAM/outputs/probe')):
            raise ValueError("Holdout validation failed: root.is_relative_to(Path('/home/dongyicheng/OpenWAM/outputs/probe'))")
        bind = json.loads(Path(bpath).read_text())
        if not (bind['format'] == 'openwam-runtime-binding/2' and bind['reviews']):
            raise ValueError("Holdout validation failed: bind['format'] == 'openwam-runtime-binding/2' and bind['reviews']")
        if not (bind['status'] == 'jointly_bound' and bind['prereg_sha256'] == PREREG and all(bind['signatures'].values())):
            raise ValueError("Holdout validation failed: bind['status'] == 'jointly_bound' and bind['prereg_sha256'] == PREREG and all(bind['signatures'].values())")
        if not bind['scope'] == ('Q5_alpha' if scope == 'Q5' else 'Q6_idm'):
            raise ValueError("Holdout validation failed: bind['scope'] == ('Q5_alpha' if scope == 'Q5' else 'Q6_idm')")
        if not F.sha(bind['manifest_path']) == bind['manifest_sha256']:
            raise ValueError("Holdout validation failed: F.sha(bind['manifest_path']) == bind['manifest_sha256']")
        for r in bind['reviews']:
            if not F.sha(r['path']) == r['sha256']:
                raise ValueError("Holdout validation failed: F.sha(r['path']) == r['sha256']")
        bindings[scope] = F.sha(bpath)
        stamp = json.loads((root / 'RUN_STAMP.json').read_text())
        if not stamp == {'prereg_sha256': PREREG, 'mode': 'run', 'runtime_binding_sha256': bindings[scope], 'split': 'holdout', 'holdout_refs': refs}:
            raise ValueError("Holdout validation failed: stamp == {'prereg_sha256': PREREG, 'mode': 'run', 'runtime_binding_sha256': bindings[scope], 'split': 'holdout', 'holdout_refs': refs}")
        rr = [json.loads(l) for l in (root / 'index.jsonl').read_text().splitlines()]
        by = {r['key']: r for r in rr}
        if not (len(by) == len(rr) and set(by) == keys(scope, obs, seeds, pre['pairs'])):
            raise ValueError("Holdout validation failed: len(by) == len(rr) and set(by) == keys(scope, obs, seeds, pre['pairs'])")
        paths = {Path(r['path']).resolve() for r in rr}
        if not (len(paths) == len(rr) and all((p.is_relative_to(root) for p in paths))):
            raise ValueError('Holdout validation failed: len(paths) == len(rr) and all((p.is_relative_to(root) for p in paths))')
        allowed = {root / 'index.jsonl', root / 'RUN_STAMP.json'}
        if scope == 'Q5':
            journal = root / 'source_checks.jsonl'
            if not journal.is_file():
                raise ValueError('Q5 source-check journal missing')
            checks = [json.loads(line) for line in journal.read_text().splitlines()]
            byid = {o['obs_id']: o for o in obs}
            for check in checks:
                oid = check.get('obs_id')
                if oid not in byid or check.get('source_trace_sha256') != byid[oid]['trace_sha256'] or check.get('runtime_binding_sha256') != bindings[scope] or (check.get('result') != 'steps_and_final_bitwise_equal'):
                    raise ValueError('Q5 source-check journal identity/result mismatch')
            if len(checks) < len(obs) or [check['obs_id'] for check in checks[-len(obs):]] != [o['obs_id'] for o in obs]:
                raise ValueError('Q5 latest source-check invocation does not cover all observations in order')
            allowed.add(journal)
        if (root / 'failures.jsonl').exists():
            raise ValueError(scope + ' failure journal present')
        if not {p.resolve() for p in root.rglob('*') if p.is_file()} == paths | allowed:
            raise ValueError("Holdout validation failed: {p.resolve() for p in root.rglob('*') if p.is_file()} == paths | allowed")
        for r in rr:
            if not F.sha(r['path']) == r['sha256']:
                raise ValueError("Holdout validation failed: F.sha(r['path']) == r['sha256']")
        indexes[scope] = by
        audit[scope] = {'n_units': len(rr), 'counts': dict(collections.Counter((r['kind'] for r in rr))), 'index_sha256': F.sha(root / 'index.jsonl'), 'all_unit_sha_match': True, 'exact_keyset': True}
    out.mkdir(parents=True)
    F.write(out / 'START.json', {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'adapter_sha256': F.sha(__file__), 'frozen_metric_code_sha256': ANALYSIS, 'input_refs': refs, 'bindings': bindings, 'prereg_sha256': PREREG})
    rows = []
    cache = {}

    def get(scope, key, o, kind, v=None, a=None, donor=None):
        if (scope, key) not in cache:
            cache[scope, key] = torch.load(indexes[scope][key]['path'], map_location='cpu', weights_only=False)
        tr = cache[scope, key]
        m = tr['meta']
        if kind not in ['source']:
            for (k, value) in {'format': f'openwam-g2-{scope.lower()}/2', 'kind': kind, 'prereg_sha256': PREREG, 'runtime_binding_sha256': bindings[scope], 'obs_id': o['obs_id'], 'task': o['task'], 'episode': o['episode']}.items():
                if not m[k] == value:
                    raise ValueError('Holdout validation failed: m[k] == value')
            if v is not None:
                if not (m['video_seed'] == v and m['action_seed'] == a and (m['donor'] == donor)):
                    raise ValueError("Holdout validation failed: m['video_seed'] == v and m['action_seed'] == a and (m['donor'] == donor)")
        actions = np.asarray(tr['final']['actions_physical'])
        if not (actions.shape == (32, 20) and np.isfinite(actions).all()):
            raise ValueError('Holdout validation failed: actions.shape == (32, 20) and np.isfinite(actions).all()')
        return tr

    def ref(scope, key):
        return {k: indexes[scope][key][k] for k in ['path', 'sha256']}

    def row(group, o, key, m):
        return {'group': group, 'obs_id': o['obs_id'], 'task': o['task'], 'episode': o['episode'], 'stage': o['relative_stage'], 'key': key, 'metrics': m}
    for o in obs:
        cache.clear()
        oid = o['obs_id']
        if not (F.sha(o['record']) == o['record_sha256'] and F.sha(o['trace']) == o['trace_sha256']):
            raise ValueError("Holdout validation failed: F.sha(o['record']) == o['record_sha256'] and F.sha(o['trace']) == o['trace_sha256']")
        rec = json.loads(Path(o['record']).read_text())
        bas = {}
        for scope in ['Q5', 'Q6']:
            for (v, a) in seeds:
                key = f'baseline|{oid}|{v}-{a}'
                br = get(scope, key, o, 'baseline', v, a, None)
                if not br['meta']['source_trace_sha256' if scope == 'Q5' else 'record_sha256'] == o['trace_sha256' if scope == 'Q5' else 'record_sha256']:
                    raise ValueError("Holdout validation failed: br['meta']['source_trace_sha256' if scope == 'Q5' else 'record_sha256'] == o['trace_sha256' if scope == 'Q5' else 'record_sha256']")
                bas[scope, v, a] = br
                ik = f'identity|{oid}|{v}-{a}'
                tr = get(scope, ik, o, 'identity', v, a, [v, a])
                if not tr['meta']['receiver_baseline'] == ref(scope, key):
                    raise ValueError("Holdout validation failed: tr['meta']['receiver_baseline'] == ref(scope, key)")
                sk = ('t_a', 'action_latents_in', 'action_velocity') if scope == 'Q6' else ('t_v', 't_a', 'video_latents_in', 'action_latents_in', 'video_velocity', 'action_velocity')
                if not (F.step_eq(tr['steps'], br['steps'], sk) and F.eq(tr['final']['actions_physical'], br['final']['actions_physical'])):
                    raise ValueError("Holdout validation failed: F.step_eq(tr['steps'], br['steps'], sk) and F.eq(tr['final']['actions_physical'], br['final']['actions_physical'])")
            for pair in pre['pairs']:
                (vr, ar) = pair['receiver']
                (vd, ad) = pair['donor']
                kind = 'swap' if scope == 'Q5' else 'alt'
                key = f'{kind}|{oid}|{vr}-{ar}|{vd}-{ad}'
                tr = get(scope, key, o, kind, vr, ar, [vd, ad])
                if not (tr['meta']['receiver_baseline'] == ref(scope, f'baseline|{oid}|{vr}-{ar}') and tr['meta']['donor_baseline'] == ref(scope, f'baseline|{oid}|{vd}-{ad}')):
                    raise ValueError("Holdout validation failed: tr['meta']['receiver_baseline'] == ref(scope, f'baseline|{oid}|{vr}-{ar}') and tr['meta']['donor_baseline'] == ref(scope, f'baseline|{oid}|{vd}-{ad}')")
                receiver = np.asarray(bas[scope, vr, ar]['final']['actions_physical'], np.float64)
                donor = np.asarray(bas[scope, vd, ad]['final']['actions_physical'], np.float64)
                action = np.asarray(tr['final']['actions_physical'], np.float64)
                rows.extend([row(scope + '_effect', o, key, F.physical(action, receiver)), row(scope + '_natural', o, key, F.physical(donor, receiver))])
                dr = F.direction(action, receiver, donor)
                eligible = dr.pop('eligible')
                rows.append(row(scope + '_direction_coverage', o, key, {'eligible_fraction': float(eligible)}))
                if eligible:
                    rows.append(row(scope + '_direction', o, key, dr))
        for pair in pre['idm_action_scatter_pairs']:
            v = pair['video_seed']
            (a, b) = pair['action_seed_pair']
            rows.append(row('Q6_action_scatter', o, f'{v}-{a}-{b}', F.physical(bas['Q6', v, a]['final']['actions_physical'], bas['Q6', v, b]['final']['actions_physical'])))
        if o['gt']['eligible']:
            commands = np.asarray(rec['actions'], np.float64)
            if not (commands.shape == (32, 20) and np.isfinite(commands).all()):
                raise ValueError('Holdout validation failed: commands.shape == (32, 20) and np.isfinite(commands).all()')
            for (v, a) in seeds:
                key = f'gt|{oid}|{v}-{a}'
                tr = get('Q6', key, o, 'gt', v, a, 'gt')
                if not (tr['meta']['receiver_baseline'] == ref('Q6', f'baseline|{oid}|{v}-{a}') and tr['meta']['gt_future'] == ref('Q6', 'gt_future|' + oid)):
                    raise ValueError("Holdout validation failed: tr['meta']['receiver_baseline'] == ref('Q6', f'baseline|{oid}|{v}-{a}') and tr['meta']['gt_future'] == ref('Q6', 'gt_future|' + oid)")
                ga = tr['final']['actions_physical']
                ba = bas['Q6', v, a]['final']['actions_physical']
                mg = F.physical(ga, commands)
                ms = F.physical(ba, commands)
                m = F.physical(ga, ba)
                m.update(E_gt_m=mg['D_xyz_m'], E_self_m=ms['D_xyz_m'], delta_E_m=mg['D_xyz_m'] - ms['D_xyz_m'])
                rows.append(row('GT_command_consistency', o, key, m))
    counts = collections.Counter((r['group'] for r in rows))
    if not (counts['Q5_effect'] == counts['Q6_effect'] == 48 * len(obs) and counts['Q6_action_scatter'] == 24 * len(obs) and (counts['GT_command_consistency'] == 16 * sum((o['gt']['eligible'] for o in obs)))):
        raise ValueError("Holdout validation failed: counts['Q5_effect'] == counts['Q6_effect'] == 48 * len(obs) and counts['Q6_action_scatter'] == 24 * len(obs) and (counts['GT_command_consistency'] == 16 * sum((o['gt']['eligible'] for o in obs)))")
    groups = {}
    for group in sorted(set(counts) | {'GT_command_consistency'}):
        rr = [r for r in rows if r['group'] == group]
        ms = list(rr[0]['metrics']) if rr else ['D_xyz_m', 'max_xyz_m', 'rotation_mean_deg', 'rotation_max_deg', 'rotation_invalid_A', 'rotation_invalid_B', 'rotation_valid_comparisons', 'gripper_mean_abs', 'gripper_max_abs', 'E_gt_m', 'E_self_m', 'delta_E_m']
        groups[group] = F.hierarchy(rr, ms, obs)
    rp = out / 'rows.jsonl'
    rp.write_text(''.join((json.dumps(r, allow_nan=False) + '\n' for r in rows)))
    F.write(out / 'analysis.json', {'format': 'openwam-codex-g2-A3/1', 'split': 'holdout', 'status': 'submitted_pending_claude_independent_review', 'prereg_sha256': PREREG, 'input_refs': refs, 'bindings': bindings, 'source_audit': audit, 'groups': groups, 'counts': dict(counts), 'rows_sha256': F.sha(rp), 'adapter_sha256': F.sha(__file__), 'frozen_metric_code_sha256': ANALYSIS, 'explore_effects_loaded': False, 'statistics': pre['statistics'], 'bootstrap_rng_detail': 'frozen hierarchy, reset per task/metric; sorted eligible episodes; seed20261018'})
if __name__ == '__main__':
    main()
