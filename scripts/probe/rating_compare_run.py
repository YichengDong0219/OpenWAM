"""Human/AI agreement driver for pack C (HUMAN_AI_COMPARISON_V1.md).

Loads sealed rater files, the private pair mapping and the frozen comparison
receipt, then reports the three fixed comparisons with rating_compare.py and
rating_rules_v2.py. Adds only what the rules require beyond those pure
functions: strict loading, episode->task equal-weight description and per-rater
label-shape descriptives. Refuses to overwrite its output.
"""
import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from rating_compare import clip_comparison, clip_summary, counts, pair_summary
from rating_rules import FIELDS, arm, event, validate
from rating_rules_v2 import summarize

CLIPS = [f'K{i:03d}' for i in range(1, 61)]
EVENTS = FIELDS[1:]
COMPARISONS = (('human', 'ai1'), ('human', 'ai2'), ('ai1', 'ai2'))
MODES = (('clear_primary', False), ('partial_sensitivity', True))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rater(path, expected_sha):
    if sha(path) != expected_sha:
        raise ValueError(f'sealed file changed: {path}')
    rows = {}
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            raise ValueError(f'blank line {n}: {path}')
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f'non-object line {n}: {path}')
        cid = str(row.get('clip_id', ''))
        cid = cid[2:] if cid.startswith('C-') else cid
        if 'doc_id' in row and row['doc_id'] != 'C-'+cid:
            raise ValueError(f'doc_id/clip_id mismatch line {n}: {path}')
        if 'pack' in row and row['pack'] != 'C':
            raise ValueError(f'non-C pack line {n}: {path}')
        if cid not in CLIPS:
            raise ValueError(f'unknown clip identity line {n}: {path}')
        if cid in rows:
            raise ValueError(f'duplicate clip identity {cid}: {path}')
        if validate(row):
            raise ValueError(f'invalid schema {cid}: {validate(row)}')
        rows[cid] = row
    return rows


def load_pairs(mapping_path):
    clips = json.loads(Path(mapping_path).read_text())['clips']
    by_obs = defaultdict(dict)
    for c in clips:
        if c['pack'] != 'C' or c['clip_id'] not in CLIPS or c['role'] in by_obs[c['obs_id']]:
            raise ValueError('mapping identity')
        by_obs[c['obs_id']][c['role']] = c
    pairs = []
    for obs, roles in sorted(by_obs.items()):
        r, d = roles['receiver'], roles['donor']
        if (r['task'], r['episode']) != (d['task'], d['episode']):
            raise ValueError('pair crosses episode: '+obs)
        pairs.append({'pair_id': obs, 'receiver': ('C', r['clip_id']), 'donor': ('C', d['clip_id']),
                      'task': r['task'], 'episode': r['episode']})
    if len(pairs) != 30 or sorted(k for p in pairs for k in (p['receiver'][1], p['donor'][1])) != CLIPS:
        raise ValueError('expected 30 pairs covering K001..K060')
    clip_unit = {c['clip_id']: (c['task'], c['episode']) for c in clips}
    return pairs, clip_unit


def hierarchy(statuses, units):
    """statuses/units: parallel lists. clip|pair -> episode (counts) -> task (equal-weight over episodes)."""
    eps = defaultdict(list)
    for s, u in zip(statuses, units):
        eps[u].append(s)
    tasks = defaultdict(dict)
    for (task, ep), st in sorted(eps.items()):
        tasks[task][str(ep)] = counts(st)
    out = {}
    for task, by_ep in sorted(tasks.items()):
        e = list(by_ep.values())
        dec = [x['agreement_decidable'] for x in e if x['agreement_decidable'] is not None]
        out[task] = {'episodes': by_ep, 'n_episodes': len(e),
                     'eq_weight_agreement_all': sum(x['agreement_all'] for x in e)/len(e),
                     'eq_weight_coverage': sum(x['coverage'] for x in e)/len(e),
                     'eq_weight_agreement_decidable': sum(dec)/len(dec) if dec else None,
                     'episodes_with_decidable': len(dec),
                     'pooled_counts': counts([s for st in (eps[(task, int(k))] for k in by_ep) for s in st])}
    return out


def rater_shape(rows):
    out = {'n_rows': len(rows), 'legibility': dict(Counter(r['legibility'] for r in rows.values())),
           'active_arm': dict(Counter(arm(r.get('active_arm'))[1] or arm(r.get('active_arm'))[0]
                                      for r in rows.values()))}
    for f in EVENTS:
        kinds, lengths = Counter(), Counter()
        for r in rows.values():
            status, v = event(r.get(f))
            kind = status if status != 'known' else ('none' if v is None else 'interval')
            kinds[kind] += 1
            if kind == 'interval':
                lengths[v[1]-v[0]] += 1
        out[f] = {'kinds': dict(kinds), 'interval_length_b_minus_a': {str(k): lengths[k] for k in sorted(lengths)}}
    return out


def analyse(raters, pairs, clip_unit):
    k001_pair = [p['pair_id'] for p in pairs if ('C', 'K001') in (p['receiver'], p['donor'])]
    if len(k001_pair) != 1:
        raise ValueError('K001 pair')
    keyed = {name: {('C', k): v for k, v in rows.items()} for name, rows in raters.items()}
    out = {'rater_shape': {n: rater_shape(r) for n, r in raters.items()}, 'pair_decisions': {}, 'comparisons': {}}
    for mode, partial in MODES:
        dec = {}
        for name in raters:
            s = summarize(pairs, keyed[name], include_partial=partial)
            dec[name] = {d['pair_id']: d['decision'] for d in s['pairs']}
            out['pair_decisions'].setdefault(mode, {})[name] = {'per_pair': dec[name], 'main': s['main'],
                                                                 'exclude_potential_unblinding': s['exclude_potential_unblinding']}
        for a, b in COMPARISONS:
            key = f'{a}-{b}'
            res = out['comparisons'].setdefault(key, {}).setdefault(mode, {})
            for excl in (False, True):
                tag = 'exclude_K001' if excl else 'all'
                clips = [k for k in CLIPS if not (excl and k == 'K001')]
                per_clip = {k: clip_comparison(raters[a].get(k), raters[b].get(k), include_partial=partial)['fields']
                            for k in clips}
                ps_keys = [p for p in pairs if not (excl and p['pair_id'] in k001_pair)]
                pair_status = []
                for p in ps_keys:
                    x, y = dec[a][p['pair_id']], dec[b][p['pair_id']]
                    pair_status.append('indeterminate' if 'indeterminate' in (x, y) else 'agreement' if x == y else 'disagreement')
                res[tag] = {
                    'clip_level': clip_summary(raters[a], raters[b], include_partial=partial, exclude_k001=excl),
                    'clip_level_by_task': {f: hierarchy([per_clip[k][f] for k in clips], [clip_unit[k] for k in clips])
                                           for f in FIELDS},
                    'pair_level': pair_summary(dec[a], dec[b], excluded_pair_ids=k001_pair if excl else ()),
                    'pair_level_by_task': hierarchy(pair_status, [(p['task'], p['episode']) for p in ps_keys]),
                    'per_clip_status': per_clip,
                }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack-root', required=True)
    ap.add_argument('--freeze-receipt', required=True)
    ap.add_argument('--seal-manifest', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    root = Path(args.pack_root)
    receipt = json.loads(Path(args.freeze_receipt).read_text())
    if receipt.get('status') != 'jointly_frozen':
        raise SystemExit('comparison rules not jointly frozen')
    here = Path(__file__).resolve().parent
    for name, digest in receipt['code_sha256'].items():
        if sha(here/name) != digest:
            raise SystemExit(f'code differs from frozen receipt: {name}')
    seal = json.loads(Path(args.seal_manifest).read_text())
    files = {'human': (root/'sealed_human/rater_human_C.jsonl', seal['human']['C']['sha256']),
             'ai1': (root/'sealed_ai/rater_ai1.jsonl', seal['ai']['rater_ai1']['sha256']),
             'ai2': (root/'sealed_ai/rater_ai2.jsonl', seal['ai']['rater_ai2']['sha256'])}
    raters = {n: load_rater(p, h) for n, (p, h) in files.items()}
    pairs, clip_unit = load_pairs(root/'private/mapping.json')
    result = analyse(raters, pairs, clip_unit)
    result['provenance'] = {'freeze_receipt_sha256': sha(args.freeze_receipt), 'seal_manifest_sha256': sha(args.seal_manifest),
                            'inputs': {n: {'path': str(p), 'sha256': h} for n, (p, h) in files.items()},
                            'mapping_sha256': sha(root/'private/mapping.json'),
                            'code_sha256': {n: sha(here/n) for n in ('rating_compare_run.py', 'rating_compare.py',
                                                                      'rating_rules.py', 'rating_rules_v2.py')}}
    out = Path(args.out)
    if out.exists():
        raise SystemExit('refusing to overwrite '+str(out))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp')
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=1, sort_keys=True))
    os.replace(tmp, out)
    print(json.dumps({'written': str(out), 'sha256': sha(out)}))


if __name__ == '__main__':
    main()
