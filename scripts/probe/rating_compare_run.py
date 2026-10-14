"""Human/AI agreement driver for pack C (HUMAN_AI_COMPARISON_V1.md).

Loads sealed rater files, the private pair mapping and the frozen comparison
receipt, then reports the three fixed comparisons with rating_compare.py and
rating_rules_v2.py. Adds only what the rules require beyond those pure
functions: strict loading, episode->task equal-weight description and per-rater
label-shape descriptives.

Every binding is checked BEFORE any rating file is parsed (R21 review): receipt
status/signatures, the exact four code hashes, rules doc/candidate, seal manifest
and its three sealed inputs, the private mapping, raw rating-file hashes and the
absence of the output. The first label parse and the end of the analysis are
timestamped (UTC) in a journal that never carries label values.
"""
import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from rating_compare import clip_comparison, clip_summary, counts, pair_summary
from rating_rules import FIELDS, arm, event, validate
from rating_rules_v2 import summarize

CLIPS = [f'K{i:03d}' for i in range(1, 61)]
EVENTS = FIELDS[1:]
COMPARISONS = (('human', 'ai1'), ('human', 'ai2'), ('ai1', 'ai2'))
MODES = (('clear_primary', False), ('partial_sensitivity', True))
CODE_FILES = ('rating_compare_run.py', 'rating_compare.py', 'rating_rules.py', 'rating_rules_v2.py')
SEALED = {'human': ('sealed_human/rater_human_C.jsonl', ('human', 'C')),
          'ai1': ('sealed_ai/rater_ai1.jsonl', ('ai', 'rater_ai1')),
          'ai2': ('sealed_ai/rater_ai2.jsonl', ('ai', 'rater_ai2'))}
HEX64 = re.compile(r'[0-9a-f]{64}')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def is_hex64(x):
    return isinstance(x, str) and HEX64.fullmatch(x) is not None


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
    if not isinstance(clips, list) or len(clips) != 60:
        raise ValueError('mapping must have exactly 60 clips')
    seen, by_obs = set(), defaultdict(dict)
    for c in clips:
        if c.get('pack') != 'C' or c.get('clip_id') not in CLIPS or c['clip_id'] in seen:
            raise ValueError('mapping clip identity')
        if c.get('role') not in ('receiver', 'donor') or c['role'] in by_obs[c.get('obs_id')]:
            raise ValueError('mapping role')
        if not isinstance(c.get('task'), str) or isinstance(c.get('episode'), bool) or not isinstance(c.get('episode'), int):
            raise ValueError('mapping task/episode type')
        seen.add(c['clip_id'])
        by_obs[c['obs_id']][c['role']] = c
    if len(by_obs) != 30 or any(set(r) != {'receiver', 'donor'} for r in by_obs.values()):
        raise ValueError('mapping must have 30 observations with exactly one receiver and one donor')
    pairs = []
    for obs, roles in sorted(by_obs.items()):
        r, d = roles['receiver'], roles['donor']
        if (r['task'], r['episode']) != (d['task'], d['episode']):
            raise ValueError('pair crosses episode: '+obs)
        pairs.append({'pair_id': obs, 'receiver': ('C', r['clip_id']), 'donor': ('C', d['clip_id']),
                      'task': r['task'], 'episode': r['episode']})
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
    """Distributions over the fixed 60 ids; a whole missing row is counted as 'missing_row'."""
    present = [rows[k] for k in CLIPS if k in rows]
    n_missing = len(CLIPS) - len(present)
    leg = Counter(r['legibility'] for r in present)
    arms = Counter()
    for r in present:
        status, v = arm(r.get('active_arm'))
        arms[v if status == 'known' else status] += 1
    out = {'n_ids': len(CLIPS), 'n_rows': len(present), 'n_missing_rows': n_missing,
           'legibility': {**leg, 'missing_row': n_missing}, 'active_arm': {**arms, 'missing_row': n_missing}}
    for f in EVENTS:
        kinds, lengths = Counter(), Counter()
        for r in present:
            status, v = event(r.get(f))
            kind = status if status != 'known' else ('none' if v is None else 'interval')
            kinds[kind] += 1
            if kind == 'interval':
                lengths[v[1]-v[0]] += 1
        kinds['missing_row'] = n_missing
        if sum(kinds.values()) != len(CLIPS):
            raise AssertionError('event kinds must sum to 60')
        out[f] = {'kinds': dict(kinds), 'n_intervals': sum(lengths.values()),
                  'interval_length_b_minus_a': {str(k): lengths[k] for k in sorted(lengths)}}
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


def preflight(args, code_dir):
    """All checks that need no rating parse. Returns the bound inputs; raises SystemExit on any mismatch."""
    receipt_path = Path(args.freeze_receipt)
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('format') != 'openwam-human-ai-comparison-freeze/1' or receipt.get('status') != 'jointly_frozen':
        raise SystemExit('comparison rules not jointly frozen')
    sig = receipt.get('signatures') or {}
    if not all(isinstance(sig.get(k), str) and sig[k].strip() for k in ('claude', 'codex')):
        raise SystemExit('freeze receipt lacks both signatures')
    code = receipt.get('code_sha256')
    if not isinstance(code, dict) or set(code) != set(CODE_FILES) or not all(is_hex64(v) for v in code.values()):
        raise SystemExit('freeze receipt must bind exactly the four code files with sha256 values')
    for name in CODE_FILES:
        if sha(code_dir/name) != code[name]:
            raise SystemExit(f'code differs from frozen receipt: {name}')
    for arg, key in (('rules_doc', 'rules_doc_sha256'), ('candidate', 'candidate_json_sha256')):
        if not is_hex64(receipt.get(key)) or sha(getattr(args, arg)) != receipt[key]:
            raise SystemExit(f'{key} mismatch')
    for key in ('seal_manifest_sha256', 'mapping_sha256'):
        if not is_hex64(receipt.get(key)):
            raise SystemExit(f'freeze receipt lacks {key}')
    if sha(args.seal_manifest) != receipt['seal_manifest_sha256']:
        raise SystemExit('seal manifest differs from frozen receipt')
    seal = json.loads(Path(args.seal_manifest).read_text())
    bound = receipt.get('sealed_inputs') or {}
    root = Path(args.pack_root)
    files = {}
    for name, (rel, (top, sub)) in SEALED.items():
        h = seal[top][sub]['sha256']
        if not is_hex64(bound.get(name)) or h != bound[name]:
            raise SystemExit(f'seal manifest / receipt disagree for {name}')
        if sha(root/rel) != h:
            raise SystemExit(f'sealed file changed: {name}')
        files[name] = (root/rel, h)
    if set(bound) != set(SEALED):
        raise SystemExit('receipt sealed_inputs must name exactly human, ai1, ai2')
    mapping = root/'private/mapping.json'
    if sha(mapping) != receipt['mapping_sha256']:
        raise SystemExit('private mapping differs from frozen receipt')
    out, journal = Path(args.out), Path(args.out).with_suffix('.journal.json')
    if out.exists() or journal.exists():
        raise SystemExit('refusing to overwrite '+str(out))
    return {'receipt_sha256': sha(receipt_path), 'files': files, 'mapping': mapping, 'out': out, 'journal': journal,
            'code_sha256': dict(code), 'seal_manifest_sha256': receipt['seal_manifest_sha256'],
            'mapping_sha256': receipt['mapping_sha256'], 'rules_doc_sha256': receipt['rules_doc_sha256'],
            'candidate_json_sha256': receipt['candidate_json_sha256']}


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True))
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack-root', required=True)
    ap.add_argument('--freeze-receipt', required=True)
    ap.add_argument('--seal-manifest', required=True)
    ap.add_argument('--rules-doc', required=True)
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    b = preflight(args, Path(__file__).resolve().parent)
    pairs, clip_unit = load_pairs(b['mapping'])  # identity only; hash already bound
    inputs = {n: {'path': str(p), 'sha256': h} for n, (p, h) in b['files'].items()}
    journal = {'label_read_started_at': utc(), 'inputs': inputs, 'freeze_receipt_sha256': b['receipt_sha256'],
               'status': 'started'}
    write_json(b['journal'], journal)
    try:
        raters = {n: load_rater(p, h) for n, (p, h) in b['files'].items()}
        result = analyse(raters, pairs, clip_unit)
    except Exception as e:
        journal.update(status='failed', failed_at=utc(), error_type=type(e).__name__)
        write_json(b['journal'], journal)
        raise SystemExit(f'analysis failed ({type(e).__name__}); see journal')
    journal.update(status='finished', analysis_finished_at=utc())
    result['provenance'] = {'freeze_receipt_sha256': b['receipt_sha256'], 'seal_manifest_sha256': b['seal_manifest_sha256'],
                            'mapping_sha256': b['mapping_sha256'], 'rules_doc_sha256': b['rules_doc_sha256'],
                            'candidate_json_sha256': b['candidate_json_sha256'], 'inputs': inputs,
                            'code_sha256': b['code_sha256'],
                            'label_read_started_at': journal['label_read_started_at'],
                            'analysis_finished_at': journal['analysis_finished_at']}
    write_json(b['out'], result)
    journal['result_sha256'] = sha(b['out'])
    write_json(b['journal'], journal)
    print(json.dumps({'written': str(b['out']), 'sha256': journal['result_sha256']}))


if __name__ == '__main__':
    main()
