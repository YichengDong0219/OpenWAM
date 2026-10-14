"""Synthetic regression for rating_compare_run.py (R21 review items). No real ratings are used.

Pre-parse refusals are proven by the absence of the journal, which the driver writes
immediately before the first rating parse.
"""
import hashlib
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rating_compare_run as M  # noqa: E402

CODE = M.CODE_FILES
K = [f'K{i:03d}' for i in range(1, 61)]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def row(k, rng, **kw):
    r = {'clip_id': k, 'active_arm': rng.choice(['left', 'right', 'both', 'uncertain']),
         'legibility': rng.choice(['clear', 'clear', 'partial', 'illegible']), 'uncertain_reason': 'x', 'notes': ''}
    for f in M.EVENTS:
        r[f] = rng.choice(['none', 'uncertain', '2-3', '3-4', '5-6', '0-8'])
    r.update(kw)
    return r


class Fixture:
    def __init__(self, d, n_ai2=60, mapping_extra=None):
        rng = random.Random(5)
        self.d = Path(d)
        root = self.root = self.d/'pack'
        for sub in ('private', 'sealed_ai', 'sealed_human'):
            (root/sub).mkdir(parents=True)
        obs = [(f'obs{i:02d}', ['taskA', 'taskB', 'taskC'][i//10], (i % 10)//2) for i in range(30)]
        items = [(o, r) for o in obs for r in ('receiver', 'donor')]
        rng.shuffle(items)
        clips = [{'pack': 'C', 'clip_id': K[n], 'order': n, 'obs_id': o[0], 'task': o[1], 'episode': o[2], 'role': r}
                 for n, (o, r) in enumerate(items)]
        clips += mapping_extra or []
        (root/'private/mapping.json').write_text(json.dumps({'clips': clips}))
        self.write('sealed_human/rater_human_C.jsonl', [dict(row(k, rng), doc_id='C-'+k, pack='C') for k in K])
        self.write('sealed_ai/rater_ai1.jsonl', [row(k, rng) for k in K])
        self.write('sealed_ai/rater_ai2.jsonl', [row(k, rng) for k in K[:n_ai2]])
        self.seal = {'human': {'C': {'sha256': sha(root/'sealed_human/rater_human_C.jsonl')}},
                     'ai': {'rater_ai1': {'sha256': sha(root/'sealed_ai/rater_ai1.jsonl')},
                            'rater_ai2': {'sha256': sha(root/'sealed_ai/rater_ai2.jsonl')}}}
        (self.d/'seal.json').write_text(json.dumps(self.seal))
        (self.d/'rules.md').write_text('rules')
        (self.d/'cand.json').write_text('{}')
        self.receipt = {'format': 'openwam-human-ai-comparison-freeze/1', 'status': 'jointly_frozen',
                        'signatures': {'claude': 'ok', 'codex': 'ok'},
                        'code_sha256': {n: sha(HERE/n) for n in CODE},
                        'rules_doc_sha256': sha(self.d/'rules.md'), 'candidate_json_sha256': sha(self.d/'cand.json'),
                        'seal_manifest_sha256': sha(self.d/'seal.json'), 'mapping_sha256': sha(root/'private/mapping.json'),
                        'sealed_inputs': {'human': self.seal['human']['C']['sha256'],
                                          'ai1': self.seal['ai']['rater_ai1']['sha256'],
                                          'ai2': self.seal['ai']['rater_ai2']['sha256']}}
        self.save_receipt()

    def write(self, rel, rows):
        (self.root/rel).write_text(''.join(json.dumps(r)+'\n' for r in rows))

    def save_receipt(self, **over):
        r = json.loads(json.dumps(self.receipt))
        r.update(over)
        (self.d/'receipt.json').write_text(json.dumps(r))

    def run(self, out='out/res.json'):
        p = subprocess.run([sys.executable, str(HERE/'rating_compare_run.py'), '--pack-root', str(self.root),
                            '--freeze-receipt', str(self.d/'receipt.json'), '--seal-manifest', str(self.d/'seal.json'),
                            '--rules-doc', str(self.d/'rules.md'), '--candidate', str(self.d/'cand.json'),
                            '--out', str(self.d/out)], capture_output=True, text=True)
        journal = (self.d/out).with_suffix('.journal.json')
        return p.returncode, (p.stderr.strip().splitlines() or [''])[-1], journal.exists()


class DriverChecks(unittest.TestCase):
    def refused_preparse(self, fx, msg):
        code, err, journal = fx.run()
        self.assertNotEqual(code, 0)
        self.assertIn(msg, err)
        self.assertFalse(journal, 'journal written => a rating parse was attempted')

    def test_ok_and_denominators(self):
        with tempfile.TemporaryDirectory() as d:
            fx = Fixture(d, n_ai2=59)
            code, err, journal = fx.run()
            self.assertEqual(code, 0, err)
            res = json.loads((fx.d/'out/res.json').read_text())
            j = json.loads((fx.d/'out/res.journal.json').read_text())
            self.assertEqual(j['status'], 'finished')
            self.assertTrue(j['label_read_started_at'] <= j['analysis_finished_at'])
            self.assertEqual(j['result_sha256'], sha(fx.d/'out/res.json'))
            for cmp in res['comparisons'].values():
                for mode in cmp.values():
                    for tag, x in mode.items():
                        n_clip, n_pair = (59, 29) if tag == 'exclude_K001' else (60, 30)
                        for f, c in x['clip_level'].items():
                            self.assertEqual(c['denominator'], n_clip)
                            pooled = [t['pooled_counts'] for t in x['clip_level_by_task'][f].values()]
                            self.assertEqual(sum(p['denominator'] for p in pooled), n_clip)
                            self.assertEqual(sum(p['agreement'] for p in pooled), c['agreement'])
                        self.assertEqual(x['pair_level']['counts']['denominator'], n_pair)
                        pooled = [t['pooled_counts'] for t in x['pair_level_by_task'].values()]
                        self.assertEqual(sum(p['denominator'] for p in pooled), n_pair)
            # item 3: missing whole row is counted in every distribution, kinds sum to 60
            s = res['rater_shape']['ai2']
            self.assertEqual((s['n_rows'], s['n_missing_rows']), (59, 1))
            self.assertEqual(s['legibility']['missing_row'], 1)
            self.assertEqual(s['active_arm']['missing_row'], 1)
            for f in M.EVENTS:
                self.assertEqual(sum(s[f]['kinds'].values()), 60)
                self.assertEqual(s[f]['kinds']['missing_row'], 1)
                self.assertEqual(sum(s[f]['interval_length_b_minus_a'].values()), s[f]['n_intervals'])
                self.assertEqual(s[f]['n_intervals'], s[f]['kinds'].get('interval', 0))
            # output existence is refused before parsing: the existing journal is untouched
            before = (fx.d/'out/res.journal.json').read_bytes()
            code, err, _ = fx.run()
            self.assertNotEqual(code, 0)
            self.assertIn('refusing to overwrite', err)
            self.assertEqual((fx.d/'out/res.journal.json').read_bytes(), before)

    def test_item1_seal_and_mapping_bound(self):
        with tempfile.TemporaryDirectory() as d:
            fx = Fixture(d)
            fx.save_receipt(seal_manifest_sha256='0'*64)
            self.refused_preparse(fx, 'seal manifest differs')
        with tempfile.TemporaryDirectory() as d:
            fx = Fixture(d)
            fx.save_receipt(mapping_sha256='0'*64)
            self.refused_preparse(fx, 'private mapping differs')
        with tempfile.TemporaryDirectory() as d:
            fx = Fixture(d)
            r = dict(fx.receipt['sealed_inputs'], ai1='0'*64)
            fx.save_receipt(sealed_inputs=r)
            self.refused_preparse(fx, 'disagree for ai1')
        with tempfile.TemporaryDirectory() as d:
            fx = Fixture(d)
            p = fx.root/'sealed_ai/rater_ai2.jsonl'
            p.write_text(p.read_text() + p.read_text().splitlines()[0] + '\n')
            self.refused_preparse(fx, 'sealed file changed: ai2')
        with tempfile.TemporaryDirectory() as d:
            fx = Fixture(d)
            (fx.d/'rules.md').write_text('edited')
            self.refused_preparse(fx, 'rules_doc_sha256 mismatch')

    def test_item2_code_hashes_exact(self):
        for code in ({}, {n: sha(HERE/n) for n in CODE[:3]},
                     {**{n: sha(HERE/n) for n in CODE}, '../x.py': '0'*64},
                     {**{n: sha(HERE/n) for n in CODE}, 'rating_rules.py': 'abc'}):
            with tempfile.TemporaryDirectory() as d:
                fx = Fixture(d)
                fx.save_receipt(code_sha256=code)
                self.refused_preparse(fx, 'exactly the four code files')
        with tempfile.TemporaryDirectory() as d:
            fx = Fixture(d)
            fx.save_receipt(code_sha256={**fx.receipt['code_sha256'], 'rating_rules.py': '1'*64})
            self.refused_preparse(fx, 'code differs')

    def test_status_and_signatures(self):
        with tempfile.TemporaryDirectory() as d:
            fx = Fixture(d)
            fx.save_receipt(status='claude_accepted_pending_codex_driver_review')
            self.refused_preparse(fx, 'not jointly frozen')
        with tempfile.TemporaryDirectory() as d:
            fx = Fixture(d)
            fx.save_receipt(signatures={'claude': 'ok', 'codex': None})
            self.refused_preparse(fx, 'both signatures')

    def test_item4_mapping_strict(self):
        extra = [{'pack': 'C', 'clip_id': 'K001', 'order': 60, 'obs_id': 'obs99', 'task': 'other', 'episode': 99,
                  'role': 'extra'}]
        with tempfile.TemporaryDirectory() as d:
            fx = Fixture(d, mapping_extra=extra)
            code, err, journal = fx.run()
            self.assertNotEqual(code, 0)
            self.assertIn('exactly 60 clips', err)
            self.assertFalse(journal)
        base = None
        with tempfile.TemporaryDirectory() as d:
            base = json.loads((Fixture(d).root/'private/mapping.json').read_text())['clips']
        bad = {'role': lambda c: c[0].update(role='extra'),
               'dup_id': lambda c: c[1].update(clip_id=c[0]['clip_id']),
               'pack': lambda c: c[0].update(pack='A'),
               'cross_episode': lambda c: next(x for x in c if x['obs_id'] == c[0]['obs_id'] and x is not c[0]).update(episode=9),
               'episode_type': lambda c: c[0].update(episode='0')}
        for name, mutate in bad.items():
            clips = json.loads(json.dumps(base))
            mutate(clips)
            with tempfile.TemporaryDirectory() as d:
                p = Path(d)/'m.json'
                p.write_text(json.dumps({'clips': clips}))
                with self.assertRaises(ValueError, msg=name):
                    M.load_pairs(p)


if __name__ == '__main__':
    unittest.main()
