"""CPU checks of source availability, deterministic sampling and release guards."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import g2_prereg as g2


def complete_record():
    return {'complete': True, 'inference_horizon': 32, 'executed_count': 32,
            'termination': 'success', 'actions_executed': [
                {'offset': i, 'count_delta': 1, 'exception': None, 'planner': []}
                for i in range(1, 33)], 'post_obs': [{'offset': i} for i in range(4, 33, 4)]}


class ProtocolChecks(unittest.TestCase):
    def test_terminal_full_window_and_short_complete(self):
        self.assertEqual(g2.gt_reasons(complete_record()), [])
        short = complete_record()
        short.update(executed_count=4, actions_executed=short['actions_executed'][:4], post_obs=[{'offset': 4}])
        self.assertIn('executed_count_not_32', g2.gt_reasons(short))
        self.assertIn('post_offsets_not_full_window', g2.gt_reasons(short))

    def test_missing_duplicate_offsets_and_execution_error(self):
        bad = complete_record()
        bad['post_obs'][-1] = {'offset': 28}
        bad['actions_executed'][2]['count_delta'] = 0
        self.assertIn('post_offsets_not_full_window', g2.gt_reasons(bad))
        self.assertIn('execution_alignment_or_exception', g2.gt_reasons(bad))

    def test_sampling_preserves_episodes_and_logs_fallback(self):
        rows = [{'task': t, 'episode': ep, 'chunk': i} for t in g2.TASKS for ep in range(3) for i in range(2)]
        before = copy.deepcopy(rows)
        selected = g2.select_holdout(rows)
        self.assertEqual(len(selected), 18)
        self.assertEqual(rows, before)
        self.assertEqual(selected, g2.select_holdout(list(reversed(rows))))
        self.assertEqual(sum(o['selection_fallback'] for o in selected), 6)
        for task in g2.TASKS:
            for ep in range(3):
                self.assertEqual({o['chunk'] for o in selected if (o['task'], o['episode']) == (task, ep)}, {0, 1})

    def test_incomplete_inventory_is_failure_not_replacement(self):
        rows = [{'task': t, 'episode': ep, 'chunk': i} for t in g2.TASKS for ep in range(3) for i in range(3)]
        for changed in (rows[:-3], rows+[rows[0]], [r for r in rows if not (r['task']==g2.TASKS[0] and r['episode']==0 and r['chunk']==1)]):
            with self.assertRaises(ValueError): g2.select_holdout(changed)

    def test_atomic_protocol_refuses_changed_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'protocol.json'
            g2.atomic_json(p, {'a': 1})
            g2.atomic_json(p, {'a': 1})
            with self.assertRaises(ValueError): g2.atomic_json(p, {'a': 2})
            self.assertEqual(g2.read(p), {'a': 1})
            with self.assertRaises(ValueError): g2.atomic_json(Path(tmp)/'nan.json', {'a': float('nan')})

    def test_source_png_tampering_is_not_availability_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); trace = root/'trace.pt'; trace.write_bytes(b'opaque_trace')
            png = root/'image.png'; png.write_bytes(b'opaque_png')
            d = complete_record()
            d.update(task='hanging_mug', episode=0, chunk=0, actions=[[0.0]*20 for _ in range(32)],
                     probe_receipt={'trace_path': str(trace), 'sha256': g2.sha(trace)})
            for entry in [d, *d['post_obs']]:
                entry['pngs'] = {c: {'path': str(png), 'sha256': g2.sha(png)} for c in ('head', 'left', 'right')}
            rec = root/'record.json'; rec.write_text(json.dumps(d))
            o = {'obs_id': 'hanging_mug_ep0000_chunk0000', 'task': 'hanging_mug', 'episode': 0, 'chunk': 0,
                 'trace': str(trace), 'trace_sha256': g2.sha(trace), 'record': str(rec), 'record_sha256': g2.sha(rec),
                 'selection_slot': 0, 'requested_stage': 'early', 'relative_stage': 'early',
                 'relative_progress': 0.1, 'selection_fallback': False, 'split': 'holdout',
                 'n_chunks': 4, 'episode_outcome': 'success'}
            self.assertTrue(g2.source_observation(o)['gt']['eligible'])
            self.assertIsNone(g2.source_observation(o, g1_source=False)['source_g1_split'])
            png.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'PNG hash'): g2.source_observation(o)

    def test_holdout_rejects_missing_joint_explore_freeze(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); prereg=root/'p.json'; freeze=root/'f.json'; explore=root/'e.json'
            prereg.write_text(json.dumps({'format': 'openwam-g2-prereg/1'}))
            freeze.write_text(json.dumps({'format':'openwam-g2-freeze-receipt/1', 'status':'frozen',
                'prereg_sha256':g2.sha(prereg), 'approvals':[]}))
            explore.write_text(json.dumps({'format':'openwam-g2-explore-freeze-receipt/1'}))
            args=SimpleNamespace(prereg=prereg, freeze=freeze, explore_freeze=explore, capture_root=root/'not_collected')
            with self.assertRaisesRegex(ValueError, 'joint approvals'): g2.gated_holdout(args)


if __name__ == '__main__':
    unittest.main()
