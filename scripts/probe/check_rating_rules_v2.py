"""Synthetic checks only; no human scores are embedded or read."""
import unittest
from rating_rules_v2 import summarize


PAIRS=[{'pair_id':str(i),'receiver':('C',f'K{2*i+1:03d}'),'donor':('C',f'K{2*i+2:03d}')} for i in range(30)]


def row(**kwargs):
    value={'active_arm':'left','grasp_frame_interval':'none','release_frame_interval':'none',
           'handover_frame_interval':'none','legibility':'clear','uncertain_reason':''}
    value.update(kwargs);return value


class DenominatorChecks(unittest.TestCase):
    def test_unrated_stays_in_denominator(self):
        s=summarize(PAIRS,{})
        self.assertEqual(s['main']['pair_denominator'],30)
        self.assertEqual(s['main']['indeterminate'],30)
        self.assertIsNone(s['main']['divergence_decidable'])
        self.assertEqual(s['exclude_potential_unblinding']['pair_denominator'],29)

    def test_complete_compatible_and_unknown(self):
        rows={('C',f'K{i:03d}'):row() for i in range(1,61)}
        self.assertEqual(summarize(PAIRS,rows)['main']['compatible_labels'],30)
        rows[('C','K001')]=row(active_arm='uncertain',uncertain_reason='occlusion')
        s=summarize(PAIRS,rows)
        self.assertEqual(s['main']['indeterminate'],1)
        self.assertEqual(s['exclude_potential_unblinding']['compatible_labels'],29)

    def test_partial_and_unblinding_sensitivity_are_separate(self):
        rows={('C',f'K{i:03d}'):row() for i in range(1,61)}
        rows[('C','K001')]=row(active_arm='right',legibility='partial')
        self.assertEqual(summarize(PAIRS,rows)['main']['indeterminate'],1)
        s=summarize(PAIRS,rows,include_partial=True)
        self.assertEqual(s['main']['visible_label_divergence'],1)
        self.assertEqual(s['exclude_potential_unblinding']['visible_label_divergence'],0)

    def test_practice_rows_and_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):summarize(PAIRS,{('A','C001'):row()})
        duplicate=list(PAIRS);duplicate[-1]=duplicate[0]
        with self.assertRaises(ValueError):summarize(duplicate,{})


if __name__=='__main__':unittest.main()
