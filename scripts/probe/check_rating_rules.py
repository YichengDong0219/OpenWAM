"""Synthetic protocol boundary checks; contains no real annotations."""
import unittest
from rating_rules import classify_pair, event, validate


def clip(**kwargs):
    row = dict(active_arm='left', grasp_frame_interval='none', release_frame_interval='none',
               handover_frame_interval='none', legibility='clear', uncertain_reason='')
    row.update(kwargs)
    return row


class RulesChecks(unittest.TestCase):
    def test_arm_disagreement(self):
        self.assertEqual(classify_pair(clip(), clip(active_arm='right'))['decision'], 'visible_label_divergence')

    def test_event_presence_is_distinct_from_missing(self):
        self.assertEqual(classify_pair(clip(), clip(release_frame_interval='3-4'))['decision'], 'visible_label_divergence')
        self.assertEqual(classify_pair(clip(), clip(release_frame_interval=''))['decision'], 'indeterminate')

    def test_overlapping_and_touching_intervals_are_not_disagreement(self):
        a = clip(grasp_frame_interval='1-3')
        self.assertEqual(classify_pair(a, clip(grasp_frame_interval='3-5'))['decision'], 'compatible_labels')
        self.assertEqual(classify_pair(a, clip(grasp_frame_interval='4-5'))['decision'], 'visible_label_divergence')

    def test_uncertain_never_becomes_none(self):
        a = clip(active_arm='uncertain', uncertain_reason='occlusion')
        self.assertEqual(classify_pair(a, clip())['decision'], 'indeterminate')
        self.assertEqual(classify_pair(a, clip(release_frame_interval='3-4'))['decision'], 'visible_label_divergence')

    def test_illegibility_and_partial_denominators(self):
        a = clip(legibility='illegible', uncertain_reason='blur', active_arm='right')
        self.assertEqual(classify_pair(a, clip())['decision'], 'indeterminate')
        a = clip(legibility='partial', active_arm='right')
        self.assertEqual(classify_pair(a, clip())['decision'], 'indeterminate')
        self.assertEqual(classify_pair(a, clip(), include_partial=True)['decision'], 'visible_label_divergence')

    def test_invalid_interval_and_reason_are_flagged(self):
        self.assertEqual(event('8-2')[0], 'invalid')
        self.assertEqual(event('0-9')[0], 'invalid')
        self.assertEqual(event('4-4'), ('known', (4, 4)))
        self.assertIn('missing_uncertain_reason', validate(clip(active_arm='uncertain')))
        self.assertEqual(classify_pair(clip(), clip(grasp_frame_interval='8-2'))['decision'], 'indeterminate')


if __name__ == '__main__': unittest.main()
