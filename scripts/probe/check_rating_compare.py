"""Synthetic examples only; no human or AI annotations used."""
import unittest
from rating_compare import clip_comparison,clip_summary,pair_summary


def row(**kwargs):
    r=dict(active_arm='left',grasp_frame_interval='none',release_frame_interval='none',
           handover_frame_interval='none',legibility='clear',uncertain_reason='')
    r.update(kwargs);return r


class CompareChecks(unittest.TestCase):
    def test_unknown_matches_not_agreement(self):
        r=row(active_arm='uncertain',uncertain_reason='occlusion')
        self.assertEqual(clip_comparison(r,r)['fields']['active_arm'],'indeterminate')
        s=clip_summary({},{});self.assertEqual(s['active_arm']['denominator'],60)
        self.assertIsNone(s['active_arm']['agreement_decidable'])

    def test_closed_interval_compatibility(self):
        a=row(grasp_frame_interval='2-3')
        self.assertEqual(clip_comparison(a,row(grasp_frame_interval='3-4'))['fields']['grasp_frame_interval'],'agreement')
        self.assertEqual(clip_comparison(a,row(grasp_frame_interval='4-5'))['fields']['grasp_frame_interval'],'disagreement')
        self.assertEqual(clip_comparison(a,row())['fields']['grasp_frame_interval'],'disagreement')

    def test_partial_and_exclusion(self):
        a={'K001':row(legibility='partial')};b={'K001':row()}
        self.assertEqual(clip_summary(a,b)['active_arm']['indeterminate'],60)
        self.assertEqual(clip_summary(a,b,include_partial=True)['active_arm']['agreement'],1)
        self.assertEqual(clip_summary(a,b,exclude_k001=True)['active_arm']['denominator'],59)

    def test_pair_unknown_and_raw_matrix(self):
        a={str(i):'indeterminate' for i in range(30)}
        s=pair_summary(a,a);self.assertEqual(s['counts']['indeterminate'],30)
        self.assertEqual(s['raw_same_category_count'],30)
        self.assertIsNone(s['counts']['agreement_decidable'])
        self.assertEqual(pair_summary(a,a,excluded_pair_ids=['0'])['counts']['denominator'],29)

    def test_invalid_and_old_identity_rejected(self):
        with self.assertRaises(ValueError):clip_summary({'K001':row(grasp_frame_interval='8-2')},{})
        with self.assertRaises(ValueError):clip_summary({'C001':row()},{})


if __name__=='__main__':unittest.main()
