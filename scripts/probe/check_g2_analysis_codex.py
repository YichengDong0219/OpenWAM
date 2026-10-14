"""Result-blind analytical fixtures for A3; not independent acceptance."""
import unittest
import numpy as np
from g2_analysis_codex import physical,direction,hierarchy

def pose():
 a=np.zeros((32,20),np.float64);a[:,[3,13]]=1;a[:,[7,17]]=1;return a
class Tests(unittest.TestCase):
 def test_xyz_units_grip_rotation(self):
  a=pose();b=a.copy();b[:,0]=.003;b[:,1]=.004;b[:,9]=.2;m=physical(a,b);self.assertAlmostEqual(m['D_xyz_m'],.0025);self.assertAlmostEqual(m['max_xyz_m'],.005);self.assertAlmostEqual(m['gripper_mean_abs'],.1);self.assertEqual(m['rotation_invalid_A'],0);self.assertEqual(m['rotation_mean_deg'],0)
 def test_known_quarter_turn(self):
  a=pose();b=a.copy();b[:,3:9]=[0,1,0,-1,0,0];b[:,13:19]=[0,1,0,-1,0,0];self.assertAlmostEqual(physical(a,b)['rotation_mean_deg'],90)
 def test_direction_threshold_zero_sign(self):
  a=pose();d=a.copy();d[:,0]=d[:,10]=.001;r=direction(a,a,d);self.assertTrue(r['eligible']);self.assertEqual(r['p'],0);self.assertIsNone(r['cosine']);self.assertEqual(direction(a,a,a)['p'],None);s=a.copy();s[:,0]=s[:,10]=-.002;r=direction(s,a,d);self.assertAlmostEqual(r['p'],-2);self.assertAlmostEqual(r['s_m'],-.002);self.assertAlmostEqual(r['r_perp_m'],0);self.assertAlmostEqual(r['cosine'],-1)
 def test_orthogonal_residual(self):
  a=pose();d=a.copy();d[:,0]=d[:,10]=.002;s=a.copy();s[:,1]=s[:,11]=.003;r=direction(s,a,d);self.assertEqual(r['p'],0);self.assertAlmostEqual(r['r_perp_m'],.003);self.assertEqual(r['cosine'],0)
 def test_hierarchy_unequal_rows_and_empty_episode(self):
  obs=[{'obs_id':'a','episode':0,'task':'t'},{'obs_id':'b','episode':0,'task':'t'},{'obs_id':'c','episode':1,'task':'t'},{'obs_id':'d','episode':2,'task':'t'}]
  rows=[{'obs_id':'a','metrics':{'x':0}}]*10+[{'obs_id':'b','metrics':{'x':2}},{'obs_id':'c','metrics':{'x':9}}]
  s=hierarchy(rows,['x'],obs)['tasks']['t']['metrics']['x'];self.assertEqual(s['mean'],5);self.assertEqual(s['empty_episodes'],[2]);self.assertEqual(s['n_eligible_episodes'],2);self.assertEqual(s['ci95'],[1,9])
 def test_degenerate_rotations_explicit(self):
  z=np.zeros((32,20));m=physical(z,z);self.assertEqual(m['rotation_invalid_A'],64);self.assertEqual(m['rotation_valid_comparisons'],0);self.assertIsNone(m['rotation_mean_deg']);self.assertEqual(m['D_xyz_m'],0)
 def test_nonfinite_actions_rejected(self):
  a=pose();a[1,2]=np.nan
  with self.assertRaises(AssertionError):physical(a,pose())
if __name__=='__main__':unittest.main()
