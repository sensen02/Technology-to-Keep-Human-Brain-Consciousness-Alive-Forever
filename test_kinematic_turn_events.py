import unittest
import numpy as np
from kinematic_turn_events import extract_turns
class TurnTests(unittest.TestCase):
 def test_plateau(self):
  v=np.r_[np.arange(30.),np.full(12,29.),np.arange(28.,-1,-1)]
  r=extract_turns(v,np.arange(len(v)))
  self.assertEqual(len(r['events']),1);self.assertIn('PLATEAU_EXTREMUM',r['events'][0]['ambiguity'])
 def test_jitter_candidates_retained(self):
  t=np.arange(150);v=np.sin(t*.2)+.03*np.sin(t*2)
  r=extract_turns(v,t);self.assertGreater(len(r['events']),3)
  self.assertTrue(any('SHORT_EVENT_INTERVAL' in e['ambiguity'] for e in r['events']))
 def test_missing_segments_not_joined(self):
  v=np.sin(np.arange(180)*.1);v[70:100]=np.nan
  r=extract_turns(v,np.arange(len(v)))
  self.assertTrue(np.isnan(r['smoothed'][70:100]).all())
  self.assertFalse(any(c['start_frame']<70 and c['end_frame']>100 for c in r['candidate_cycles']))
  self.assertEqual(len(r['boundaries']),2)
 def test_incomplete_monotonic(self):
  r=extract_turns(np.arange(40.),np.arange(40))
  self.assertEqual(r['events'],[]);self.assertEqual(r['candidate_cycles'],[])
  self.assertEqual(len(r['boundaries']),1)
 def test_positive_scale(self):
  v=np.sin(np.arange(180)*.08);f=np.arange(180)
  a=extract_turns(v,f);b=extract_turns(v*7,f)
  self.assertEqual([e['source_frame'] for e in a['events']],[e['source_frame'] for e in b['events']])
if __name__=='__main__':unittest.main()
