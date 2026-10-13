import json,copy,unittest
from pathlib import Path
import numpy as np
import run_spike_filter_reconstruction as r
class ReconstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=Path(r.validate_bundle(r.OUT)['path'])
        cls.d=json.loads((p/'closure_results.json').read_text());cls.m=json.loads((p/'motor_manifest.json').read_text())['measured']
        with np.load(p/'closure_traces.npz') as z:cls.a={k:z[k] for k in z.files}
    def test_exact_and_duplicate_rejected(self):
        self.assertEqual(len(r.reconstruct(self.m,self.d['measured']['episodes'],self.a,self.d['provenance']['motor_translation'])),7)
        eps=copy.deepcopy(self.d['measured']['episodes'])
        record=next(x for x in eps['normal']['records'] if x['probe_emitted_spike_events_step_column'])
        record['probe_emitted_spike_events_step_column'].append(record['probe_emitted_spike_events_step_column'][0])
        with self.assertRaises(ValueError):r.reconstruct(self.m,eps,self.a,self.d['provenance']['motor_translation'])
    def test_removed_motor_event_detected(self):
        eps=copy.deepcopy(self.d['measured']['episodes']); ep=eps['normal'];motor={i for row in self.m['rows'] for i in row['local_indices']}
        removed=False
        for record in ep['records']:
            events=record['probe_emitted_spike_events_step_column']
            for i,event in enumerate(events):
                if ep['probe_local_indices'][event[1]] in motor:events.pop(i);removed=True;break
            if removed:break
        self.assertTrue(removed)
        with self.assertRaises(AssertionError):r.reconstruct(self.m,eps,self.a,self.d['provenance']['motor_translation'])
if __name__=='__main__':unittest.main(verbosity=2)
