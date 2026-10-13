"""Pure validators for compact targeted neural observations."""
import unittest
import numpy as np

def verify(summary, spike_counts, *, require_raw_equals_emitted):
    ids=summary['probe_local_indices']
    assert len(ids)==len(set(ids))
    dt=summary['probe_spike_dt_s']
    for k,r in enumerate(summary['records']):
        steps=round((r['time_end_s']-r['time_start_s'])/dt)
        counts=np.zeros(len(ids),dtype=int)
        events=r['probe_emitted_spike_events_step_column']
        assert len(events)==len({tuple(e) for e in events})
        for step,col in events:
            assert 0<=step<steps and 0<=col<len(ids)
            counts[col]+=1
        np.testing.assert_array_equal(counts,spike_counts[k,ids])
        if require_raw_equals_emitted:assert events==r['probe_raw_spike_events_step_column']

class ProbeTests(unittest.TestCase):
    def test_sparse_event_validation(self):
        summary=dict(probe_local_indices=[1,3],probe_spike_dt_s=.0005,records=[dict(time_start_s=0,time_end_s=.005,probe_emitted_spike_events_step_column=[[0,0],[9,1]],probe_raw_spike_events_step_column=[[0,0],[9,1]])])
        spikes=np.array([[0,1,0,1]])
        verify(summary,spikes,require_raw_equals_emitted=True)
        summary['records'][0]['probe_emitted_spike_events_step_column'].append([0,0])
        with self.assertRaises(AssertionError):verify(summary,spikes,require_raw_equals_emitted=False)
if __name__=='__main__':unittest.main(verbosity=2)
