"""Zero-integration tests for 550ms gated entry and output schema."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import run_motor_cadence_long as long


class LongEntryTests(unittest.TestCase):
    def test_denies_absent_authorization_before_body_setup(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'arm';out.mkdir();(out/'PREREGISTRATION.md').write_text('test')
            with patch.object(long.closure,'prepare_body',side_effect=AssertionError('body invoked')):
                with self.assertRaisesRegex(ValueError,'human-scope receipt'):
                    long.execute(out=out,period_ms=1,authorization=Path(tmp)/'missing')
                for period in (0,2,10,'1',True):
                    with self.subTest(period=period),self.assertRaises(ValueError):
                        long.execute(out=out,period_ms=period,authorization=Path(tmp)/'missing')
                with self.assertRaisesRegex(ValueError,'only normal 550ms'):
                    long.execute(out=out,period_ms=1,authorization=Path(tmp)/'missing',
                                 replay=np.zeros((110,1)))
                with self.assertRaisesRegex(ValueError,'only normal 550ms'):
                    long.execute(out=out,period_ms=1,authorization=Path(tmp)/'missing',
                                 duration_s=.01)

    def test_rejects_unconfirmed_or_wrong_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'arm';out.mkdir();(out/'PREREGISTRATION.md').write_text('test')
            scope=Path(tmp)/'scope.json'
            valid=dict(authority='direct-human',objective='two frozen motor cadence arms',
                       duration_s_per_arm=.55,maximum_body_steps_total=176000,
                       motor_periods_ms=[1,5],human_confirmation='test-only synthetic receipt',
                       preregistration_sha256=long.sha256(out/'PREREGISTRATION.md'))
            for change in ({'authority':'advisor'}, {'duration_s_per_arm':.01},
                           {'maximum_body_steps_total':176001},
                           {'motor_periods_ms':[1]}, {'human_confirmation':''},
                           {'preregistration_sha256':'wrong'}):
                scope.write_text(json.dumps({**valid,**change}))
                with self.subTest(change=change),self.assertRaisesRegex(ValueError,'human-scope'):
                    long.require_scope(out,1,scope)
            (out/'SUMMARY.json').write_text('{}')
            scope.write_text(json.dumps(valid))
            with self.assertRaisesRegex(ValueError,'fresh output'):
                long.require_scope(out,1,scope)

    def test_frozen_identity_rejected_before_body_setup(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'arm';out.mkdir();(out/'PREREGISTRATION.md').write_text('test')
            scope=Path(tmp)/'scope.json'
            scope.write_text(json.dumps(dict(
                authority='direct-human',objective='two frozen motor cadence arms',
                duration_s_per_arm=.55,maximum_body_steps_total=176000,
                motor_periods_ms=[1,5],human_confirmation='test fixture, not approval',
                preregistration_sha256=long.sha256(out/'PREREGISTRATION.md'))))
            with patch.object(long,'require_frozen_inputs',side_effect=ValueError('frozen FIR mismatch')), \
                 patch.object(long.closure,'prepare_body',side_effect=AssertionError('body invoked')):
                with self.assertRaisesRegex(ValueError,'frozen FIR mismatch'):
                    long.execute(out=out,period_ms=1,authorization=scope)
            self.assertEqual({p.name for p in out.iterdir()},{'PREREGISTRATION.md'})

    def test_verify_both_arm_arrays_without_simulator(self):
        # Fake arrays are contract fixtures, never real body/neural evidence.
        steps=np.column_stack((np.arange(1,88001)*long.BODY_DT,
                               np.ones(88000),np.zeros(88000),np.zeros(88000)))
        for period,intervals,ticks in ((1,550,2),(5,110,10)):
            with self.subTest(period=period):
                commands=np.zeros((intervals,2)); decoded=np.zeros_like(commands)
                start=np.arange(intervals)*period/1000
                end=start+period/1000
                current=np.zeros((110,3)); act=np.zeros((intervals,2))
                if period==1:
                    evidence=dict(motor_start_s=start,motor_end_s=end,
                                  motor_applied_controls=commands,
                                  motor_decoded_next_controls=decoded,
                                  sensory_sample_start_s=np.arange(110)*.005,
                                  sensory_current_nA=current,motor_activation_end=act,
                                  motor_source_end_tick=np.r_[-1,np.arange(2,1100,2)],
                                  motor_sensory_index=np.repeat(np.arange(110),5),
                                  motor_event_complete_ticks=tuple(
                                      np.array([k*ticks+1]) if k==0 else np.array([],dtype=int)
                                      for k in range(intervals)),
                                  motor_event_local_indices=tuple(
                                      np.array([0]) if k==0 else np.array([],dtype=int)
                                      for k in range(intervals)))
                    result={}
                else:
                    evidence=dict(interval_start_s=start,interval_end_s=end,
                                  decoded_next_controls=decoded,sensory_current_nA=current,
                                  activation=act,applied_source_interval=np.r_[-1,np.arange(109)])
                    result=dict(probe_local_indices=[0],motor_manifest_rows=[dict(local_indices=[0])],
                                records=[dict(probe_emitted_spike_events_step_column=
                                              [[0,0]] if k==0 else []) for k in range(intervals)])
                arrays=long.verify_observations(evidence,commands,steps,result,period,3)
                self.assertEqual(arrays['motor_event_offsets'].shape,(intervals+1,))
                self.assertEqual(arrays['motor_event_complete_ticks_flat'].tolist(),[1])
                self.assertEqual(arrays['motor_source'][0],-1)
                bad_commands=commands.copy();bad_commands[0,0]=1
                with self.assertRaises(AssertionError):
                    long.verify_observations(evidence,bad_commands,steps,result,period,3)
                bad_steps=steps[:-1]
                with self.assertRaises(AssertionError):
                    long.verify_observations(evidence,commands,bad_steps,result,period,3)


if __name__=='__main__':
    unittest.main()
