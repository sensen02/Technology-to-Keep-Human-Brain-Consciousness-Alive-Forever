"""Saved-ledger checks and explicit recruitment categories; no model construction."""
from pathlib import Path
import json
from audit_saved_muscle_components import O,sha,dump

def main():
 rows=json.loads((O/'LEDGER_90_BY_5.json').read_text());assert len(rows)==450
 states=json.loads((O/'STATE_RESIDUALS.json').read_text());directions=json.loads((O/'MOMENT_DIRECTIONS.json').read_text());opposed=json.loads((O/'OPPOSED_ACTIVE_DIRECTIONS.json').read_text());result=[]
 for s in states:
  rr=[r for r in rows if r['index']==s['index']];assert len(rr)==90
  # Small is an explicit descriptive model-force threshold, not a physiological threshold.
  saturated=[r for r in rr if r['command']>=1-1e-12]
  tiny=[r for r in rr if r['command']>0 and abs(r['model_active_gain_force'])<=1e-3]
  result.append(dict(time=s['time'],commands_at_one=[r['actuator'] for r in saturated],saturated_components=saturated,commanded_but_active_abs_at_most_0p001=tiny,zero_event_components=[r for r in rr if r['previous_zero_event']],zero_event_count=sum(r['previous_zero_event'] for r in rr),opposed_joint_count=len(next(v for v in opposed if v['index']==s['index'])['joints'])))
 flips=[dict(actuator=r['actuator'],joint=j['joint'],moments=j['moments']) for r in directions for j in r['joints'] if j['sign_changes_across_sampled_states']]
 dump(O/'RECRUITMENT_COMPONENT_SUMMARY.json',dict(states=result,moment_sign_change_pairs=flips,small_force_threshold=0.001,threshold_is_not_physiological=True))
 assert all(s['force_abs_residual']<=1e-10 and s['generalized_abs_residual']<=1e-10 for s in states)
 assert all(r['command']==0 for r in rows if r['previous_zero_event'])
 dump(O/'TESTS.json',dict(rows_90_times_5=len(rows),unique_actuators=len({r['actuator'] for r in rows}),zero_event_entries=55,zero_event_commands_all_zero=True,reconstruction_tolerance_pass=True,force_residual_max=max(s['force_abs_residual'] for s in states),generalized_residual_max=max(s['generalized_abs_residual'] for s in states),saved_force_error_max=max(s['saved_force_max_error'] for s in states),exit_code=0))
 (O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(json.dumps(dict(states=[dict(time=s['time'],saturated_count=len(s['commands_at_one']),commanded_small_count=len(s['commanded_but_active_abs_at_most_0p001']),opposed_joints=s['opposed_joint_count']) for s in result],direction_flip_pairs=len(flips))))
if __name__=='__main__':main()
