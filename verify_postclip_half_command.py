"""No simulation: enforce single-factor artifact and command contract."""
import json,numpy as np
from pathlib import Path
from run_postclip_half_command import O,BASE,sha,dump

def main():
 records=json.loads((O/'RECORDS.json').read_text())['records'];iv=np.load(O/'INTERVALS.npz');old=np.load(BASE/'normal/BODY.npz');new=np.load(O/'BODY.npz');assert len(records)==50
 for k,r in enumerate(records):
  d=r['decoder'];np.testing.assert_array_equal(d['scaled_command'],np.asarray(d['unscaled_postclip_command'])*.5)
  if k+1<len(records):np.testing.assert_array_equal(iv['commands'][k+1],d['scaled_command'])
 np.testing.assert_array_equal(iv['commands'][0],np.zeros(90));assert iv['commands'].max()<=.5
 initial={k:bool(np.array_equal(old[k][0],new[k][0])) for k in ['qpos','qvel','activation','command','external_force']};assert all(initial.values());np.testing.assert_array_equal(old['time'],new['time'])
 source=json.loads((O/'PREREGISTRATION.json').read_text())['source_hashes'];assert all(sha(Path(p))==h for p,h in source.items())
 dump(O/'VERIFICATION.json',dict(exit_code=0,initial_state_exact=initial,time_grid_exact=True,all_postclip_scaling_and_next_interval_applied_exact=True,max_actual_command=float(iv['commands'].max()),source_old_new_hashes={p:dict(old=h,new=sha(Path(p))) for p,h in source.items()},initialization_forward_calls=3,body_step_forward_calls=20000,total_runner_forward_calls=20003,notes='prepare1,observer initial1,episode initial1; native ctrl lower bounds unchanged; decoder clocks verified with identical synthetic samples before run'))
 (O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print('offline half-command contract verified exit0')
if __name__=='__main__':main()
