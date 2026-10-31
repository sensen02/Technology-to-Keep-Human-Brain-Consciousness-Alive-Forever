"""Offline recruitment/phase-trial integrity and behavior metrics. No simulator imports."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np

def main():
 ap=argparse.ArgumentParser();ap.add_argument('directory');args=ap.parse_args();p=Path(args.directory)
 with np.load(p/'BODY.npz') as z:b={k:z[k] for k in z.files}
 a=np.load(p/'HANDOVER_BEFORE.npz');h=np.load(p/'HANDOVER_AFTER.npz');switch=float(h['time'])
 assert all(np.array_equal(a[k],h[k]) for k in a.files if k not in ['ctrl','qfrc_applied','xfrc_applied'])
 assert not h['qfrc_applied'].any() and not h['xfrc_applied'].any()
 t=b['time'];assert np.all(np.diff(t)>0)
 assert all(np.isfinite(x).all() for x in b.values() if np.issubdtype(x.dtype,np.number))
 neural=t>=switch-1e-9;warm=~neural;assert neural.any() and warm.any()
 assert not b['assist'][neural].any() and not b['assist'][:,:6].any() and not b['external'].any() and not b['warning'].any()
 j=int(np.flatnonzero(neural)[0]);assert abs(t[j]-switch)<1e-9
 for key,zkey in [('qpos','qpos'),('qvel','qvel'),('activation','act')]:np.testing.assert_array_equal(b[key][j],h[zkey])
 rec=json.loads((p/'NEURAL_RECORDS.json').read_text());mapping=json.loads((p/'MAPPING.json').read_text());groups={}
 for row in mapping['rows']:
  i=row['actuator_index']
  if not row['local_indices']:assert not b['command'][neural,i].any()
  else:groups.setdefault(tuple(row['pool_root_ids']),[]).append(i)
 assert len(groups)==66
 for ids in groups.values():np.testing.assert_array_equal(b['command'][:,ids],np.repeat(b['command'][:,ids[:1]],len(ids),axis=1))
 for k,x in enumerate(rec):
  assert x['applied_source_interval']==k-1 and not x['auxiliary_force_max'] and not x['external_force_max']
  if k:np.testing.assert_array_equal(x['applied_command'],rec[k-1]['decoded_next_command'])
  else:assert not np.asarray(x['applied_command']).any()
 first=np.flatnonzero(neural&(b['nonfoot_count']>0));late=t>=t[-1]-.1;weight=json.loads((p/'GAINS.json').read_text())['weight']
 pre=warm&(t>=.25);dt=float(np.median(np.diff(t)));flights=[]
 for leg in range(6):
  pad=np.r_[False,(~b['foot_contact'][:,leg])&pre,False].astype(int);start=np.flatnonzero(np.diff(pad)==1);end=np.flatnonzero(np.diff(pad)==-1);flights.append(int(np.sum((end-start)*dt>=.010-1e-9)))
 # A necessary support screen, not an autonomous-walking certification.
 support=bool(not b['nonfoot_count'][neural].any() and np.mean(b['foot_force'][late,:,2].sum(1))/weight>.7)
 result=dict(passed=True,saved_states=len(t),handover_time_s=switch,boundary_state_exact=True,root_and_neural_assistance_zero=True,unresolved16_commands_zero=True,shared66_pools_tied=True,previous_completed_command_causal=True,prewarm_air_bouts_ge10ms=flights,prewarm_dx_native=float(b['qpos'][j,0]-b['qpos'][0,0]),prewarm_nonfoot_absent=bool(not b['nonfoot_count'][warm].any()),neural_duration_s=float(t[-1]-switch),first_nonfoot_relative_s=None if not len(first) else float(t[first[0]]-switch),last100ms_nonfoot_fraction=float(np.mean(b['nonfoot_count'][late]>0)),last100ms_foot_load_ratio=float(np.mean(b['foot_force'][late,:,2].sum(1))/weight),final_com_height=float(b['com'][-1,2]),motor_spikes=sum(x['motor_spikes'] for x in rec),support_necessary_screen=support,autonomous_walking_certified=False,new_physics_steps=0,new_neural_steps=0,body_sha256=hashlib.sha256((p/'BODY.npz').read_bytes()).hexdigest())
 out=p/'PARENT_RECRUITMENT_VERIFICATION.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
