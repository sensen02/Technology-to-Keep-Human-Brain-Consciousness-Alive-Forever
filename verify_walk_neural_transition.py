"""Saved-only same-run walking-to-neural transition verifier; no simulator imports."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np

def main():
 ap=argparse.ArgumentParser();ap.add_argument('directory');args=ap.parse_args();o=Path(args.directory)
 with np.load(o/'BODY.npz') as f:b={k:f[k] for k in f.files}
 before=np.load(o/'HANDOVER_BEFORE.npz');after=np.load(o/'HANDOVER_AFTER.npz')
 protected=set(before.files)-{'ctrl','qfrc_applied','xfrc_applied'}
 assert all(np.array_equal(before[k],after[k]) for k in protected)
 assert before['time']>0 and np.any(before['qvel']) and np.any(before['qfrc_applied'])
 assert not after['qfrc_applied'].any() and not after['xfrc_applied'].any()
 t=b['time'];switch=float(after['time']);warm=t<switch-1e-10;neural=t>=switch-1e-10
 assert np.all(np.diff(t)>0) and neural.any() and warm.any()
 for a in b.values():
  if np.issubdtype(a.dtype,np.number):assert np.isfinite(a).all()
 assist=b.get('assist_torque',b.get('assist'));ext=b.get('external_force',b.get('external'));warn=b.get('warnings',b.get('warning'))
 assert not assist[:,:6].any() and not ext.any() and not warn.any()
 assert not assist[neural].any() and np.any(assist[warm])
 boundary=int(np.flatnonzero(neural)[0]);assert abs(t[boundary]-switch)<1e-10
 for name,state_name in [('qpos','qpos'),('qvel','qvel'),('activation','act'),('command','ctrl')]:np.testing.assert_array_equal(b[name][boundary],after[state_name])
 rec=json.loads((o/'NEURAL_RECORDS.json').read_text());assert len(rec)>0
 for k,r in enumerate(rec):
  assert r['applied_source_interval']==k-1
  assert abs(r['absolute_start_s']-(switch+r['relative_start_s']))<1e-10
  assert r['auxiliary_force_max']==0 and r['external_force_max']==0
 assert not np.asarray(rec[0]['applied_command']).any()
 assert np.any([r['motor_spikes'] for r in rec]),'No actual motor-neuron recruitment'
 fc=b['foot_contact'];nf=b.get('nonfoot_floor_count',b.get('nonfoot_count'));com=b.get('com_position',b.get('com'));ff=b.get('ground_force_foot_world',b.get('foot_force'))
 # Evaluate the assisted segment alone; never count neural collapse as walking prewarm.
 assisted=t<switch-1e-10;gait=assisted&(t>=.25);dt=float(np.median(np.diff(t)));durable=[]
 for j in range(6):
  pad=np.r_[False,(~fc[:,j])&gait,False].astype(int);starts=np.flatnonzero(np.diff(pad)==1);ends=np.flatnonzero(np.diff(pad)==-1);durable.append(int(((ends-starts)*dt>=.010-1e-9).sum()))
 assert min(durable)>=3 and not nf[assisted].any()
 dx=float(b['qpos'][boundary,0]-b['qpos'][0,0]);assert dx>.3
 ix=np.flatnonzero(neural&(nf>0));last=t>=t[-1]-.2
 weight=json.loads((o/'GAINS.json').read_text())['weight']
 result=dict(passed=True,same_run_boundary_state_exact=True,boundary_time_s=switch,neural_duration_s=float(t[-1]-switch),saved_states=len(t),prewarm_forward_displacement_native=dx,prewarm_durable_air_bouts_ge10ms=durable,prewarm_nonfoot_contact_absent=True,neural_assist_and_external_force_zero=True,motor_spikes=sum(r['motor_spikes'] for r in rec),neural_first_nonfoot_absolute_s=None if len(ix)==0 else float(t[ix[0]]),neural_first_nonfoot_relative_s=None if len(ix)==0 else float(t[ix[0]]-switch),neural_final_com_height=float(com[-1,2]),last200ms_nonfoot_fraction=float(np.mean(nf[last]>0)),last200ms_mean_foot_load_ratio=float(np.mean(ff[last,:,2].sum(1))/weight),new_physics_steps=0,new_neural_steps=0,body_sha256=hashlib.sha256((o/'BODY.npz').read_bytes()).hexdigest(),note='Pass means real assisted-walk to neural transition with actual recruitment; does not imply autonomous walking, which must be evaluated separately.')
 target=o/'PARENT_TRANSITION_VERIFICATION.json';assert not target.exists();target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
