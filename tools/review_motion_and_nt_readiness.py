"""Parent pure-array review of geometry targets and NT dynamics; no model/rollout."""
from pathlib import Path
import sys,json,hashlib,math
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from engine.embodied.kinematic_loss import segment_direction_loss,masked_cartesian_huber
T=Path('/home/sensen/Desktop/engine修复验收/s4-real-motion-retarget/20261102T011240Z')
D=Path('/home/sensen/Desktop/engine修复验收/s4-sourced-nt-dynamic-replay/20261102T011240Z')
O=Path('/home/sensen/Desktop/engine修复验收/s4-motion-nt-readiness-review/20261102T011240Z')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 O.mkdir(parents=True,exist_ok=False)
 inputs=[Path(__file__),R/'engine/embodied/kinematic_loss.py',T/'RETARGETED_TARGETS.npz',T/'RESULT.json',D/'INPUTS.npz',D/'MAPPING.json',D/'BASELINE_GATE.json',D/'RESULT.json']
 inputs += [D/f'{label}_{suffix}.npz' for label in ['BASELINE','CANDIDATE','SELECTED_CURRENT_CUT'] for suffix in ['NEURAL','DECODE']]
 hashes={str(p):sha(p) for p in inputs}
 dump(O/'PREREGISTRATION.json',dict(scope='Post-run independent arrays review only, no selection/fit or simulation',source_sha256=hashes,checks='Native bounds/root/masked claws/proper common rotation/direction error and CSR effect-independent decoder reconstruction/causal command hold',calls=dict(model_load=0,network=0,body_step=0,neural_step=0,training=0)))
 z=np.load(T/'RETARGETED_TARGETS.npz',allow_pickle=False);m=z['model_target_mask'];p=z['achieved_points'];q=z['reference_points'];norm=z['model_chord_lengths'].sum(1)[:,None]
 direction=segment_direction_loss(p,q,m);cartesian=masked_cartesian_huber(p,q,m,normalizer=norm)
 observed=z['direction_residual_rad'][z['direction_target_mask']];assert abs(direction['angular_rms_rad']-float(np.sqrt(np.mean(observed**2))))<1e-12
 rot=z['rotation_source_to_model'];assert abs(np.linalg.det(rot)-1)<1e-12 and abs(rot.T@rot-np.eye(3)).max()<1e-12
 bounds=z['joint_bounds_rad'];j=z['native_joint_qpos'];assert (j>=bounds[:,:,0]-1e-12).all() and (j<=bounds[:,:,1]+1e-12).all()
 assert np.array_equal(z['qpos'][:,z['joint_qpos_addresses']],j) and not z['root_valid'].any() and not bool(z['runtime_marker_loss_enabled']) and not m[:,:,4].any()
 assert np.array_equal(z['qpos'][:,:7],np.tile([0.,0,0,1,0,0,0],(37,1)))
 variation=float((abs(np.linalg.norm(np.diff(p[:,:,:4],axis=2),axis=-1)-z['model_chord_lengths'])/z['model_chord_lengths']).max());assert variation<1e-12
 geometric=dict(direction=direction,cartesian_huber_normalized_by_leg_length=cartesian,angular_mean_degrees=float(observed.mean()*180/np.pi),angular_max_degrees=float(observed.max()*180/np.pi),native_bound_saturated_joint_frames=int(z['bound_saturation'].sum()),adjacent_jumps_over_half_rad=int(z['branch_jump_mask'].sum()),all_jacobian_ranks=np.unique(z['geometric_jacobian_rank']).tolist(),model_chord_max_relative_variation=variation,admitted_as_observed_joint_angle_truth=False)
 gate=json.loads((D/'BASELINE_GATE.json').read_text());assert gate['passed']
 inp=np.load(D/'INPUTS.npz');motor=inp['motor_local'];selected=inp['selected_local'];mapping=json.loads((D/'MAPPING.json').read_text());groups=[np.array(row['local_indices'],int) for row in mapping['rows']]
 baseline_rates=np.load(D/'BASELINE_INITIAL.npz')['decoder_rates'];pending=np.load(D/'BASELINE_INITIAL.npz')['pending'];decay=math.exp(-.0005/.005);metrics={};events={};maxrate=0.
 for label in ['BASELINE','CANDIDATE','SELECTED_CURRENT_CUT']:
  n=np.load(D/f'{label}_NEURAL.npz');a=np.load(D/f'{label}_DECODE.npz');ev=np.unpackbits(n['spikes_packed'],axis=1,count=12000).astype(bool);events[label]=ev
  assert np.array_equal(ev.reshape(50,2,12000).sum(1),n['counts']) and np.array_equal(ev[:,motor],n['motor_events'])
  rate=baseline_rates.copy()
  for tick in range(100):
   instantaneous=np.array([float(ev[tick,group].sum())/len(group)/.0005 if len(group) else 0. for group in groups]);rate=decay*rate+(1-decay)*instantaneous
   maxrate=max(maxrate,float(abs(rate-a['rates_halfstep_hz'][tick]).max()))
  assert np.array_equal(a['applied'][0],pending) and np.array_equal(a['applied'][1:],a['decoded_next'][:-1])
  metrics[label]=dict(network_spikes=int(ev.sum()),MN204_spikes=int(ev[:,motor].sum()),selected_source_spikes=int(ev[:,selected].sum()),applied_saturated_elements=int((a['applied']>=1).sum()),applied_total_elements=int(a['applied'].size),added_edge_events=int(n['added_arrival_event_counts'].sum()),direct_MN_added_events=int(n['added_arrival_event_counts'][:,motor].sum()))
 assert maxrate<1e-10
 cut=inp['cut_current_nA'];base=inp['saved_current_nA'];other=np.ones(12000,bool);other[selected]=False
 assert not cut[:,selected].any() and np.array_equal(cut[:,other],base[:,other])
 cuttrace=np.load(D/'SELECTED_CURRENT_CUT_NEURAL.npz')['added_arrival_event_counts'];assert not cuttrace[5:].any()
 def onset(a,b):
  idx=np.flatnonzero(np.any(a!=b,axis=1));return None if not len(idx) else float((idx[0]+1)*.0005)
 dynamic=dict(baseline_original_exact=True,arms=metrics,decoder_max_error=maxrate,candidate_first_MN_difference_end_s=onset(events['CANDIDATE'][:,motor],events['BASELINE'][:,motor]),cut_first_MN_difference_end_s=onset(events['SELECTED_CURRENT_CUT'][:,motor],events['CANDIDATE'][:,motor]),source_current_cut_exact=True,cut_added_events_after_initial5ticks=0,physical_support_tested=False)
 assert all(sha(Path(f))==h for f,h in hashes.items())
 dump(O/'RESULT.json',dict(passed=True,source_sha256=hashes,geometry=geometric,dynamics=dynamic,calls=dict(model_load=0,network=0,body_step=0,neural_step=0,training=0),limits='Arithmetic and lineage review. IK branches are derived candidates with rank deficiency, not measured joint truth. Candidate graph is a changed model at inherited baseline boundary, and held sensory inputs are not new physical feedback. Source cut retains inherited neural memory. Saturation is descriptive, not performance improvement.'));print((O/'RESULT.json').read_text())
if __name__=='__main__':main()
