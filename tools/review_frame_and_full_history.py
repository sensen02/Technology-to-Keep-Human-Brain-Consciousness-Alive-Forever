"""Independent scalar/frame and full-history replay review; no dynamics or fitting."""
from pathlib import Path
import json,csv,hashlib,math
import numpy as np
from scipy import sparse
ROOT=Path('/home/sensen/Desktop/engine修复验收')
S=ROOT/'s1-continuous-fly-kinematics/20261102T004326Z'
F=ROOT/'s4-fixed-motion-frame-calibration/20261102T023600Z'
N=ROOT/'s4-calibrated-nt-full-history/20261102T023600Z'
B=ROOT/'s5-banc888-physical-handover/20261101T220018Z'
G=ROOT/'s3-sourced-nt-candidate-graph/20261102T004326Z'
O=ROOT/'s4-frame-fullhistory-review/20261102T023600Z'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def normalize(v):return v/math.sqrt(math.fsum(float(x)**2 for x in v))
def main():
 O.mkdir(parents=True,exist_ok=False)
 files=[Path(__file__),S/'annotated_3d_walking_kinematics.csv',F/'FRAME_ARRAYS.npz',F/'FRAME_COMPARISON.json',F/'TRAIN_LOCKED_GEOMETRY_TARGETS.npz',N/'MAPPING.json',N/'BASELINE_REUSED.npz',N/'INPUTS.npz',N/'RESULT.json']
 files += [N/f'{label}_{kind}.npz' for label in ['CANDIDATE','SELECTED_CURRENT_CUT'] for kind in ['ACTUAL_We','ACTUAL_Wi','INITIAL','BOUNDARY','FINAL','NEURAL','DECODE']]
 files += [B/'ACTUAL_We.npz',B/'ACTUAL_Wi.npz',G/'ADDED_We.npz']
 hashes={str(p):sha(p) for p in files};dump(O/'PREREGISTRATION.json',dict(scope='Independent completed outputs review, no model/fitting/neural integration',source_sha256=hashes,selection='Do not change frame or gain choices; recompute fixed training-only gates and actual array causal timeline'))
 rows=list(csv.DictReader((S/'annotated_3d_walking_kinematics.csv').open()));unit=normalize;forward=np.zeros(3);lateral=np.zeros(3)
 def vec(row,key):return np.array([float(row[key+'_'+axis]) for axis in 'xyz'])
 for row in rows[:22]:
  forward+=(unit(vec(row,'He')-vec(row,'Th'))+unit(vec(row,'Th')-vec(row,'Abd')))/44
  lateral+=(sum((vec(row,key+'-ThC') for key in ['LF','LM','LH']),np.zeros(3))-sum((vec(row,key+'-ThC') for key in ['RF','RM','RH']),np.zeros(3)))/66
 x=unit(forward);y=unit(lateral-x*np.dot(x,lateral));rotation=np.array([x,y,np.cross(x,y)])
 fa=np.load(F/'FRAME_ARRAYS.npz');comparison=json.loads((F/'FRAME_COMPARISON.json').read_text());locked=np.load(F/'TRAIN_LOCKED_GEOMETRY_TARGETS.npz')
 error=float(abs(rotation-fa['anatomical_R']).max());assert error<1e-14 and abs(np.linalg.det(rotation)-1)<1e-12
 tr=fa['split']==0;src=fa['source_anchor_train_mean'];model=fa['model_anchor_points'];src=(src-src.mean(0))/np.linalg.norm(src-src.mean(0));model=(model-model.mean(0))/np.linalg.norm(model-model.mean(0))
 oldanchor=float(np.linalg.norm(src@fa['old_R'].T-model,axis=1).mean());newanchor=float(np.linalg.norm(src@rotation.T-model,axis=1).mean())
 improvement=float((fa['old_lower_deg'][tr][:,[1,4]]-fa['anatomical_lower_deg'][tr][:,[1,4]]).mean());delta=fa['anatomical_lower_deg'][tr].mean(0)-fa['old_lower_deg'][tr].mean(0);maxdelta=fa['anatomical_lower_deg'][tr].max(0)-fa['old_lower_deg'][tr].max(0)
 gates=dict(anchor=bool(newanchor-oldanchor<=.05),middle=bool(improvement>=10),six_leg_mean=bool(np.all(delta<=2)),six_leg_max=bool(np.all(maxdelta<=5)))
 assert gates==comparison['gates'] and not all(gates.values()) and comparison['selected_R']=='old'
 assert np.array_equal(locked['rotation_source_to_model'],fa['old_R']) and not locked['native_joint_loss_mask'].any() and not locked['root_valid'].any() and not bool(locked['runtime_enabled'])
 mapping=json.loads((N/'MAPPING.json').read_text());groups=[np.array(r['local_indices'],int) for r in mapping['rows']];inp=np.load(N/'INPUTS.npz');baseline=np.load(N/'BASELINE_REUSED.npz');orig=np.load(B/'NEURAL_RASTER.npz')['spikes'][:2500];stored=np.unpackbits(baseline['spikes_packed'],axis=1,count=12000).astype(bool);assert np.array_equal(stored,orig)
 baseE=sparse.load_npz(B/'ACTUAL_We.npz');baseI=sparse.load_npz(B/'ACTUAL_Wi.npz');add=sparse.load_npz(G/'ADDED_We.npz');expected=(baseE+.03125*add).tocsr();stats={};rateerror=0.;all_events={};all_decode={}
 for label in ['CANDIDATE','SELECTED_CURRENT_CUT']:
  e=sparse.load_npz(N/f'{label}_ACTUAL_We.npz');i=sparse.load_npz(N/f'{label}_ACTUAL_Wi.npz')
  for a,b in [(e,expected),(i,baseI)]:assert all(np.array_equal(v,w) for v,w in [(a.indptr,b.indptr),(a.indices,b.indices),(a.data,b.data)])
  n=np.load(N/f'{label}_NEURAL.npz');d=np.load(N/f'{label}_DECODE.npz');boundary=np.load(N/f'{label}_BOUNDARY.npz');initial=np.load(N/f'{label}_INITIAL.npz');end=np.load(N/f'{label}_FINAL.npz');sp=np.unpackbits(n['spikes_packed'],axis=1,count=12000).astype(bool);all_events[label]=sp;all_decode[label]=d
  assert not initial['_ring'].any() and not initial['spike_count'].any() and not initial['decoder_rates'].any() and not initial['pending'].any()
  assert np.array_equal(sp.reshape(1250,2,12000).sum(1),n['counts']) and np.array_equal(boundary['spike_count'],sp[:2200].sum(0))
  assert np.array_equal(boundary['_ring'],sp[2195:2200]) and int(boundary['steps'])==2200 and float(boundary['t_ms'])==1100 and int(boundary['ring_i'])==0
  assert np.array_equal(boundary['pending'],d['applied'][1100]) and np.array_equal(boundary['pending'],d['decoded_next'][1099])
  assert np.array_equal(d['applied'][0],np.zeros(90)) and np.array_equal(d['applied'][1:],d['decoded_next'][:-1])
  rate=np.zeros(90);decay=math.exp(-.0005/.005)
  for tick,event in enumerate(sp):
   instant=np.array([event[g].sum()/len(g)/.0005 if len(g) else 0. for g in groups]);rate=decay*rate+(1-decay)*instant
   rateerror=max(rateerror,float(abs(rate-d['rates_halfstep_hz'][tick]).max()))
  assert np.allclose(rate,end['decoder_rates'],rtol=0,atol=1e-12) and np.array_equal(end['spike_count'],sp.sum(0))
  stats[label]=dict(prewarm_MN_spikes=int(n['motor_events'][:2200].sum()),release150_MN_spikes=int(n['motor_events'][2200:].sum()),prewarm_applied_saturated=int((d['applied'][:1100]>=1).sum()),release150_applied_saturated=int((d['applied'][1100:]>=1).sum()),release_pending_diff_elements=int((boundary['pending']!=baseline['applied'][1100]).sum()),release_first20ms_command_diff_elements=int((d['applied'][1100:1120]!=baseline['applied'][1100:1120]).sum()))
  if label=='SELECTED_CURRENT_CUT':
   assert not n['selected_events'].any() and not n['selected_arrivals'].any() and not n['added_arrival_event_counts'].any() and not n['added_arrival_increment_uS'].any()
   assert np.array_equal(n['motor_events'],baseline['motor_events']) and np.array_equal(d['decoded_next'],baseline['decoded_next']) and np.array_equal(d['applied'],baseline['applied'])
 assert rateerror<1e-10
 report=dict(passed=True,source_sha256=hashes,frame=dict(train_only_landmark_R_max_error=error,old_anchor_mean=oldanchor,anatomical_anchor_mean=newanchor,middle_necessary_bound_improvement_deg=improvement,gates=gates,selected='old',full_record_ik_triggered=False,independent_real_frame_identified=False),neural_history=dict(ticks_per_arm=2500,networks_simulated_by_this_review=0,candidate_stats=stats,decoder_max_error=rateerror,baseline_original_raster_exact=True,cut_selected_has_zero_spikes_arrivals_added_effects=True,cut_all_MN_and_commands_equal_original_baseline=True,candidate_own_warm_boundary_preserved=True,physical_feedback_tested=False),calls=dict(model_compile=0,optimizer=0,network=0,neural_step=0,body_step=0))
 assert all(sha(Path(p))==h for p,h in hashes.items());dump(O/'RESULT.json',report);print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))
if __name__=='__main__':main()
