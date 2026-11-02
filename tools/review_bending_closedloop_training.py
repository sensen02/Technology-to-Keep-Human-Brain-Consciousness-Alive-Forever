"""Parent training review plus independent saved-pose FK; no body/neural steps."""
from pathlib import Path
import json,csv,math,hashlib
import numpy as np
from scipy import sparse
ROOT=Path('/home/sensen/Desktop/engine修复验收')
S=ROOT/'s4-bending-closedloop-training/20261102T042848Z'
T=ROOT/'s4-invariant-motion-target/20261102T023600Z/INVARIANT_BENDING_TARGETS.npz'
B=ROOT/'s5-banc888-physical-handover/20261101T220018Z'
G=ROOT/'s4-bending-group-authority/20261102T042848Z/MN_GROUP_MASKS.npz'
C=ROOT/'s4-neutral-bending-cache-score/20261102T042848Z'
O=ROOT/'s4-bending-training-parent-review/20261102T042848Z'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for data in iter(lambda:f.read(1048576),b''):h.update(data)
 return h.hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def scalar_angles(points):
 result=np.empty(points.shape[:-2]+(2,))
 for frame in range(len(points)):
  for leg in range(6):
   for k in range(2):
    a=points[frame,leg,k]-points[frame,leg,k+1];b=points[frame,leg,k+2]-points[frame,leg,k+1]
    cross=[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
    result[frame,leg,k]=math.atan2(math.sqrt(math.fsum(float(v)**2 for v in cross)),math.fsum(float(v*w) for v,w in zip(a,b)))
 return result
def main():
 assert (S/'RESULT.json').exists() and (S/'SELECTED_CANDIDATE.json').exists()
 O.mkdir(parents=True,exist_ok=False)
 sources=[Path(__file__),B/'RUN/MODEL.xml',S/'PREREGISTRATION.json',S/'EXACT_GATE.json',S/'FROZEN_REFERENCE.json',S/'TRAINING_SCORES.json',S/'SPSA_HISTORY.json',S/'SELECTED_CANDIDATE.json',S/'RESULT.json',T,G,C/'RESULT.json']
 sources.extend(S/p for p in ['PARAMETER_GROUPS.npz','PARAMETER_GROUPS.json'])
 identities={str(p):sha(p) for p in sources};dump(O/'PREREGISTRATION.json',dict(scope='Independent completedtraining review; no refit/reselection/body/neuralsteps; one modelcompile and284saved-posekinematics for independentFK',source_sha256=identities,checks='Rawgeometry angle, per-step physical terms, quaternionatan2, fixedreferences, disjointpostMNparam groups, exactCSR masks and incomingvalues,19trialSPSAorder/selection,fullprewarm candidateprefix and neutralcache comparability.'))
 target=np.load(T,allow_pickle=False);truth=scalar_angles(target['observed_keypoints']);assert abs(truth-target['angles_rad']).max()<1e-12
 authority=np.load(G);groups=np.load(S/'PARAMETER_GROUPS.npz');assert np.array_equal(groups['leg_index'],authority['source_group']) and np.array_equal(groups['root_ids'],authority['root_ids'])
 assert np.array_equal(groups['readout_indices'],np.flatnonzero(authority['source_group']>=0))
 gate=json.loads((S/'EXACT_GATE.json').read_text());assert gate['passed'] and gate['checks']['raster_exact'] and gate['checks']['commands_exact'] and max(gate['body_max_abs'].values())==0
 frozen=json.loads((S/'FROZEN_REFERENCE.json').read_text());gatebody=np.load(S/'GATE_NEUTRAL_50MS/BODYLOSS_NATIVE.npz')['rows'];assert frozen['h0']==gatebody[0,10] and np.array_equal(frozen['q0'],gatebody[0,14:18])
 history=json.loads((S/'TRAINING_SCORES.json').read_text());assert len(history)==19 and len(json.loads((S/'SPSA_HISTORY.json').read_text()))==6
 names=[h['label'] for h in history]+['VALIDATION_SELECTED_300MS'];review=[];maxterm=0.;maxangle=0.;sample_hashes={};readout=groups['readout_indices'];legindex=groups['leg_index'];roots=groups['root_ids'];mask=legindex>=0
 import mujoco as mj
 from flygym.compose.fly.musculoskeletal import _load_mjcf
 model=_load_mjcf(B/'RUN/MODEL.xml').compile();data_fk=mj.MjData(model);fk_calls=0;fk_max_error=0.
 legs=['LF','LM','LH','RF','RM','RH'];body_markers=np.array([[model.body(leg+'Coxa').id,model.body(leg+('Trochanter' if leg[1]=='F' else 'Femur')).id,model.body(leg+'Tibia').id,model.body(leg+'Tarsus1').id] for leg in legs])
 baseE=sparse.load_npz(B/'ACTUAL_We.npz');baseI=sparse.load_npz(B/'ACTUAL_Wi.npz');node_rows=np.repeat(np.arange(12000),np.diff(baseE.indptr));node_rows_i=np.repeat(np.arange(12000),np.diff(baseI.indptr))
 for name in names:
  p=S/name;score=json.loads((p/'SCORE.json').read_text());params=np.load(p/'PARAMETERS.npz');z=params['z'];n=score['nframes'];assert n==(37 if name.startswith('VALIDATION') else 13)
  assert np.array_equal(params['leg_index'],legindex) and np.array_equal(params['root_ids'],roots) and np.max(abs(z))<=1
  gain=np.exp(z[:6]*math.log(2));theta=2*z[6:];assert np.array_equal(gain,params['gain']) and np.array_equal(theta,params['theta_mV'])
  expected=params['source_offset'].copy();expected[mask]+=theta[legindex[mask]];assert np.array_equal(expected,params['threshold_offset'])
  for key,base,rows in [('We',baseE,node_rows),('Wi',baseI,node_rows_i)]:
   actual=sparse.load_npz(p/f'ACTUAL_{key}.npz');scale=np.ones(12000);scale[mask]=gain[legindex[mask]]
   assert np.array_equal(actual.indptr,base.indptr) and np.array_equal(actual.indices,base.indices) and np.array_equal(actual.data,base.data*scale[rows]) and np.all(actual.data>0)
  assert sha(p/'RUN/MODEL.xml')=='3470c72f7e68f181ec1a80a5dc5fbff182b750091e7a1ddc7ccfbfb590b80cb6'
  run=json.loads((p/'RUN/EXECUTION.json').read_text());assert run['exit_code']==0 and run['error'] is None
  data=np.load(p/'BODYLOSS_NATIVE.npz')['rows'];assert data.shape==(8001 if n==13 else 24001,20) and np.isfinite(data).all();data=data[1:]
  assert np.array_equal(np.round((data[:,0]-1.3)/.0000125).astype(int),np.arange(1,len(data)+1))
  samples=np.load(p/'TARGET_FK_SAMPLES.npz');assert np.array_equal(samples['source_frame'],target['source_frame'][:n]) and np.array_equal(samples['mask'],target['angle_valid'][:n])
  fk=np.load(p/'RELEASE_FK.npz');nearest=np.array([np.argmin(abs(fk['time']-(1.3+t))) for t in target['time_relative_s'][:n]])
  assert np.array_equal(nearest,samples['FK_indices']) and np.array_equal(fk['points'][nearest],samples['points']) and abs(samples['time_error_s']).max()<=.00025+1e-10
  saved_body=np.load(p/'RUN/BODY.npz');body_times=saved_body['time'];qpos_saved=saved_body['qpos']
  for target_index,when in enumerate(samples['physical_absolute_time_s']):
   body_index=int(np.argmin(abs(body_times-when)));assert abs(float(body_times[body_index]-when))<1e-10
   data_fk.qpos[:]=qpos_saved[body_index];mj.mj_kinematics(model,data_fk);fk_calls+=1
   fk_max_error=max(fk_max_error,float(abs(data_fk.xpos[body_markers]-samples['points'][target_index]).max()))
  assert fk_max_error<1e-10
  angles=scalar_angles(samples['points']);maxangle=max(maxangle,float(abs(angles-samples['prediction_angles_rad']).max()));assert np.allclose(angles,samples['prediction_angles_rad'],atol=1e-12,rtol=0)
  q=data[:,14:18];q0=np.array(frozen['q0']);dot=q[:,0]*q0[0]+q[:,1:]@q0[1:];v=q0[0]*q[:,1:]-q[:,0,None]*q0[1:]-np.cross(q[:,1:],q0[1:]);rot=2*np.arctan2(np.linalg.norm(v,axis=1),abs(dot))
  terms=dict(motion=float(np.mean((angles-truth[:n])**2)),height=float(np.mean(((data[:,10]-frozen['h0'])/frozen['h0'])**2)),foot=float(np.mean((data[:,1:7].sum(1)/frozen['weight']-1)**2)),nonfoot=float(np.mean(data[:,7]/frozen['weight'])),rotation=float(.01*np.mean(rot**2)),ctrl=float(.001*np.mean(data[:,18])),reg=float(.001*np.sum((z[:6]*math.log(2))**2+z[6:]**2)))
  error=max(abs(terms[k]-score['terms'][k]) for k in terms);maxterm=max(maxterm,error);assert error<1e-10 and abs(math.fsum(terms.values())-score['loss'])<1e-10
  assert np.all(data[:,19]==0) and np.all(data[:,7]>=0)
  events=np.unpackbits(np.load(p/'RASTER.npz')['packed'],axis=1,count=12000).astype(bool);records=json.loads((p/'RUN/PREWARM_NEURAL_RECORDS.json').read_text())+json.loads((p/'RUN/NEURAL_RECORDS.json').read_text())
  assert len(events)==2*len(records) and np.array_equal(np.zeros(90),records[0]['applied_command'])
  assert all(np.array_equal(records[i]['applied_command'],records[i-1]['decoded_next_command']) for i in range(1,len(records)))
  for stage,steps in [('PREWARM',2200),('RELEASE',len(events))]:
   state=np.load(p/(stage+'_STATE.npz'));assert np.array_equal(state['spike_count'],events[:steps].sum(0)) and np.array_equal(state['threshold_offset_mV'],expected)
  initial=np.load(p/'NETWORK_INITIAL.npz');assert np.all(initial['v']==-65) and not initial['_ring'].any() and not initial['spike_count'].any() and not initial['decoder_rates'].any()
  review.append(dict(label=name,total=score['loss'],terms=terms,first_positiveNFz_s=None if not (data[:,7]>0).any() else float(data[np.flatnonzero(data[:,7]>0)[0],0]-1.3),positiveNFz_endpoints=int((data[:,7]>0).sum()),mean_foot_load=float(data[:,1:7].sum(1).mean()/frozen['weight']),per_leg_angle_MSE=np.mean((angles-truth[:n])**2,axis=(0,2)).tolist()))
  for file in ['SCORE.json','PARAMETERS.npz','TARGET_FK_SAMPLES.npz','BODYLOSS_NATIVE.npz','ACTUAL_We.npz','ACTUAL_Wi.npz']:sample_hashes[str(p/file)]=sha(p/file)
 rng=np.random.default_rng(4213);z=np.zeros(12)
 for k,h in enumerate(json.loads((S/'SPSA_HISTORY.json').read_text())):
  delta=rng.choice([-1.,1.],12);a=.05/(k+1)**.602;c=.05/(k+1)**.101;assert np.array_equal(delta,h['delta']) and a==h['a'] and c==h['c'] and np.array_equal(z,h['z_before'])
  assert np.array_equal(np.clip(z+c*delta,-1,1),history[3*k+1]['z']) and np.array_equal(np.clip(z-c*delta,-1,1),history[3*k+2]['z'])
  grad=(history[3*k+1]['loss']-history[3*k+2]['loss'])/(2*c)*delta;z=np.clip(z-a*grad,-1,1);assert np.array_equal(grad,h['gradient']) and np.array_equal(z,history[3*k+3]['z'])
 selected=json.loads((S/'SELECTED_CANDIDATE.json').read_text());assert selected==min(history,key=lambda r:r['loss'])
 for file in ['RASTER.npz','BODYLOSS_NATIVE.npz','RUN/BODY.npz','PREWARM_STATE.npz','PARAMETERS.npz']:
  one=np.load(S/selected['label']/file);two=np.load(S/'VALIDATION_SELECTED_300MS'/file)
  for key in one.files:
   x=one[key];y=two[key];assert np.array_equal(x,y[:len(x)] if x.ndim and y.shape!=x.shape else y),(file,key)
 cached=json.loads((C/'RESULT.json').read_text());assert abs(cached['windows']['13']['total_loss']-history[0]['loss'])<1e-10
 # Descriptive paired attribution before/after first positive nonfoot event, no new fitting.
 n0=np.load(S/history[0]['label']/'BODYLOSS_NATIVE.npz')['rows'][1:];nc=np.load(S/selected['label']/'BODYLOSS_NATIVE.npz')['rows'][1:]
 first0=np.flatnonzero(n0[:,7]>0);firstc=np.flatnonzero(nc[:,7]>0);boundary=min(int(first0[0]) if len(first0) else len(n0),int(firstc[0]) if len(firstc) else len(nc))
 def physical_perstep(data):
  quat=data[:,14:18];q0=np.array(frozen['q0']);cos=abs(quat@q0)/np.linalg.norm(quat,axis=1);angle=2*np.arccos(np.clip(cos,0,1))
  return ((data[:,10]-frozen['h0'])/frozen['h0'])**2+(data[:,1:7].sum(1)/frozen['weight']-1)**2+data[:,7]/frozen['weight']+.01*angle**2+.001*data[:,18]
 support_delta=physical_perstep(nc)-physical_perstep(n0);p0=np.load(S/history[0]['label']/'TARGET_FK_SAMPLES.npz');pc=np.load(S/selected['label']/'TARGET_FK_SAMPLES.npz')
 error0=np.mean((p0['prediction_angles_rad']-p0['target_angles_rad'])**2,axis=(1,2));errorc=np.mean((pc['prediction_angles_rad']-pc['target_angles_rad'])**2,axis=(1,2));eventtime=(boundary+1)*.0000125;early=pc['source_nominal_time_s']<eventtime
 attribution=dict(common_first_positive_nonfoot_s=eventtime,support_loss_delta_before_event=float(support_delta[:boundary].sum()/len(support_delta)),support_loss_delta_from_event=float(support_delta[boundary:].sum()/len(support_delta)),motion_loss_delta_before_event=float((errorc-error0)[early].sum()/len(errorc)),motion_loss_delta_from_event=float((errorc-error0)[~early].sum()/len(errorc)),regularization_delta=float(selected['terms']['reg']-history[0]['terms']['reg']),interpretation='Paired native-time and source-time partition only; no causal ablation. Samples are aligned by declared release schedule; subset mask cannot establish walking.')
 attribution['total_reconstructed_delta']=sum(attribution[key] for key in ['support_loss_delta_before_event','support_loss_delta_from_event','motion_loss_delta_before_event','motion_loss_delta_from_event','regularization_delta'])
 assert abs(attribution['total_reconstructed_delta']-(selected['loss']-history[0]['loss']))<1e-10
 val=review[-1];old=cached['windows']['37'];report=dict(passed=True,source_sha256=identities,trial_files_sha256=sample_hashes,trials_checked=20,source_scalar_angle_max_error_rad=maxangle,max_loss_term_difference=maxterm,SPSA_updates_exact=6,all_postMN_gain_and_threshold_groups_match=True,all_modelXML_identity=True,candidate_validation_prefix_exact=True,neutral_training_matches_original_cache=True,selected_label=selected['label'],neutral_train=history[0],selected_train=selected,selected_validation=val,neutral_cached_validation=old,pertrial_recomputed=review,paired_training_loss_time_attribution=attribution,independent_saved_pose_FK_max_error=fk_max_error,calls=dict(model_compile=1,kinematics=fk_calls,network=0,neural_step=0,body_step=0,optimizer=0),limits='All20learning/validation outputs recomputed independently; gateway has its own exactphysicalreproduction evidence. Neither summedlossimprovement nor localbending identifies standing.300ms includes trainingprefix,not independentanimal validation.')
 assert all(sha(Path(p))==h for p,h in identities.items());dump(O/'RESULT.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','trial_files_sha256','pertrial_recomputed','neutral_train','selected_train']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
