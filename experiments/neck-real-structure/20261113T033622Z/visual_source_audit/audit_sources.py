"""Stdlib source/evidence hashes and NPY headers only; never deserialize array payloads/models."""
from pathlib import Path
import json,hashlib,zipfile,struct,ast,collections,sys
ROOT=Path(__file__).resolve().parent
PROJECT=Path('/run/media/sensen/Data2/cell_wound_prototype')
VIS=PROJECT/'experiments/visual-orientation/20261112T045316Z'
V7=VIS/'landmark_opponent_v7'
NECK=PROJECT/'experiments/neck-real-structure/20261113T000915Z'
EXPECTED_VISUAL='996980806ac107137d6d5f4743de207fb3350e84a8a145dc5f52b8807398bb87'
EXPECTED_BRIDGE='415304b7027dda17f359350052934670c1aa2bd44a0025b1dd0089d549997b27'
cache={}
def sha(path):
 p=str(Path(path).resolve())
 if p not in cache:
  with Path(p).open('rb') as f:cache[p]=hashlib.file_digest(f,'sha256').hexdigest()
 return cache[p]
def read(path):return json.loads(Path(path).read_text())
def headers(path):
 out={}
 with zipfile.ZipFile(path) as archive:
  for name in archive.namelist():
   if not name.endswith('.npy'):raise ValueError('Unexpected archive entry '+name)
   with archive.open(name) as f:
    if f.read(6)!=b'\x93NUMPY':raise ValueError('Not NPY header')
    version=tuple(f.read(2));n=struct.unpack('<H' if version[0]==1 else '<I',f.read(2 if version[0]==1 else 4))[0]
    if n>65536:raise ValueError('Unbounded NPY header')
    header=ast.literal_eval(f.read(n).decode('latin1' if version[0]<3 else 'utf8'))
   out[name[:-4]]=dict(dtype=header['descr'],shape=list(header['shape']),fortran_order=header['fortran_order'],header_only_no_payload_read=True)
 return out

def main():
 checks={};evidence={};selected=[]
 def verify_map(label,mapping):
  failures=[]
  for p,h in mapping.items():
   if sha(p)!=h:failures.append(p)
  checks[label]=not failures
  if failures:raise ValueError('Hash drift '+str(failures))
 accepted=read(V7/'FINAL_ACCEPTANCE.json');verify_map('visual_final_acceptance_evidence_hashes',accepted['source_evidence_hashes']);checks['visual_checkpoint_exact']=sha(V7/'policy/TRAIN/checkpoint_001000.pt')==EXPECTED_VISUAL
 saved=read(V7/'policy/TRAIN/VERIFY_001000.json');checks['saved_final1000_proof']=saved['all_passed'] and saved['step']==1000 and saved['sha256']==EXPECTED_VISUAL and all(saved['checks'].values());verify_map('saved_visual_verifier_source_hashes',saved['source_hashes'])
 validation=read(V7/'policy/VALIDATION/RESULT.json');va=read(V7/'policy/VALIDATION/VERIFY.json');checks['offline_validation_proof']=validation['all_targets_pass'] and va['integrity_all_pass'] and all(va['checks'].values())
 if 'result_sha256' in va:checks['offline_validation_result_sha']=va['result_sha256']==sha(V7/'policy/VALIDATION/RESULT.json')
 summary=read(V7/'body/VISUAL_BODY_SUMMARY.json');verify_map('visual_body_summary_declared_sources',summary['source_hashes'])
 ids=('V_empty','V_center','V_left_pulse','V_right_pulse','V_left_to_right','V_right_to_left','V_left_occluded','V_flip_fixedframe')
 for name in ids:
  folder=V7/'body/trials'/name;r=read(folder/'RESULT.json');a=read(folder/'VERIFY.json');cap=read(folder/'CAPTURES.json');verify_map(name+'_runtime_source_hashes',r['source_hashes'])
  checks[name+'_complete_prior_integrity']=a['integrity_all_pass'] and all(a['checks'].values()) and a['result_sha256']==sha(folder/'RESULT.json') and r['visual_checkpoint_sha256']==EXPECTED_VISUAL
  for filename,key in [('capture_rgb.npz','capture_rgb'),('neural_decisions.npz','neural_decisions'),('geometry_1ms.npz','geometry'),('CAPTURES.json','captures_metadata')]:checks[name+'_artifact_'+key]=sha(folder/filename)==r[key]['sha256']
  raw=headers(folder/'capture_rgb.npz');decisions=headers(folder/'neural_decisions.npz');checks[name+'_archive_header_shapes']=raw['rgb']['shape']==[770,2,32,32,3] and raw['rgb']['dtype']=='|u1' and decisions['X238']['shape']==[2880,238] and decisions['held_features']['shape']==[2880,16] and decisions['vision_hidden_after']['shape']==[2880,32]
  checks[name+'_metadata_50Hz_20ms_release']=len(cap)==770 and all(c['capture_id']==j and abs(c['capture_time_s']-.02*j)<1e-9 and abs(c['release_time_s']-c['capture_time_s']-.02)<1e-9 and c['render_does_not_modify_physical_data'] and c['contact_pipeline_unchanged'] for j,c in enumerate(cap))
  selected.append(dict(condition=name,result=str(folder/'RESULT.json'),verify=str(folder/'VERIFY.json'),independent_prior_checks=len(a['checks']),capture_rgb=dict(path=str(folder/'capture_rgb.npz'),sha256=sha(folder/'capture_rgb.npz'),headers=raw),neural_decisions=dict(path=str(folder/'neural_decisions.npz'),sha256=sha(folder/'neural_decisions.npz'),headers=decisions),capture_metadata=str(folder/'CAPTURES.json'),original_engineering_pass=r['engineering_pass'],slip=r['autonomous']['slip_ratio_max'],functional_stable_body=r['functional_stable_body'],neural_records=2880,autonomous_slice=[960,2880],image_actor_freeze_control=name=='V_flip_fixedframe',recording_labels_not_inputs=True))
 final=read(NECK/'FINAL_SCOPED_ACCEPTANCE.json');verify_map('neck_final_scoped_acceptance_evidence_hashes',final['evidence_hashes']);delivery=read(NECK/'DELIVERY.json')
 for row in delivery['items']:checks['neck_delivery_'+Path(row['desktop']).name]=sha(row['source'])==sha(row['desktop'])==row['sha256']
 checks['actual_bridge_weights_exact']=sha(NECK/'feedback_aggregation_v2/TRAIN/checkpoint_000400.pt')==EXPECTED_BRIDGE
 checks['scoped_neck_done_not_biological_full_neck']=final['status']=='SCOPED_IMPLEMENTATION_OBJECTIVE_VERIFIED_AND_DELIVERED' and final['empty_original_five_gates_pass'] and not final['complete_biological_neck_model'] and final['latest_brain_sha256']==EXPECTED_BRIDGE
 paths=[VIS/'camera.py',VIS/'PREREG_CAMERA_PROBE.json',VIS/'visual_body_v1/runtime.py',VIS/'visual_body_v1/runner.py',VIS/'visual_body_v1/audit.py',VIS/'visual_body_v1/summary.py',VIS/'visual_body_v2/adapter.py',VIS/'landmark_far_v3/body/runtime.py',VIS/'landmark_far_v3/body/adapter.py',VIS/'landmark_far_v3/body/summary.py',VIS/'landmark_coverage_v5/body/runtime.py',V7/'policy/model.py',V7/'policy/common.py',V7/'body/runtime.py',V7/'body/adapter.py',V7/'body/runner.py',V7/'body/audit.py',V7/'FINAL_ACCEPTANCE.json',V7/'policy/VALIDATION/RESULT.json',V7/'policy/VALIDATION/VERIFY.json',V7/'body/VISUAL_BODY_SUMMARY.json',NECK/'FINAL_SCOPED_ACCEPTANCE.json',NECK/'DELIVERY.json',ROOT/'PREREG.json',Path(__file__)]
 for p in paths:evidence[str(p)]=sha(p)
 checks['no_numerical_backend_imported']=not any(k in sys.modules for k in ('numpy','torch','mujoco','physics','student_wave','flygym'))
 report=dict(status='SOURCE_AND_SAVED_EVIDENCE_AUDIT_ONLY_NO_NEW_VISION_EXECUTION',all_checks_passed=all(checks.values()),checks=checks,visual_checkpoint=dict(path=str(V7/'policy/TRAIN/checkpoint_001000.pt'),sha256=EXPECTED_VISUAL,format='VISION_OPPONENT_POLICY_V7',step=1000,architecture='VISION_FIXED_NEURAL_OPPONENT_ARCH_V7'),bridge_checkpoint=dict(path=str(NECK/'feedback_aggregation_v2/TRAIN/checkpoint_000400.pt'),sha256=EXPECTED_BRIDGE,format='SOURCE_NECK_FEEDBACK_AGGREGATION_V2',step=400,brain_hidden=16,DN=2,AN=1,body_sensory_channels=111,source_binding_sha256='9e88a04dd91aa6f9c1c1c68e0562325205e0f36f5cdd9ae41d1dd46c8171dc23'),input_contract=dict(rgb_shape_per_exposure=[2,32,32,3],dtype='uint8',eye_order=['LEFT','RIGHT'],eye_yaw_deg=[25.,-25.],eye_pitch_deg=-15.,vertical_FOV_deg=100.,near_native=.005,far_native=20.,capture_interval_s=.02,release_delay_s=.02,decision_s=.005,feature_cache=16,features_per_eye=8,visual_recurrent_memory=32,visual_alpha_formula='-expm1(-.005/.1)',output='raw_turn=tanh(latent),continuous[-1,1]',fixed_opponent_weights=[1.,-.5,-.5],retina_biological_validity=False),far_target_geometry=dict(landmark_center='handover eye midpoint+2*horizontalheadX+2*tan(angle)*horizontalheadY;z1.2',left_right_angles_radians=[.4,-.4],render_only=True,body1_near_default_not_same_source=True),existing_acceptance=dict(status=accepted['status'],offline_required_pass=accepted['offline_required_pass'],body_normal6_functional_pass=accepted['body_normal6_functional_pass'],control2_pass=accepted['body_control2_pass'],original_engineering_all8_pass=False,original_slip_failure_conditions=accepted['original_slip_failure_conditions'],weak_LTR_body_effect_deg=accepted['weak_left_to_right_body_effect_deg'],scope=accepted['scope']),archive_candidates=selected,minimal_restore=dict(recommended='Frozen v7 CNN/GRU entirely brain-side; continuous actual released RGB-derived rawturn replaces experimental cue1, then brain16->DN2->source neck->body neural decoder. Finite neural calibration required before treating continuousvisual range as equivalent to discretecue training.',alternative='Use actual orderedCNN16 or visualh32 as richer braininput only with explicit modelinput dimension change and new source-preserving preflight/training.',prohibited_actor_inputs=['stimulus condition/category','used_stimulus_code','target_center_xyz','world target location','capture metadata labels','future images or unreleased frames','desired joint action or motor hidden state'],metadata_role='renderer construction, causal timestamp checks, offline labels and evaluation only; never forwarding category/position into neural actor',visual_brain_memory_if_preserved='visual32+bridgebrain16=48 brain-side recurrent states; with motor256 total304, cache16 and AN1/DN2 buffers separate',neural_timing='50Hz capture atv releasesv+.020;5msvisual recurrentupdatesusinglatestreleasedfeaturecache;brain16decision consumesreleasedAN;DN2realidentitytransportreleasesafter2msandnext5msmotor decision consumes. No reset atstimuluschanges/handover.',neck_dynamics_limit='Same-axon engineering delay, not reconstructed native electrophysiology; eyes are brain-side direct sensory path and need no fabricated neck reroute.'),hashes_of_explicitly_reviewed_files=evidence,all_checked_file_sha256=dict(sorted(cache.items())),models_deserialized=0,arrays_loaded=0,array_headers_only=True,NN_forward_calls=0,Core_constructions=0,training_updates=0,physics_steps=0,network_access=False,limitations=['Old policy is bounded red far-landmark orientation only, not generic navigation or biological compound eye','No new restored-vision runtime, training, body loop, olfactory fusion or electrodes executed by this audit','Prior functional PASS cannot be inherited as new real-neck closedloop acceptance','Archive stimuli labels and renderer target coordinates must never replace actualRGB in brain input'])
 if not report['all_checks_passed']:raise ValueError('Audit rejected '+str([k for k,v in checks.items() if not v]))
 with (ROOT/'NEURAL_VISUAL_SOURCE_REVIEW.json').open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps(dict(status=report['status'],checks=len(checks),all_passed=True,files_hashed=len(cache),archive_conditions=len(selected),NN_forward_calls=0,arrays_loaded=0,headers_only=True),indent=2))
if __name__=='__main__':main()
