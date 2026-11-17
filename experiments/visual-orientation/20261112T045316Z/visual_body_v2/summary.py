"""Normal6functional and2predeclared visual controls separately, noposthoc gatechanges."""
from runtime import *
from adapter import namespace

def main():
 verify();import numpy as np
 namespace('summary.py')['main']();normal=read(ROOT/'NORMAL6_SUMMARY.json');controls=[];proof=dict(normal['source_hashes'])
 for condition in CONTROL_IDS:
  folder=ROOT/'trials'/condition;r=read(folder/'RESULT.json');a=read(folder/'VERIFY.json')
  if not a['integrity_all_pass'] or not all(a['checks'].values()) or a['result_sha256']!=sha(folder/'RESULT.json'):raise ValueError('controlfullindependentauditrequired')
  d=arrays(folder/'neural_decisions.npz');rgb=arrays(folder/'capture_rgb.npz')['rgb'];cap=read(folder/'CAPTURES.json');e=d['time'][960:]-5.8;ids=d['image_capture_id'][960:];flags=dict(stable_body=r['functional_stable_body']);metrics={}
  if condition=='V_left_occluded':
   eligible=[]
   for j,(time_s,fid) in enumerate(zip(e,ids)):
    c=cap[int(fid)]
    if 3.6-1e-8<=time_s<=3.82+1e-8 and 3.6-1e-10<=c['capture_time_s']-5.8<=3.8+1e-10 and c['counterfactual_available'] and c['occlusion_exact']:eligible.append(j)
   origin=None if not eligible else float(e[eligible[0]]);mask=np.zeros(len(e),bool) if origin is None else (e>=origin+.5-1e-8)&(e<origin+1.-1e-8);visibleclear=bool(mask.any()) and all(cap[int(fid)]['counterfactual_available'] and cap[int(fid)]['occlusion_exact'] for fid in ids[mask]);rms=None if not mask.any() else float(np.sqrt(np.mean(d['raw_turn'][960:][mask]**2)))
   metrics.update(first_released_complete_occlusion_s=origin,offset_turn_RMS=rms,occlusion_returns_visible_samples_in_window=None if not mask.any() else sum(not cap[int(fid)]['occlusion_exact'] for fid in ids[mask]));flags.update(complete_occlusion_precondition_by3_82=origin is not None,continuous_sampled_occlusion_in_recovery=visibleclear,occlusion_recovery=rms is not None and rms<=.05)
  else:
   freeze=e>=4.8-1e-8;actorids=d['image_capture_id'][960:];rawids=d['raw_released_image_capture_id'][960:];held=d['held_features'][960:];reference=int(actorids[np.flatnonzero(freeze)[0]]);feature=held[np.flatnonzero(freeze)[0]];changed=any(not np.array_equal(rgb[int(fid)],rgb[reference]) for fid in rawids[freeze]);mae=max(float(np.abs(rgb[int(fid)].astype(float)-rgb[reference]).mean()) for fid in rawids[freeze]);metrics.update(frozen_actor_capture_id=reference,frozen_capture_ANN_s=cap[reference]['capture_time_s']-5.8,actual_stream_max_MAE_vs_frozen=mae,turnzero_or_right_not_required=True)
   flags.update(actor_captureANN4_78=abs(cap[reference]['capture_time_s']-5.8-4.78)<=1e-10 and cap[reference]['stimulus']=='left',actorimageheld_exact=bool(np.all(actorids[freeze]==reference)),featuresheld_exact=np.array_equal(held[freeze],np.broadcast_to(feature,held[freeze].shape)),realcamera_continues=np.unique(rawids[freeze]).size>1 and changed,worldswitch_right=any(c['stimulus']=='right' for c in cap[530:]),actual_modelreplay_audited=a['offline_RGB_CNN_GRU_motor_replay'])
  controls.append(dict(condition=condition,control_pass=all(flags.values()),flags=flags,metrics=metrics,counts_as_new_closed_loop_success=False,original_engineering_pass=r['engineering_pass'],slip=r['autonomous']['slip_ratio_max'],result=str(folder/'RESULT.json'),audit=str(folder/'VERIFY.json')))
  for n in ('RESULT.json','VERIFY.json','capture_rgb.npz','CAPTURES.json','neural_decisions.npz'):proof[str(folder/n)]=sha(folder/n)
  if condition=='V_left_occluded':proof[r['occlusion_counterfactual']['path']]=r['occlusion_counterfactual']['sha256']
 result=dict(kind='VISUAL_BODY_OCCLUSION_FIXEDFRAME_V2',normal6_all_functional_pass=normal['all_functional_pass'],control2_all_pass=all(c['control_pass'] for c in controls),normal6=normal['rows'],control2=controls,all_declared_targets_pass=normal['all_functional_pass'] and all(c['control_pass'] for c in controls),source_hashes=proof,unique_physical_trials=8,normal_and_controls_separate=True,fixedframe_not_required_turnzero=True,olfaction_connected=False,staticpretrain_not_dynamicbodyqualification=True)
 save(ROOT/'VISUAL_BODY_SUMMARY.json',result);verify();print(json.dumps(dict(normal6_all_functional_pass=result['normal6_all_functional_pass'],control2_all_pass=result['control2_all_pass'],controls=controls),indent=2))
if __name__=='__main__':main()
