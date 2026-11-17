"""Matched stimulus functional effects, separately from raw motor engineering gates."""
from runtime import *
def main():
 verify_frozen();r0=read(ROOT/'trials/B_noodor/RESULT.json');g0=arrays(ROOT/'trials/B_noodor/geometry_1ms.npz');y0=np.unwrap(g0['thorax_yaw']);times=g0['time'];neutral=arrays(MOTOR_DIR/'EVAL_neutral/geometry_1ms.npz');yn=np.unwrap(neutral['thorax_yaw']);results=[];source_hashes={};pr=read(ROOT/'PREREG.json')
 for trial in B_IDS:
  folder=ROOT/'trials'/trial;r=read(folder/'RESULT.json');audit=read(folder/'VERIFY.json');assert audit['integrity_all_pass'] and all(audit['checks'].values()) and sha(folder/'RESULT.json')==audit['result_sha256']
  d=arrays(folder/'neural_decisions.npz');g=arrays(folder/'geometry_1ms.npz');yaw=np.unwrap(g['thorax_yaw']);e=d['time'][960:]-5.8;z=d['receptors'][960:];turn=d['raw_turn'][960:];flags={};metrics={}
  def window(a,b):return (e>=a-1e-8)&(e<b-1e-8)
  def signfraction(a,b,sign):
   mask=window(a,b)&(np.mean(z,axis=1)>=.05)&(sign*(z[:,0]-z[:,1])>=.001);n=int(mask.sum());return (None if n==0 else float(np.mean(sign*turn[mask]>0))),n
  stable=basic_functional(r['autonomous'],r['autonomous_cycles'],r['native']);flags['stable_body']=stable
  if trial=='B_noodor':
   metrics['turn_RMS']=float(np.sqrt(np.mean(turn**2)));metrics['extra_yaw_vs_archived_neutral_deg']=float(np.degrees(delta_at(times,yaw,0,9.6)-delta_at(times,yn,0,9.6)));flags['noodor_turn_RMS']=metrics['turn_RMS']<=.02;flags['noodor_extra_yaw']=abs(metrics['extra_yaw_vs_archived_neutral_deg'])<=2
  elif trial=='B_equal':
   metrics['extra_yaw_vs_matched_noodor_deg']=float(np.degrees(delta_at(times,yaw,0,9.6)-delta_at(times,y0,0,9.6)));flags['equal_extra_yaw']=abs(metrics['extra_yaw_vs_matched_noodor_deg'])<=2
  else:
   first=1 if trial in ('B_left_pulse','B_left_to_right') else -1
   a,b=(2.6,3.6) if 'pulse' in trial else (3.8,4.8);fraction,n=signfraction(a,b,first);metrics['first_plateau_correct_sign_fraction']=fraction;metrics['first_plateau_qualified_samples']=n;flags['first_plateau_sign']=n>=10 and fraction>=.9
   mask=window(.6,2.6)&(first*turn>0)&(np.mean(z,axis=1)>=.05)&(first*(z[:,0]-z[:,1])>=.001)
   metrics['first_correct_response_elapsed_s']=float(e[np.flatnonzero(mask)[0]]) if np.any(mask) else None;flags['response_within2s']=bool(np.any(mask))
   if 'pulse' in trial:
    effect=float(np.degrees(delta_at(times,yaw,0,4.)-delta_at(times,y0,0,4.)));metrics['body_extra_yaw_at4s_deg']=effect;flags['body_signed_effect1deg']=first*effect>=1
   else:
    fraction,n=signfraction(8.6,9.6,-first);metrics['second_plateau_correct_sign_fraction']=fraction;metrics['second_plateau_qualified_samples']=n;flags['second_plateau_sign']=n>=10 and fraction>=.9
    effect=float(np.degrees(delta_at(times,yaw,4.8,7.8)-delta_at(times,y0,4.8,7.8)));metrics['body_extra_yaw_after_switch3s_deg']=effect;flags['reversal_body_effect']=(-first)*effect>=.25
  if trial in ('B_equal','B_left_pulse','B_right_pulse'):
   mask=window(4.1,4.6);metrics['offset_turn_RMS_elapsed4_1_to4_6']=float(np.sqrt(np.mean(turn[mask]**2)));flags['command_recovers_after_off']=metrics['offset_turn_RMS_elapsed4_1_to4_6']<=.05
  metrics['fullANN_actual_yaw_deg']=float(np.degrees(r['yaw_change_rad']));metrics['slip_ratio_max']=r['autonomous']['slip_ratio_max']
  results.append(dict(trial=trial,functional_pass=all(flags.values()),functional_flags=flags,metrics=metrics,original_engineering_pass=r['engineering_pass'],strict_endpoint_qualified=r['strict_endpoint_qualified'],result=str(folder/'RESULT.json'),audit=str(folder/'VERIFY.json'),neural_trace=str(folder/'neural_decisions.npz'),geometry_trace=str(folder/'geometry_1ms.npz')))
  for path in (folder/'RESULT.json',folder/'VERIFY.json',folder/'neural_decisions.npz',folder/'geometry_1ms.npz'):source_hashes[str(path)]=sha(path)
 save_new(ROOT/'B_FUNCTIONAL_SUMMARY.json',dict(status='B_COMPLETE_FUNCTIONAL_TARGETS_PASS' if all(r['functional_pass'] for r in results) else 'B_COMPLETE_WITH_FAILED_FUNCTIONAL_TARGETS',all_functional_pass=all(r['functional_pass'] for r in results),rows=results,source_hashes=source_hashes,source_kind='CONCENTRATION_CLAMP_NOT_NAVIGATION',no_training=True,food_success=False,independent_trials=6,correlated_times_not_independent_replicates=True,offset_convention='off beginsANN3.6,envelopeends3.8; RMS4.1<=ANNelapsed<4.6',video_status='PENDING_PARENT_RENDER'))
 print(json.dumps({r['trial']:dict(functional_pass=r['functional_pass'],metrics=r['metrics']) for r in results},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
