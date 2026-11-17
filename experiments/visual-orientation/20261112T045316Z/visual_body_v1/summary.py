"""Predeclared actual visualbody function gates, separately from strict motor engineering."""
from runtime import *
def main():
 verify();import numpy as np
 empty=ROOT/'trials/V_empty';g0=arrays(empty/'geometry_1ms.npz');y0=np.unwrap(g0['thorax_yaw']);neutral=arrays(NEUTRAL/'trace_1ms.npz');q=neutral['qpos'][:,3:7];ny=np.unwrap(np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2)));rows=[];proof={}
 def delta(y,a,b):
  times=g0['time'];return float(y[int(np.argmin(abs(times-(5.8+b))))]-y[int(np.argmin(abs(times-(5.8+a))))])
 for condition in IDS:
  folder=ROOT/'trials'/condition;r=read(folder/'RESULT.json');a=read(folder/'VERIFY.json')
  if not a['integrity_all_pass'] or not all(a['checks'].values()) or a['result_sha256']!=sha(folder/'RESULT.json'):raise ValueError('full independent visualbody audit required')
  d=arrays(folder/'neural_decisions.npz');g=arrays(folder/'geometry_1ms.npz');e=d['time'][960:]-5.8;turn=d['raw_turn'][960:];code=d['used_stimulus_code'][960:];yaw=np.unwrap(g['thorax_yaw']);flags=dict(stable_body=r['functional_stable_body']);metrics={}
  if not np.array_equal(g['time'],g0['time']):raise ValueError('matchedempty clock must identical')
  def window(a,b):return (e>=a-1e-8)&(e<b-1e-8)
  if condition=='V_empty':
   rms=float(np.sqrt(np.mean(turn**2)));extra=float(np.degrees(delta(yaw,0,9.6)-delta(ny,0,9.6)));metrics.update(turn_RMS=rms,extra_yaw_vs_archived_neutral_deg=extra);flags.update(empty_RMS=rms<=.02,empty_extra_yaw=abs(extra)<=2.)
  elif condition=='V_center':
   rms=float(np.sqrt(np.mean(turn**2)));extra=float(np.degrees(delta(yaw,0,9.6)-delta(y0,0,9.6)));metrics.update(turn_RMS=rms,extra_yaw_vs_empty_deg=extra);flags.update(center_RMS=rms<=.05,center_extra_yaw=abs(extra)<=2.)
  else:
   first=1 if condition in ('V_left_pulse','V_left_to_right') else -1;expected=1 if first>0 else 2;pulse='pulse' in condition
   w=window(2.6,3.6) if pulse else window(3.8,4.8);mask=w&(code==expected);count=int(mask.sum());fraction=None if count==0 else float(np.mean(first*turn[mask]>0));flags['first_plateau_sign']=count>=10 and fraction>=.9;metrics.update(first_plateau_samples=count,first_plateau_sign_fraction=fraction)
   onset=np.flatnonzero(code==expected);origin=None if len(onset)==0 else float(e[onset[0]]);mask=np.zeros(len(e),dtype=bool) if origin is None else (code==expected)&window(origin,origin+2.)&(first*turn>0);response=None if not mask.any() else float(e[np.flatnonzero(mask)[0]])
   metrics.update(released_visible_on_s=origin,response_latency_s=None if response is None else response-origin);flags['response_within2s']=response is not None
   if pulse:
    effect=float(np.degrees(delta(yaw,0,4.)-delta(y0,0,4.)));metrics['body_effect_at4s_vs_empty_deg']=effect;flags['body_effect1deg']=first*effect>=1.
   else:
    mask=window(8.6,9.6)&(code==(2 if expected==1 else 1));n=int(mask.sum());fraction=None if not n else float(np.mean(-first*turn[mask]>0));flags['second_plateau_sign']=n>=10 and fraction>=.9;metrics['second_plateau_sign_fraction']=fraction;effect=float(np.degrees(delta(yaw,4.8,7.8)-delta(y0,4.8,7.8)));metrics['body_effect_after_switch3s_vs_empty_deg']=effect;flags['reversal_body_effect']=(-first)*effect>=.25
  if condition in ('V_center','V_left_pulse','V_right_pulse'):
   # First released empty exposure after actual disappearance; scene labels are diagnostics only.
   mask=(e>=3.6)&(code==0);off=None if not mask.any() else float(e[np.flatnonzero(mask)[0]])
   rec=np.zeros(len(e),bool) if off is None else window(off+.5,off+1.);rms=None if not rec.any() else float(np.sqrt(np.mean(turn[rec]**2)));metrics.update(released_off_s=off,offset_turn_RMS=rms);flags['offset_recovery']=rms is not None and rms<=.05
  rows.append(dict(condition=condition,functional_pass=all(flags.values()),functional_flags=flags,metrics=metrics,original_engineering_pass=r['engineering_pass'],slip=r['autonomous']['slip_ratio_max'],actual_yaw_deg=float(np.degrees(r['yaw_change_rad'])),result=str(folder/'RESULT.json'),audit=str(folder/'VERIFY.json')))
  for name in ('RESULT.json','VERIFY.json','neural_decisions.npz','geometry_1ms.npz','capture_rgb.npz','CAPTURES.json'):proof[str(folder/name)]=sha(folder/name)
 save(ROOT/'VISUAL_BODY_SUMMARY.json',dict(status='VISUAL_BODY_FUNCTIONAL_PASS' if all(row['functional_pass'] for row in rows) else 'VISUAL_BODY_FUNCTIONAL_FAILED_TARGETS',all_functional_pass=all(row['functional_pass'] for row in rows),rows=rows,source_hashes=proof,vision_only=True,olfaction_connected=False,source_kind='WORLD_FIXED_LANDMARK_ACTUAL_MOVING_HEAD_RGB',six_unique_physical_trials=6,replicates_per_condition=1,geometry_labels_not_policy_inputs=True,not_food_ingestion_or_parking=True));verify();print(json.dumps({r['condition']:dict(pass_all=r['functional_pass'],metrics=r['metrics']) for r in rows},indent=2))
if __name__=='__main__':main()
