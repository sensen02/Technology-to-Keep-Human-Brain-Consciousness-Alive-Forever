"""Once-only final500 actualRGB long-history validation and honest visual controls."""
from data import *
from model import production
def metrics_flags(y,target,meta,time):
 motif=meta['motif'];flags={};metrics=dict(action_MSE=float(np.mean((y-target)**2)),turn_RMS=float(np.sqrt(np.mean(y*y))),final_halfsecond_RMS=float(np.sqrt(np.mean(y[time>=9.1]**2))),signed_mean=float(y.mean()),peak_abs_turn=float(abs(y).max()))
 def region(a,b):return (time>=a-1e-9)&(time<b-1e-9)
 if motif=='empty':flags['empty_RMS']=metrics['turn_RMS']<=.02
 elif motif=='center':flags['center_RMS']=metrics['turn_RMS']<=.05
 elif motif=='occluder_only':flags['occluder_only_RMS']=metrics['turn_RMS']<=.05
 elif motif=='fixed_left_while_world_flips':metrics['fixedframe_not_required_zero']=True
 else:
  sign=1 if motif.startswith('left') else -1;on=meta['released_on_s'];off=meta['released_off_s'];switch=meta['released_switch_s'];end=off if off is not None else switch;plateau=region(end-.5,end)&(sign*target>.05);n=int(plateau.sum());frac=None if n==0 else float(np.mean(sign*y[plateau]>0));metrics.update(first_plateau_samples=n,first_sign_fraction=frac);flags['first_sign_fraction']=n>=10 and frac>=.9
  response=(time>=on-1e-9)&(time<on+2.-1e-9)&(sign*y>0)&(sign*target>.05);latency=None if not response.any() else float(time[np.flatnonzero(response)[0]]-on);metrics['first_response_from_released_visible_frame_s']=latency;flags['response_within2s']=latency is not None and latency<=2.
  if off is not None:
   rec=region(off+.5,off+1.);v=y[rec];rms=float(np.sqrt(np.mean(v*v)));metrics.update(released_off_s=off,offset_RMS=rms,offset_signed_mean=float(v.mean()),offset_std=float(v.std()),offset_peak=float(abs(v).max()));flags['offset_RMS']=rms<=.05
  else:
   mask=region(9.1,9.6)&(-sign*target>.05);n=int(mask.sum());frac=None if not n else float(np.mean(-sign*y[mask]>0));metrics['second_sign_fraction']=frac;flags['second_sign_fraction']=n>=10 and frac>=.9
 return metrics,flags

def main():
 cpu1();frozen();out=HERE/'VALIDATION'
 if out.exists():raise FileExistsError('once-only frozen finalvalidation')
 ck=HERE/'TRAIN/checkpoint_000500.pt';saved=read(HERE/'TRAIN/VERIFY_000500.json')
 if not saved['all_passed'] or saved['sha256']!=sha(ck):raise ValueError('independent finalsaved proof required')
 m,p=load_checked(ck,sha(ck));rgb,manifest=library();rows=[];groups={}
 with torch.no_grad():
  for name in ('validation','controls'):
   d=arrays(name);metadata=read(HERE/'DATASET.json')['splits'][name]['rows'];turns=[];latents=[];states=[];swap=[]
   for j,meta in enumerate(metadata):
    x,_,_=batch(rgb,d,[j]);t,l,h=production(m,x);y=t[:,0].numpy().copy();metrics,flags=metrics_flags(y,d['target'][:,j],meta,d['decision_time']);turns.append(y);latents.append(l[:,0].numpy().copy());states.append(h[:,0].numpy().copy())
    # Eye-only swap is a diagnostic, not assumed to be a full scene reflection.
    st,_,_=production(m,x.flip(2));swap.append(st[:,0].numpy().copy());metrics['eyeswap_original_direction_fraction']=float(np.mean(np.sign(st[:,0].numpy()[abs(d['target'][:,j])>.05])==np.sign(d['target'][:,j][abs(d['target'][:,j])>.05]))) if np.any(abs(d['target'][:,j])>.05) else None
    required=meta['motif']!='fixed_left_while_world_flips';rows.append(dict(split=name,index=j,config=meta,metrics=metrics,flags=flags,functional_required=required,all_targets_pass=all(flags.values()) if required else None,eye_swap_not_full_reflection=True))
   groups[name]=dict(turn=np.stack(turns,1),latent=np.stack(latents,1),hidden=np.stack(states,1),eyeswap_turn=np.stack(swap,1),target=d['target'],time=d['decision_time'],frame_ids=d['frame_ids'],used_capture_index=d['used_capture_index'])
 out.mkdir();art={}
 for name,g in groups.items():
  path=out/(name+'_OUTPUTS.npz');np.savez_compressed(path,**g);art[name]=dict(path=str(path),sha256=sha(path))
 passes=all(r['all_targets_pass'] is True for r in rows if r['functional_required']);result=dict(status='OFFLINE_VISUAL_TARGETS_PASS' if passes else 'OFFLINE_VISUAL_TARGETS_FAILED',all_targets_pass=passes,rows=rows,checkpoint=str(ck),checkpoint_sha256=sha(ck),identity=m.identity(),outputs=art,source_hashes={str(v):sha(v) for v in (HERE/'evaluate.py',HERE/'DATASET.json',HERE/'FROZEN.json',HERE/'TRAIN/VERIFY_000500.json',LIB/'LIBRARY.json',LIB/'VERIFY.json')},visual_clamp_only=True,dynamic_navigation_qualified=False,body_response_qualified=False,physical_steps=0,Core_constructions=0,motor_model_loads=0,olfaction_connected=False,fixedframe_and_eye_swap_descriptive_not_gates=True,replicates_per_configuration=1,correlated_frames_not_independent_replicates=True)
 save(out/'RESULT.json',result);guard();print(json.dumps(dict(all_targets_pass=passes,rows=[dict(split=r['split'],index=r['index'],motif=r['config']['motif'],flags=r['flags'],metrics=r['metrics']) for r in rows]),ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
