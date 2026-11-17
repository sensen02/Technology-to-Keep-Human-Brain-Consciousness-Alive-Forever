"""Parent explicitflag actual-source FIFO archived observation interventions; no training.
This is conditional offline communication replay, not changed-body physical causality.
"""
from pathlib import Path
import argparse,sys,importlib.util,json,hashlib,time,os,resource,math
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;CAL=ROOT/'neural_calibration';PORTS=ROOT/'dnp09_interface'
MODES=('baseline','AN_cut_from3s','DN_cut_from3s','AN_allzero','brain_cue_allzero')

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def fresh(p,obj):
 with Path(p).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def rollout(model,body111,cue,mode,transport_module,api,binding_path,binding_sha,torch):
 if mode not in MODES or body111.shape!=(1920,1,111) or cue.shape!=(1920,1,1):raise ValueError('Exact mode/archived batch1 fullhistory required')
 source=transport_module.SourceTransport(binding_path,binding_sha);h=body111.new_zeros((1,16));held={'DN':body111.new_zeros((1,2)),'AN':body111.new_zeros((1,1))};held_packets={};rows={key:[] for key in ('turn','DN','AN','brain_hidden','held_AN','held_DN','sent_AN','brain_cue')};reconstruction=[];consumption=[];cut_direction=None;cut_started=False;cut_hbefore=None;cut_hafter=None
 if mode=='AN_cut_from3s':cut_direction='AN'
 if mode=='DN_cut_from3s':cut_direction='DN'
 with torch.no_grad():
  for k in range(1920):
   if source.step!=5*k:raise ValueError('Actual source queue/decision clock mismatch')
   if cut_direction and source.step==3000:
    cut_hbefore=h.clone();source.set_cut(cut_direction,True);held[cut_direction]=body111.new_zeros((1,len(source.ports[cut_direction])));held_packets.pop(cut_direction,None);cut_started=True;cut_hafter=h.clone()
   for direction,packet in held_packets.items():
    used=dict(packet,consume_step=source.step);api.validate_delivery(used,source.ports[direction],direction,source.digest,source.step);consumption.append(dict(direction=direction,capture_step=packet['capture_step'],queue_release_step=packet['release_step'],neural_consume_step=source.step))
   current_cue=cue[k] if mode!='brain_cue_allzero' else torch.zeros_like(cue[k]);encoded=model.feedback(model.normalize(body111[k]));sent_an=torch.zeros_like(encoded) if mode=='AN_allzero' else encoded
   dn,h=model.brain_step(held['AN'],current_cue,h);turn=model.decoder(held['DN']);rec=model.reconstruction(encoded);reconstruction.append(rec.clone())
   for key,value in (('turn',turn),('DN',dn),('AN',encoded),('brain_hidden',h),('held_AN',held['AN']),('held_DN',held['DN']),('sent_AN',sent_an),('brain_cue',current_cue)):rows[key].append(value.clone())
   for direction,value in (('DN',dn),('AN',sent_an)):
    if cut_started and direction==cut_direction:continue
    source.send({'from':'brain' if direction=='DN' else 'other','to':'neck','direction':direction,'root_ids':source.ports[direction],'values':value[0].tolist(),'capture_step':source.step,'source_bindings_sha256':source.digest})
   for ms in range(5):
    for packet in source.advance():held[packet['direction']]=body111.new_tensor([packet['values']]);held_packets[packet['direction']]=dict(packet)
 proof=source.verify_final();journal=source.journal();proof.update(mode=mode,cut_direction=cut_direction,receiver_baseline_immediately_zero=cut_direction is not None,cut_step=3000 if cut_direction else None,cut_brain_state_exact=None if not cut_direction else bool(torch.equal(cut_hbefore,cut_hafter)),cut_inflight_packets_dropped=sum(x['event']=='drop_inflight' for x in journal),queue_journal=journal,neural_consumption=consumption,initial_delivered_AN_DN_zero=True,no_brain_reset=True,observations_are_fixed_archived_exogenous=True)
 if proof['actual_steps']!=9600 or any(x['queue_release_step']!=x['capture_step']+2 or x['neural_consume_step']!=x['capture_step']+5 for x in consumption):raise ValueError('Actual fullhistory causal queue proof failed')
 if cut_direction and (not cut_started or not proof['cut_brain_state_exact']):raise ValueError('Cut missing or brain memory reset')
 if cut_direction and any(e['event']=='enqueue' and e['direction']==cut_direction and e['capture_step']>=3000 for e in journal):raise ValueError('Cut direction still receiving packets')
 return {k:torch.stack(v) for k,v in rows.items()},torch.stack(reconstruction),proof

def summarize(np,row,teacher,reconstruction,normalized,baseline,baseline_rec):
 turn=row['turn'];active=np.abs(teacher)>.1;post=slice(600,None);delta=turn-baseline['turn'];metrics=dict(turn_RMS=float(np.sqrt(np.mean(turn**2))),turn_mean=float(turn.mean()),teacher_RMSE=float(np.sqrt(np.mean((turn-teacher)**2))),active_teacher_sign_fraction=None if not active.any() else float(np.mean(np.sign(turn[active])==np.sign(teacher[active]))),paired_turn_difference_RMS=float(np.sqrt(np.mean(delta**2))),paired_turn_difference_mean=float(delta.mean()),paired_turn_difference_post3s_RMS=float(np.sqrt(np.mean(delta[post]**2))),paired_turn_difference_post3s_mean=float(delta[post].mean()),feedback_normalized_reconstruction_RMSE=float(np.sqrt(np.mean((reconstruction-normalized)**2))),reconstruction_max_abs_difference_vs_baseline=float(np.max(abs(reconstruction-baseline_rec))))
 for key in ('DN','AN','brain_hidden','held_AN','held_DN'):
  diff=row[key]-baseline[key];metrics[key+'_paired_RMS_difference']=float(np.sqrt(np.mean(diff**2)));metrics[key+'_paired_max_abs_difference']=float(np.max(abs(diff)))
 return metrics

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--execute-parent-interventions',action='store_true');a=parser.parse_args()
 if not a.execute_parent_interventions:print('CODE_ONLY no arrays/model/sourcequeue execution');return
 if any((HERE/name).exists() for name in ('INPUTS_BEFORE.json','RESULT.json','OUTPUTS')):raise FileExistsError('Inspect outputs before any rerun')
 for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
  if os.environ.get(key)!='1':raise ValueError(key+'=1 required')
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});pr=read(HERE/'PREREG.json');checkpoint=Path(pr['checkpoint']);binding_path=Path(pr['source_binding'])
 if sha(checkpoint)!=pr['checkpoint_sha256'] or sha(binding_path)!=pr['source_binding_sha256']:raise ValueError('Final200 or sourcebinding identity mismatch')
 evaluation=read(CAL/'EVALUATION/RESULT.json');eaudit=read(CAL/'EVALUATION/VERIFY.json');saved=read(CAL/'TRAIN/VERIFY_000200.json')
 if not eaudit['integrity_all_pass'] or not all(eaudit['checks'].values()) or eaudit['result_sha256']!=sha(CAL/'EVALUATION/RESULT.json') or saved['sha256']!=pr['checkpoint_sha256'] or not saved['all_passed'] or not all(saved['checks'].values()):raise ValueError('Independent prior evaluation/storage proof required')
 if evaluation['checkpoint_sha256']!=pr['checkpoint_sha256']:raise ValueError('Final eval wrong checkpoint')
 frozen=read(CAL/'FROZEN.json');hashes=dict(frozen['source_hashes']);hashes.update(frozen['data_hashes'])
 for p in (HERE/'PREREG.json',HERE/'run_interventions.py',HERE/'codecheck.py',HERE/'CODEONLY_CHECK.json',checkpoint,binding_path,CAL/'EVALUATION/RESULT.json',CAL/'EVALUATION/VERIFY.json',CAL/'TRAIN/VERIFY_000200.json'):hashes[str(p)]=sha(p)
 hashes.update(evaluation['output_hashes'])
 for path,h in hashes.items():
  if sha(path)!=h:raise ValueError('Input drift '+path)
 fresh(HERE/'INPUTS_BEFORE.json',hashes);priorpath=list(sys.path);sentinel=object();aliases={k:sys.modules.get(k,sentinel) for k in ('common','model','data','api')};start=time.monotonic()
 try:
  sys.path.insert(0,str(CAL));io=module('_offline_intervention_frozen_common',CAL/'common.py');sys.modules['common']=io;torch,np=io.backend();architecture=module('_offline_intervention_model',CAL/'model.py');sys.modules['model']=architecture;data=module('_offline_intervention_data',CAL/'data.py');sys.modules['data']=data
  api=module('_offline_intervention_port_api',PORTS/'api.py');sys.modules['api']=api;transport_module=module('_offline_intervention_source_transport',PORTS/'source_transport.py');model,payload=io.load_checkpoint(checkpoint,pr['checkpoint_sha256']);model.requires_grad_(False);model.eval()
  def tensor_hashes():return {k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in model.state_dict().items()}
  model_before=tensor_hashes();outputs=HERE/'OUTPUTS';outputs.mkdir(exist_ok=False);rows=[];checks={};baseline_errors={};artifact_hashes={};expected=0
  with torch.no_grad():decoder_zero_output=float(model.decoder(torch.zeros((1,2),dtype=torch.float64))[0,0])
  for split,names in [('train',io.TRAIN),('development',io.DEV)]:
   d=data.arrays(split);x,cue,teacher=data.tensors(d)
   for j,condition in enumerate(names):
    body=x[:,j:j+1];instruction=cue[:,j:j+1];truth=teacher[:,j:j+1].numpy();normalized=model.normalize(body).numpy();oldrow=next(v for v in evaluation['rows'] if v['condition']==condition)
    baserow=None;base_rec=None
    for mode in MODES:
     tensors,rec,proof=rollout(model,body,instruction,mode,transport_module,api,binding_path,pr['source_binding_sha256'],torch);out={k:v.numpy() for k,v in tensors.items()};reconstruction=rec.numpy()
     if mode=='baseline':
      baserow=out;base_rec=reconstruction
      with np.load(oldrow['artifact'],allow_pickle=False) as z:
       errs={key:float(np.max(abs(out[key]-z[key]))) for key in ('turn','DN','AN','brain_hidden')};errs['reconstruction']=float(np.max(abs(reconstruction-z['reconstruction'])))
      baseline_errors[condition]=errs;checks[condition+'_baseline_matches_prior_final200']=max(errs.values())<=1e-10
     metrics=summarize(np,out,truth,reconstruction,normalized,baserow,base_rec)
     checks[condition+'_'+mode+'_queue9600and_sources']=proof['actual_steps']==9600 and proof['source_files_unchanged']
     checks[condition+'_'+mode+'_bodyfeedback_encoder_unchanged']=np.array_equal(out['AN'],baserow['AN']) and np.array_equal(reconstruction,base_rec)
     if mode in ('AN_cut_from3s','DN_cut_from3s'):
      checks[condition+'_'+mode+'_prefix_exact_before3s']=all(np.array_equal(out[key][:600],baserow[key][:600]) for key in ('turn','DN','AN','brain_hidden','held_AN','held_DN'))
      key='held_AN' if mode=='AN_cut_from3s' else 'held_DN';checks[condition+'_'+mode+'_receiver_baseline_zero']=bool(np.all(out[key][600:]==0))
     if mode=='AN_allzero':checks[condition+'_ANallzeroactualtransmitted']=bool(np.all(out['sent_AN']==0) and np.all(out['held_AN']==0))
     if mode=='brain_cue_allzero':checks[condition+'_cuezero_braininput']=bool(np.all(out['brain_cue']==0))
     artifact=outputs/(condition+'__'+mode+'.npz');np.savez_compressed(artifact,**out);artifact_hashes[str(artifact)]=sha(artifact);journal=outputs/(condition+'__'+mode+'__QUEUE.json');fresh(journal,proof);artifact_hashes[str(journal)]=sha(journal);expected+=1
     rows.append(dict(split=split,condition=condition,mode=mode,metrics=metrics,artifact=str(artifact),queue_proof=str(journal),cut_step=proof['cut_step'],cut_inflight_packets_dropped=proof['cut_inflight_packets_dropped'],cut_brain_state_exact=proof['cut_brain_state_exact'],new_physical_trial=False))
     rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.
     if time.monotonic()-start>1800 or rss>1800:raise RuntimeError('Fixed intervention wall/RSS limit exceeded')
     print(json.dumps(dict(completed=expected,total=30,condition=condition,mode=mode,paired_turn_RMS_delta=metrics['paired_turn_difference_RMS'],seconds=time.monotonic()-start)),flush=True)
  checks['all30actual_sourcequeue_replays']=expected==30;checks['all_model_parameters_and_buffers_unchanged']=tensor_hashes()==model_before;checks['no_grad_or_optimizer_state']=all(not p.requires_grad and p.grad is None for p in model.parameters());checks['no_physical_backend']=not any(k in sys.modules for k in ('physics','mujoco','student_wave','flygym'))
  for path,h in hashes.items():checks['source_unchanged:'+path]=sha(path)==h
  if not all(checks.values()):raise ValueError('Intervention verification failed '+str({k:v for k,v in checks.items() if not v}))
  result=dict(status='THIRTY_MATCHED_OFFLINE_SOURCEQUEUE_INTERVENTIONS_COMPLETE_NO_PHYSICAL_CAUSAL_CLAIM',integrity_all_pass=True,checks=checks,checkpoint_sha256=pr['checkpoint_sha256'],source_binding_sha256=pr['source_binding_sha256'],baseline_prior_output_errors=baseline_errors,rows=rows,output_hashes=artifact_hashes,source_hashes=hashes,decoder_delivered_DN_zero_output=decoder_zero_output,DNcut_does_not_imply_zero_turn=True,brain_hidden_not_reset=True,source_queue_dt_s=.001,source_queue_delay_s=.002,neural_consumer_dt_s=.005,engineering_receiver_baseline_proxy_explicit=True,normalized_reconstruction_input_unchanged=True,model_hashes_before=model_before,model_hashes_after=tensor_hashes(),actual_replays=30,sourcequeue_ticks_per_replay=9600,seconds=time.monotonic()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,training_updates=0,optimizer_steps=0,new_physics_steps=0,Core_constructions=0,observations_fixed_archived_exogenous=True,closed_loop_body_causality_verified=False,native_DNp09_function_established=False,native_AN_joint_tuning_established=False,full_neck_admitted=False,interpretation='Communication sensitivities conditional on fixed teacher-body observations; no claim interventions caused these body trajectories. Encoder reconstruction error unchanged is not proof brain uses AN.')
  fresh(HERE/'RESULT.json',result);print(json.dumps(dict(status=result['status'],integrity_all_pass=True,checks=len(checks),decoder_zero_output=decoder_zero_output,seconds=result['seconds']),indent=2))
 finally:
  sys.path[:]=priorpath
  for name,previous in aliases.items():
   if previous is sentinel:sys.modules.pop(name,None)
   else:sys.modules[name]=previous
if __name__=='__main__':main()
