"""Parent explicitflag fixed-observation/sourcequeue diagnostic; no training or physical backend."""
from pathlib import Path
import argparse,json,hashlib,importlib.util,sys,os,time
HERE=Path(__file__).resolve().parent;SCOPE=HERE.parent;BODY=SCOPE/'sensory_body';CAL=SCOPE/'sensory_fusion_recovery_v1';EXPECTED='db2510d46900644c7f8c7b556b4b892dad2a88e3e37ce13792811916a928d4c6'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def fresh(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--execute-parent',action='store_true');args=ap.parse_args()
 if not args.execute_parent:print('CODE_ONLY no arrays/models imported');return
 for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
  if os.environ.get(k)!='1':raise ValueError(k+'=1required')
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});started=time.monotonic();pr=read(HERE/'PREREG.json');checkpoint=CAL/'TRAIN/checkpoint_000400.pt'
 if pr['fixed_checkpoint_sha256']!=EXPECTED or sha(checkpoint)!=EXPECTED:raise ValueError('Actualfixeddb25identity')
 if any((HERE/n).exists() for n in ('INPUTS_BEFORE.json','RESULT.json','OUTPUTS.npz')):raise FileExistsError('Inspectpriorresultsbeforereplay')
 hashes={str(p):sha(p) for p in [HERE/'PREREG.json',HERE/'diagnose.py',checkpoint,CAL/'FROZEN.json',CAL/'DATASET.json',CAL/'train.npz',CAL/'EVALUATION/RESULT.json',CAL/'EVALUATION/VERIFY.json',BODY/'JOBS.json',BODY/'FROZEN.json',BODY/'INPUT_BINDINGS.json']};hashes.update(read(CAL/'FROZEN.json')['source_hashes']);hashes.update(read(CAL/'FROZEN.json')['data_hashes']);checks={}
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('Failurediagnostic '+k)
 jobs=read(BODY/'JOBS.json');check('all30_original_stages_exit0',len(jobs)==30 and all(v['exit_code']==0 for v in jobs));audit=read(CAL/'EVALUATION/VERIFY.json');check('actualcalibrationindependentaudit',audit['integrity_all_pass'] and all(audit['checks'].values()) and audit['result_sha256']==sha(CAL/'EVALUATION/RESULT.json'))
 for p,h in hashes.items():
  if sha(p)!=h:raise ValueError('Sourcechangedbefore '+p)
 sentinel=object();previous={k:sys.modules.get(k,sentinel) for k in ('common','model','steering_behavior')};paths=list(sys.path)
 try:
  ci=module('_failure_diag_recovery_common',CAL/'common.py');sys.modules['common']=ci;arch=module('_failure_diag_recovery_arch',CAL/'model.py');sys.modules['model']=arch;m,payload=ci.load_checkpoint(checkpoint,EXPECTED,final=True);vision,odor,vp,op=ci.load_sensory()
 finally:
  sys.path[:]=paths
  for k,v in previous.items():
   if v is sentinel:sys.modules.pop(k,None)
   else:sys.modules[k]=v
 import torch,numpy as np
 torch.set_num_threads(1);m.requires_grad_(False);odor.requires_grad_(False);vision.requires_grad_(False);before={'bridge':ci.tensor_hashes(m),'odor':ci.tensor_hashes(odor),'vision':ci.tensor_hashes(vision)};data={};results={};stats_cases=[];outputs={};proofs={}
 def stats(a):
  a=np.asarray(a,dtype=float);return dict(mean=float(a.mean()),RMS=float(np.sqrt(np.mean(a*a))),min=float(a.min()),max=float(a.max()))
 def windows(values):
  return {w['name']:stats(np.asarray(values)[(np.arange(1920)*.005>=w['start_s']-1e-9)&(np.arange(1920)*.005<w['end_s']-1e-9)]) for w in pr['windows']}
 wanted=('time','X238','actor_receptors','receptors_used','concentration','odor_raw_turn','odor_latent','visual_raw_turn','raw_turn','odor_hidden_before','odor_hidden_after','brain_hidden_before','brain_hidden_after','DN_encoded','AN_encoded','DN_held','AN_held','physical_step_ms')
 for name in pr['cases']:
  folder=BODY/'trials'/name;r=read(folder/'RESULT.json');a=read(folder/'VERIFY.json');check(name+'_independentintegrity',a['integrity_all_pass'] and all(a['checks'].values()) and a['result_sha256']==sha(folder/'RESULT.json'));check(name+'_fixedcheckpoint',r['brain_checkpoint_sha256']==EXPECTED and r['odor_checkpoint_sha256']==ci.ODOR_SHA)
  for key in ('source_hashes',):hashes.update(r[key])
  for p in (folder/'RESULT.json',folder/'VERIFY.json',folder/'neural_decisions.npz'):hashes[str(p)]=sha(p)
  check(name+'_actualarchiveSHA',sha(folder/'neural_decisions.npz')==r['neural_decisions']['sha256'])
  with np.load(folder/'neural_decisions.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in wanted}
  check(name+'_3080clock',d['X238'].shape==(3080,238) and np.max(abs(d['time']-np.arange(3080)*.005))<1e-8);data[name]=d;results[name]=r;ann={k:v[1160:] for k,v in d.items()};signal={k:ann[k] for k in ('odor_raw_turn','visual_raw_turn','raw_turn','DN_held','AN_held','actor_receptors','receptors_used','concentration')};signal['odor_hidden_norm']=np.linalg.norm(ann['odor_hidden_after'],axis=1);signal['brain_hidden_norm']=np.linalg.norm(ann['brain_hidden_after'],axis=1)
  missing=[leg for leg,n in r['autonomous_cycles']['complete_per_leg'].items() if n==0];stats_cases.append(dict(condition=name,signals={k:windows(v) for k,v in signal.items()},complete_per_leg=r['autonomous_cycles']['complete_per_leg'],missing_complete_cycle_legs=missing,stable_body=r['functional_stable_body'],original_engineering_pass=r['engineering_pass'],slip=r['autonomous']['slip_ratio_max'],native_nonfoot_peak=r['native']['post_nonfoot_peak_native']))
  # Independent actual frozenolfactory re-run for everyselectedcase, including persistentreceptorcut.
  h=odor.initial_state(1);raws=[];latents=[];hs=[]
  with torch.no_grad():
   for k in range(1920):
    j=k+1160;checkprefix=name+'_odor_before_exact' if k==0 else None
    if not torch.equal(h,torch.from_numpy(d['odor_hidden_before'][j:j+1])):raise ValueError('Actualodorhbefore '+name+' '+str(k))
    latent,h=odor.mean_step(torch.from_numpy(d['actor_receptors'][j:j+1]),h);raw=latent[:,0].tanh();raws.append(float(raw[0]));latents.append(float(latent[0,0]));hs.append(h[0].numpy().copy())
  check(name+'_frozenodorraw_h32_latent_exact',np.array_equal(np.asarray(raws),ann['odor_raw_turn']) and np.array_equal(np.asarray(latents),ann['odor_latent']) and np.array_equal(np.asarray(hs),ann['odor_hidden_after']));outputs[name+'_actual_odorraw_replayed']=np.asarray(raws)
 cut=data['O_left_odorcut'];cutmask=np.arange(3080)*5>=7400;check('actual_receptorcut_zero_fromANN1_6',np.max(abs(cut['actor_receptors'][cutmask]))==0);check('cut_odorh32_continuous',np.array_equal(cut['odor_hidden_before'][1:],cut['odor_hidden_after'][:-1]));check('cut_preANN_odorh0',np.max(abs(cut['odor_hidden_before'][:1161]))==0)
 fresh(HERE/'INPUTS_BEFORE.json',hashes)
 # Fourdeclared sensoryprobes. Actualcut is exact replayalreadycomputed above, no resetting hiddenstate.
 probe_arrays={'cold_zero_receptors':np.zeros((1920,2)),'actual_receptorcut':cut['actor_receptors'][1160:].copy(),'normal_Olefthistory_then_zero':data['O_left']['actor_receptors'][1160:].copy(),'Flefthistory_then_zero':data['F_left_agree']['actor_receptors'][1160:].copy()}
 for key in ('normal_Olefthistory_then_zero','Flefthistory_then_zero'):probe_arrays[key][320:]=0
 odor_probes=[]
 with torch.no_grad():
  for name,receptors in probe_arrays.items():
   h=odor.initial_state(1);raws=[];hs=[]
   for k in range(1920):mean,h=odor.mean_step(torch.from_numpy(receptors[k:k+1]),h);raws.append(float(mean[0,0].tanh()));hs.append(h[0].numpy().copy())
   raws=np.asarray(raws);hs=np.asarray(hs);outputs['odorprobe_'+name]=raws;outputs['odorprobe_'+name+'_h32']=hs;odor_probes.append(dict(mode=name,rawturn_windows=windows(raws),hidden_norm_windows=windows(np.linalg.norm(hs,axis=1)),receptors_zero_after1_6=bool(np.max(abs(receptors[320:]))==0),physical_counterfactual=False))
   if name=='actual_receptorcut':check('actualcut_probe_raw_h_exact',np.array_equal(raws,cut['odor_raw_turn'][1160:]) and np.array_equal(hs,cut['odor_hidden_after'][1160:]))
 # Actual1ms sourcequeue overentire15400ms; newinterventionsonly affect deliveredAN orbrainodor input.
 paths=list(sys.path);previousapi=sys.modules.get('api',sentinel)
 try:
  sys.path.insert(0,str(ci.PORTS));api=module('_failure_diag_source_api',ci.PORTS/'api.py');sys.modules['api']=api;tm=module('_failure_diag_source_transport',ci.PORTS/'source_transport.py')
 finally:
  sys.path[:]=paths
  if previousapi is sentinel:sys.modules.pop('api',None)
  else:sys.modules['api']=previousapi
 bp,b,digest=ci.binding();brain_replays=[]
 for mode in ('baseline','odor_neural_cut','AN_held_cut','both_cuts'):
  queue=tm.SourceTransport(bp,digest);held={'AN':torch.zeros((1,1),dtype=torch.float64),'DN':torch.zeros((1,2),dtype=torch.float64)};h=torch.zeros((1,16),dtype=torch.float64);packets={};an_cut=False;records={k:[] for k in ('turn','brain','DN','AN','DNheld','ANheld','odorbrain')};events=[]
  with torch.no_grad():
   for j in range(3080):
    ms=5*j
    if queue.step!=ms:raise ValueError('Sourceclock')
    if ms==7400 and mode in ('AN_held_cut','both_cuts'):queue.set_cut('AN',True);held['AN'].zero_();packets.pop('AN',None);an_cut=True
    for direction,p in packets.items():api.validate_delivery(dict(p,consume_step=ms),queue.ports[direction],direction,digest,ms);events.append(dict(direction=direction,capture_step=p['capture_step'],queue_release_step=p['release_step'],neural_consume_step=ms))
    x=torch.from_numpy(cut['X238'][j:j+1,ci.MASK]);an=m.feedback(m.normalize(x));od=0. if ms>=7400 and mode in ('odor_neural_cut','both_cuts') else float(cut['odor_raw_turn'][j]);modal=torch.tensor([[float(cut['visual_raw_turn'][j]),od]],dtype=torch.float64);usedAN=held['AN'].clone();usedDN=held['DN'].clone()
    if ms<5800:dn=torch.zeros((1,2),dtype=torch.float64)
    else:dn,h=m.brain_step(usedAN,modal,h)
    turn=0. if ms<5800 else float(m.decoder(usedDN)[0,0]);records['turn'].append(turn);records['brain'].append(h[0].numpy().copy());records['DN'].append(dn[0].numpy().copy());records['AN'].append(an[0].numpy().copy());records['DNheld'].append(usedDN[0].numpy().copy());records['ANheld'].append(usedAN[0].numpy().copy());records['odorbrain'].append(od)
    for direction,value in (('DN',dn),('AN',an)):
     if direction=='AN' and an_cut:continue
     queue.send({'from':'brain' if direction=='DN' else 'other','to':'neck','direction':direction,'root_ids':queue.ports[direction],'values':value[0].tolist(),'capture_step':ms,'source_bindings_sha256':digest})
    for _ in range(5):
     for p in queue.advance():held[p['direction']]=torch.tensor([p['values']],dtype=torch.float64);packets[p['direction']]=dict(p)
  final=queue.verify_final();check(mode+'_source15400_unchanged',final['actual_steps']==15400 and final['source_files_unchanged']);check(mode+'_release2ms_consume5ms',all(e['queue_release_step']==e['capture_step']+2 and e['neural_consume_step']==e['capture_step']+5 for e in events));records={k:np.asarray(v) for k,v in records.items()}
  if mode=='baseline':
   for key,stored in [('turn','raw_turn'),('brain','brain_hidden_after'),('DN','DN_encoded'),('AN','AN_encoded'),('DNheld','DN_held'),('ANheld','AN_held')]:check('baseline_exact_'+key,np.array_equal(records[key],cut[stored]))
   base=records
  else:
   for key in ('turn','brain','DN','AN','DNheld','ANheld'):check(mode+'_prefix_before_cut_exact_'+key,np.array_equal(records[key][:1480],base[key][:1480]))
  for key,value in records.items():outputs['brainprobe_'+mode+'_'+key]=value[1160:]
  brain_replays.append(dict(mode=mode,turn_windows=windows(records['turn'][1160:]),delta_from_actual_turn_windows=windows(records['turn'][1160:]-cut['raw_turn'][1160:]),brainnorm_windows=windows(np.linalg.norm(records['brain'][1160:],axis=1)),actual_sourcequeue_ticks=15400,source_hash_beforeafter=True,stored_body_observations_fixed=True,physical_counterfactual=False));proofs[mode]=dict(queue_final=final,consumer_count=len(events),intervention='Onlyregisteredbrainodorvalue0 and/orANmessagecut+receiver0 atANN1.6,nohreset',odor_hidden_source_records_unchanged=True)
 # Source-domain comparisons use realcalibrationinput/modal ranges andalready saved neural outputs.
 with np.load(CAL/'train.npz',allow_pickle=False) as z:tx=z['body111'].copy();tu=z['modalities'].copy();targets=z['current_desired'].copy()
 mean=m.observation_mean.numpy();std=m.observation_std.numpy();tz=(tx-mean)/std;low=tz.min((0,1));high=tz.max((0,1));domains=[];evalresult=read(CAL/'EVALUATION/RESULT.json');trAN=[];trh=[];trturn=[]
 for row in evalresult['rows']:
  if row['split']!='train':continue
  p=Path(row['artifact'])
  if sha(p)!=evalresult['output_hashes'][str(p)]:raise ValueError('StoredtrainingevaluationSHA')
  hashes[str(p)]=sha(p)
  with np.load(p,allow_pickle=False) as z:trAN.append(z['AN'].reshape(-1));trh.append(np.linalg.norm(z['brain_hidden'][:,0],axis=1));trturn.append(z['turn'].reshape(-1))
 for name,d in data.items():
  x=d['X238'][1160:,ci.MASK];z=(x-mean)/std;u=np.stack((d['visual_raw_turn'][1160:],d['odor_raw_turn'][1160:]),-1);strength=(u[:,0]*np.abs(u[:,0])+u[:,1]*np.abs(u[:,1]))/(np.abs(u[:,0])+np.abs(u[:,1])+1e-8);domains.append(dict(condition=name,normalized_body_absmax=float(abs(z).max()),normalized_body_outside_training_minmax_fraction=float(np.mean((z<low)|(z>high))),visual_stats=stats(u[:,0]),odor_stats=stats(u[:,1]),AN_stats=stats(d['AN_held'][1160:]),brainnorm_stats=stats(np.linalg.norm(d['brain_hidden_after'][1160:],axis=1)),turn_stats=stats(d['raw_turn'][1160:]),paired_engineering_teacher_windows_descriptive=windows(strength),actual_turn_minus_sameinstant_engineeringteacher_windows_descriptive=windows(d['raw_turn'][1160:]-strength),not_a_new_training_label_or_acceptance_threshold=True))
 check('all_frozen_parameters_unchanged',before=={'bridge':ci.tensor_hashes(m),'odor':ci.tensor_hashes(odor),'vision':ci.tensor_hashes(vision)});check('no_physical_backend_or_optimizer',not any(k in sys.modules for k in ('mujoco','physics','student_wave','flygym')))
 for p,h in hashes.items():
  if sha(p)!=h:raise ValueError('Sourcechangedafter '+p)
 np.savez_compressed(HERE/'OUTPUTS.npz',**outputs);report=dict(status='FIXED_OBSERVATION_OLF_MEMORY_AND_FUSION_FAILURE_DIAGNOSTIC_COMPLETE',reproduction_checks_pass=all(checks.values()),checks=checks,checkpoint_sha256=EXPECTED,case_statistics=stats_cases,odor_probes=odor_probes,brain_input_probes=brain_replays,sourcequeue_proofs=proofs,training_domain=dict(actual_body111_shape=list(tx.shape),modal_range=[dict(visual_stats=stats(tu[:,:,0]),odor_stats=stats(tu[:,:,1]))],current_teacher_stats=stats(targets),saved_final_brainnorm_stats=stats(np.concatenate(trh)),saved_final_AN_stats=stats(np.concatenate(trAN)),saved_final_turn_stats=stats(np.concatenate(trturn))),actual_domains=domains,output_path=str(HERE/'OUTPUTS.npz'),output_sha256=sha(HERE/'OUTPUTS.npz'),source_hashes=hashes,parameter_hashes_before_after=before,extra_training_updates=0,optimizer_steps=0,Core_constructions=0,new_physics_steps=0,physical_counterfactual_claim=False,rootcause_or_bistability_proven=False,seconds=time.monotonic()-started)
 fresh(HERE/'RESULT.json',report);print(json.dumps(dict(status=report['status'],checks=len(checks),all_passed=True,odor_probe_early_cut_RMS={r['mode']:r['rawturn_windows']['early_cut']['RMS'] for r in odor_probes},brain_probe_early_cut_RMS={r['mode']:r['turn_windows']['early_cut']['RMS'] for r in brain_replays},Fleft_missing_cycles=next(r['missing_complete_cycle_legs'] for r in stats_cases if r['condition']=='F_left_agree'),training_updates=0,physics_steps=0,seconds=report['seconds']),indent=2))
if __name__=='__main__':main()
