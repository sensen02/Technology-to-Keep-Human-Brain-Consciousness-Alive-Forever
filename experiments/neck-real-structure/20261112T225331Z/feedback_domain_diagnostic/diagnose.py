"""Parent CPU1 four fixed-observation replays and train-domain statistics; no training/physics."""
from pathlib import Path
import argparse,json,hashlib,sys,importlib.util,os,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;CAL=ROOT/'neural_calibration';BODY=ROOT/'body_pilot';CASE=BODY/'trials/P_empty';SHA='eed1ad3cc61a1b59086ff2324175f6bf3b5da5aa197165e6c4b009c17027c95e'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def save(p,value):
 with Path(p).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def window_indices(window):return [k for k in range(1920) if window['start_s']-1e-10<=k*.005<window['end_s']-1e-10]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--execute-parent',action='store_true');a=ap.parse_args()
 if not a.execute_parent:print('CODE_ONLY no arrays/models imported');return
 for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
  if os.environ.get(k)!='1':raise ValueError(k+'=1 required')
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});start=time.monotonic();pr=read(HERE/'PREREG.json');checkpoint=CAL/'TRAIN/checkpoint_000200.pt'
 if sha(checkpoint)!=SHA or pr['checkpoint_sha256']!=SHA:raise ValueError('Fixed final200 source identity required')
 if any((HERE/n).exists() for n in ('INPUTS_BEFORE.json','RESULT.json','OUTPUTS.npz')):raise FileExistsError('Inspect saved diagnostic before rerun')
 ca=read(CAL/'EVALUATION/VERIFY.json');ba=read(CASE/'VERIFY.json')
 if not ca['integrity_all_pass'] or not all(ca['checks'].values()) or not ba['integrity_all_pass'] or not all(ba['checks'].values()) or ca['checkpoint_sha256']!=SHA or ba['brain_checkpoint_sha256']!=SHA:raise ValueError('Completed source/body independent audits required')
 if ca['result_sha256']!=sha(CAL/'EVALUATION/RESULT.json') or ba['result_sha256']!=sha(CASE/'RESULT.json'):raise ValueError('Actual evidence hash mismatch')
 paths=[HERE/'PREREG.json',HERE/'diagnose.py',checkpoint,CAL/'train.npz',CAL/'EVALUATION/V_empty.npz',CAL/'EVALUATION/RESULT.json',CAL/'EVALUATION/VERIFY.json',CASE/'neural_decisions.npz',CASE/'RESULT.json',CASE/'VERIFY.json',BODY/'SOURCE_ONLY_DIAGNOSTIC.zh-CN.md']
 hashes={str(p):sha(p) for p in paths};hashes.update(read(CAL/'FROZEN.json')['source_hashes']);hashes.update(read(CAL/'FROZEN.json')['data_hashes']);hashes.update(read(CASE/'RESULT.json')['source_hashes'])
 for path,h in hashes.items():
  if sha(path)!=h:raise ValueError('Diagnostic source drift '+path)
 save(HERE/'INPUTS_BEFORE.json',hashes)
 sentinel=object();previous=sys.modules.get('model',sentinel)
 try:
  architecture=module('_feedbackdiagnostic_calibration_model',CAL/'model.py');sys.modules['model']=architecture;io=module('_feedbackdiagnostic_calibration_io',CAL/'common.py');model,payload=io.load_checkpoint(checkpoint,SHA,final=True)
 finally:
  if previous is sentinel:sys.modules.pop('model',None)
  else:sys.modules['model']=previous
 import torch,numpy as np
 torch.set_num_threads(1);model.requires_grad_(False);model.eval()
 def tensorhash():return {k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in model.state_dict().items()}
 before=tensorhash()
 with np.load(CAL/'train.npz',allow_pickle=False) as z:train=z['body111'].copy();traincue=z['cue'].copy()
 with np.load(CASE/'neural_decisions.npz',allow_pickle=False) as z:actual={k:z[k].copy() for k in ('X238','AN_encoded','AN_held','DN_encoded','DN_held','brain_hidden_before','brain_hidden_after','raw_turn','cue','AN_capture_step','AN_release_step')}
 with np.load(CAL/'EVALUATION/V_empty.npz',allow_pickle=False) as z:offline={k:z[k].copy() for k in ('turn','DN','AN','brain_hidden')}
 allowed=model.allowed_raw_indices.tolist();archived=train[:,0:1,:];onpolicy=actual['X238'][1160:,allowed][:,None,:];warm=actual['AN_encoded'][1159][None,:].copy()
 if train.shape!=(1920,4,111) or onpolicy.shape!=(1920,1,111) or not np.all(traincue[:,0]==0) or not np.all(actual['cue']==0):raise ValueError('Exact empty-scope source shape and zero cue required')
 if not np.array_equal(warm[0],actual['AN_held'][1160]) or actual['AN_capture_step'][1160]!=5795 or actual['AN_release_step'][1160]!=5797:raise ValueError('Actual firstwarmAN not bound to recordedcapture/release')
 if np.max(abs(actual['brain_hidden_before'][1160]))!=0 or np.max(abs(actual['DN_held'][1160]))!=0:raise ValueError('Actual initialbrainh0/DNhold0 required')
 def replay(body,initial_an):
  h=torch.zeros((1,16),dtype=torch.float64);held_an=torch.from_numpy(initial_an.copy());held_dn=torch.zeros((1,2),dtype=torch.float64);cue=torch.zeros((1,1),dtype=torch.float64);outputs={k:[] for k in ('turn','DN','AN','brain_hidden','AN_held','DN_held')}
  with torch.no_grad():
   for k in range(1920):
    x=torch.from_numpy(body[k]);an=model.feedback(model.normalize(x));dn,h=model.brain_step(held_an,cue,h);turn=model.decoder(held_dn)
    for key,value in [('turn',turn),('DN',dn),('AN',an),('brain_hidden',h),('AN_held',held_an),('DN_held',held_dn)]:outputs[key].append(value.numpy().copy())
    held_an=an;held_dn=dn
  return {k:np.stack(v) for k,v in outputs.items()}
 results={};saved={};checks={}
 for label,body,initial in [('archived_cold',archived,np.zeros((1,1))),('archived_warm',archived,warm),('actual_cold',onpolicy,np.zeros((1,1))),('actual_warm',onpolicy,warm)]:
  out=replay(body,initial);saved[label]=out
  for key,value in out.items():results[label+'__'+key]=value
  print(json.dumps(dict(completed=label,wall_s=time.monotonic()-start)),flush=True)
 for key in offline:checks['archived_cold_reproduces_original_'+key]=np.array_equal(saved['archived_cold'][key],offline[key])
 online_fields={'turn':'raw_turn','DN':'DN_encoded','AN':'AN_encoded','brain_hidden':'brain_hidden_after','AN_held':'AN_held','DN_held':'DN_held'}
 for key,record in online_fields.items():
  expected=actual[record][1160:];value=saved['actual_warm'][key][:,0]
  if key=='turn':value=value[:,0]
  checks['actual_warm_reproduces_saved_ANN_'+key]=np.array_equal(value,expected)
 checks['parameters_unchanged']=tensorhash()==before
 def stats(value):return dict(RMS=float(np.sqrt(np.mean(value**2))),mean=float(np.mean(value)),minimum=float(np.min(value)),maximum=float(np.max(value)),peak_abs=float(np.max(abs(value))))
 rows=[]
 for label,out in saved.items():rows.append(dict(condition=label,initialAN='actualwarm' if label.endswith('warm') else 'zero',windows={w['label']:{key:stats(value[window_indices(w)]) for key,value in out.items()} for w in pr['windows']}))
 comparisons=[]
 for left,right,meaning in [('archived_warm','archived_cold','InitialANeffect_fixedarchivedobservations'),('actual_warm','actual_cold','InitialANeffect_fixedactualobservations'),('actual_cold','archived_cold','Observationsequenceeffect_samecoldinitialAN'),('actual_warm','archived_warm','Observationsequenceeffect_samewarminitialAN')]:comparisons.append(dict(left=left,right=right,meaning=meaning,windows={w['label']:{key:stats(saved[left][key][window_indices(w)]-saved[right][key][window_indices(w)]) for key in saved[left]} for w in pr['windows']}))
 mean=model.observation_mean.numpy();std=model.observation_std.numpy();tz=(train-mean)/std;trainmin=tz.min((0,1));trainmax=tz.max((0,1));domains={}
 for name,raw in [('actual_settle',actual['X238'][:200,allowed]),('actual_teacher',actual['X238'][200:1160,allowed]),('actual_ANN',actual['X238'][1160:,allowed]),('archived_empty',archived[:,0])]:
  z=(raw-mean)/std;outside=(z<trainmin-1e-12)|(z>trainmax+1e-12);rank=np.argsort(np.max(abs(z),0))[::-1][:15];domains[name]=dict(rows=len(z),max_abs_z=float(np.max(abs(z))),p95_abs_z=float(np.quantile(abs(z),.95)),outside_train_minmax_fraction=float(np.mean(outside)),row_any_feature_outside_fraction=float(np.mean(outside.any(1))),top_features=[dict(raw_index=allowed[int(j)],train_std=float(std[j]),max_abs_z=float(np.max(abs(z[:,j]))),train_z_min=float(trainmin[j]),train_z_max=float(trainmax[j]),outside_fraction=float(np.mean(outside[:,j]))) for j in rank])
 anstats={name:dict(max_abs=float(np.max(abs(value))),fraction_abs_gt_0_95=float(np.mean(abs(value)>.95)),mean=float(np.mean(value)),std=float(np.std(value))) for name,value in [('actual_warmup_encoded',actual['AN_encoded'][:1160]),('actual_ANN_encoded',actual['AN_encoded'][1160:]),('archived_empty_encoded',offline['AN'])]}
 for path,h in hashes.items():checks['source_unchanged:'+path]=sha(path)==h
 output=HERE/'OUTPUTS.npz';np.savez_compressed(output,**results)
 report=dict(status='FIXED_OBSERVATION_FACTORIAL_DIAGNOSTIC_COMPLETE' if all(checks.values()) else 'DIAGNOSTIC_REPRODUCTION_CHECK_FAILED_DO_NOT_INTERPRET',all_reproduction_and_identity_checks_pass=all(checks.values()),checks=checks,rows=rows,comparisons=comparisons,train_normalization=dict(raw_indices=allowed,minimum_std=float(std.min()),maximum_std=float(std.max()),floor1e_6_count=int(np.sum(std<=1e-6)),mean_and_std_unchanged=True),domain_statistics=domains,AN_saturation_descriptive_only=anstats,actual_initial_warm_AN=warm.tolist(),parameter_tensor_SHA_before=before,parameter_tensor_SHA_after=tensorhash(),output_path=str(output),output_sha256=sha(output),source_hashes=hashes,neural_replays=4,steps_each=1920,extra_training_updates=0,optimizer_steps=0,Core_constructions=0,new_physics_steps=0,physical_counterfactual_claim=False,bistability_proven=False,interpretation='InitialAN and observation-domain effects at fixedsavedobservations are descriptive. Replays are not new closed-loop body trajectories. No threshold change or runtime reset/gating.',seconds=time.monotonic()-start)
 save(HERE/'RESULT.json',report)
 print(json.dumps(dict(status=report['status'],reproduction_checks_pass=all(checks.values()),turn_RMS={k:stats(v['turn'])['RMS'] for k,v in saved.items()},actual_initial_warm_AN=warm.tolist(),std_floor_count=report['train_normalization']['floor1e_6_count'],AN_saturation=anstats,training_updates=0,new_physics_steps=0,seconds=report['seconds']),indent=2),flush=True)
 if not all(checks.values()):raise ValueError('Diagnostic exact reproduction or identity failed, outputs preserved')
if __name__=='__main__':main()
