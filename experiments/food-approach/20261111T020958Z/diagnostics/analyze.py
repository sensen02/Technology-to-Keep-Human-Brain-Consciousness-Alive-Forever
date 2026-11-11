"""Saved-array/strict-neural analysis only. Never constructs a physical model."""
from pathlib import Path
import hashlib,json,sys,importlib.util
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
W=24.446354755878005
LEGS=['LF','LH','LM','RF','RH','RM']
INPUTS={}

def digest(p):
 p=Path(p)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return h

def bind(p):
 p=Path(p).resolve();h=digest(p);INPUTS[str(p)]=h;return h

def readj(p):bind(p);return json.loads(Path(p).read_text())
def arrays(p):
 bind(p)
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def verify_map(values):
 for p,h in values.items():
  expected=h['sha256'] if isinstance(h,dict) else h
  if bind(p)!=expected:raise ValueError('source drift '+p)
def native(v):
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,dict):return {k:native(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [native(x) for x in v]
 return v

def case(name,run,auditpath,teacher=False):
 r=readj(run/'RESULT.json');audit=readj(auditpath)
 if not audit.get('integrity_all_pass',audit.get('all_passed',False)) or not all(audit['checks'].values()):raise ValueError('failed saved audit')
 if teacher:
  ident=readj(run.parent/'IDENTITY.json');verify_map(ident['source_hashes'])
  tpath=run/'TRACE_1MS.npz';npath=run/'SAMPLES_5MS.npz'
  if bind(tpath)!=r['trace']['sha256'] or bind(npath)!=r['samples']['sha256']:raise ValueError('teacher artifact drift')
  summary=r
 else:
  ident=readj(run/'IDENTITY.json');verify_map(ident['files']);verify_map(r['source_hashes'])
  canonical={k:v for k,v in ident.items() if k!='identity_sha256'}
  if hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=ident['identity_sha256']:raise ValueError('manifest canonical digest')
  if r['identity_sha256']!=ident['identity_sha256']:raise ValueError('identity binding')
  if Path(audit['run']).resolve()!=run.resolve() or audit['checkpoint_sha256']!=r['checkpoint_sha256'] or audit['engineering_pass']!=r['engineering_pass']:raise ValueError('saved audit run/model/outcome binding')
  verify_map(audit['source_hashes'])
  tpath=run/'trace_1ms.npz';npath=run/'neural_decisions.npz'
  if bind(tpath)!=r['trace']['sha256'] or bind(npath)!=r['neural_decisions']['sha256']:raise ValueError('model artifact drift')
  summary=r['autonomous']
 t=arrays(tpath);n=arrays(npath)
 if not teacher:
  geometry=arrays(run/'geometry_1ms.npz')
  if bind(run/'geometry_1ms.npz')!=r['geometry']['sha256']:raise ValueError('geometry hash')
 else:geometry=None
 if t['time'].shape!=(15400,) or len(n['X238'])!=2880:raise ValueError('coverage')
 if any(np.max(np.abs(t[k]))!=0 for k in ('external_body_force_max','external_generalized_force_max')):raise ValueError('extra force')
 sl=slice(5800,15400);loads=t['loads'][sl];speed=t['speed'][sl];com=t['com'][sl];time=t['time'][sl]
 duration=float(time[-1]-time[0]);delta=com[-1,:2]-com[0,:2];den=float(np.linalg.norm(delta)/duration)
 vt=np.sum(loads*speed,axis=0)/np.sum(loads,axis=0);ratios=vt/den
 checks=dict(slip_exact=all(float(ratios[i])==summary['per_leg_slip_ratio'][leg] for i,leg in enumerate(LEGS)),
             denominator_exact=den==summary['forward_speed_native_per_s'],signedforward_exact=float(delta[0]/duration)==summary['forward_signed_native_per_s'])
 if not all(checks.values()):raise ValueError('whole metric mismatch '+str(checks))
 result=dict(summary_checks=checks,engineering_pass=bool(r.get('engineering_pass',r.get('all_gates'))),original_gates=summary['gates'],
             duration_s=duration,net_XY_speed_denominator=den,signed_forward=float(delta[0]/duration),per_leg={})
 for j,leg in enumerate(LEGS):
  l=loads[:,j];s=speed[:,j];num=l*s;total=float(num.sum());loadsum=float(l.sum())
  categories={}
  for cat,mask in [('loaded_ge_point1W',l>=.1*W),('unloaded_le_point05W',l<=.05*W),('intermediate', (l>.05*W)&(l<.1*W))]:
   categories[cat]=dict(endpoint_count=int(mask.sum()),numerator_share=float(num[mask].sum()/total),
                        contribution_to_full_slip=float(num[mask].sum()/loadsum/den),mean_speed_when_selected=float(s[mask].mean()) if mask.any() else None)
  windows=[]
  for a,b in [(0,3200),(3200,6400),(6400,9600)]:
   windows.append(dict(descriptive_elapsed_s=[a*.001,b*.001],numerator_share=float(num[a:b].sum()/total),
                       contribution_to_full_slip=float(num[a:b].sum()/loadsum/den),
                       loaded_numerator_share_of_window=float(num[a:b][l[a:b]>=.1*W].sum()/num[a:b].sum()),
                       load_weighted_speed=float(num[a:b].sum()/l[a:b].sum())))
  result['per_leg'][leg]=dict(slip=float(ratios[j]),absolute_load_weighted_pad_tangent_speed=float(vt[j]),
                             load_sum_native=loadsum,numerator_sum_native_speed=total,
                             numerator_integral_native_speed_s=total*.001,categories=categories,descriptive_windows=windows)
 return dict(name=name,r=r,t=t,n=n,geometry=geometry,metrics=result)

def load_io(stamp):
 p=ROOT/stamp/'train/model_io.py';bind(p)
 spec=importlib.util.spec_from_file_location('_diag_model_'+stamp,p);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 return m

def neural_analysis(c,stamp):
 io=load_io(stamp);r=c['r'];bind(r['checkpoint']);model,payload=io.load_checked(r['checkpoint'],r['checkpoint_sha256'])
 verify_map(payload['config']['source_hashes']);verify_map(payload['config']['data_hashes'])
 n=c['n'];h=torch.tensor(n['hidden_after'][:,128:],dtype=torch.float64);u=torch.tensor(n['command'],dtype=torch.float64)
 with torch.no_grad():
  pre=model.readout(h);activation=pre.tanh();correction=model.correction(h,u)
 state=model.initial_state(1);mm=[];hh=[]
 with torch.no_grad():
  for x,cmd in zip(n['X238'],n['command']):
   mean,state=model.step(torch.tensor(x[None],dtype=torch.float64),torch.tensor(cmd[None],dtype=torch.float64),state)
   mm.append(mean[0].numpy());hh.append(torch.cat(state,-1)[0].numpy())
 exact=dict(all2880_saved_mean_bitexact=np.array_equal(np.array(mm),n['mean']),all2880_saved192_hidden_bitexact=np.array_equal(np.array(hh),n['hidden_after']))
 if not all(exact.values()):raise ValueError('neural replay mismatch')
 out=dict(strict_saved_replay=exact,readout_by_leg={})
 for j,leg in enumerate(LEGS):
  z=pre[960:,j*7:(j+1)*7].numpy();a=activation[960:,j*7:(j+1)*7].numpy();co=correction[960:,j*7:(j+1)*7].numpy()
  out['readout_by_leg'][leg]=dict(preactivation_abs_p95=float(np.quantile(np.abs(z),.95)),activation_abs_p95=float(np.quantile(np.abs(a),.95)),
       activation_abs_max=float(np.abs(a).max()),tanh_abs_ge_point95_fraction=float(np.mean(np.abs(a)>=.95)),
       tanh_abs_ge_point99_fraction=float(np.mean(np.abs(a)>=.99)),correction_abs_max=float(np.abs(co).max()),
       correction_abs_p95=float(np.quantile(np.abs(co),.95)),capacity_note='ratio abs(tanh) is fraction of current command-scaled .25 bound; no external clamp')
 return out,model

def main():
 torch.set_num_threads(1)
 cases=[case('teacher',ROOT/'20261110T191449Z/teacher/left_full',ROOT/'20261110T191449Z/teacher/VERIFICATION.json',True),
        case('head2000',ROOT/'20261111T010218Z/evaluation/EVAL_left',ROOT/'20261111T010218Z/evaluation/VERIFY_left.json'),
        case('head10000',ROOT/'20261111T013852Z/evaluation_recovery/EVAL_left',ROOT/'20261111T013852Z/evaluation_recovery/VERIFY_left.json')]
 teacher=cases[0];comparisons={};checks={}
 for c in cases[1:]:
  prefix={k:np.array_equal(c['t'][k][:5800],teacher['t'][k][:5800]) for k in ('time','qpos','qvel','com','action','target','actualctrl','loads')}
  prefix['prime_X238']=np.array_equal(c['n']['X238'][:960],teacher['n']['X238'][:960])
  if not all(prefix.values()):raise ValueError('prefix mismatch')
  byleg={}
  for j,leg in enumerate(LEGS):
   actual=c['metrics']['per_leg'][leg];ref=teacher['metrics']['per_leg'][leg]
   ad=c['t']['action'][5800:,j*7:(j+1)*7]-teacher['t']['action'][5800:,j*7:(j+1)*7]
   td=c['t']['target'][5800:,j*7:(j+1)*7]-teacher['t']['target'][5800:,j*7:(j+1)*7]
   fd=c['t']['relativefoot'][5800:,j]-teacher['t']['relativefoot'][5800:,j]
   byleg[leg]=dict(vt_ratio_to_teacher=actual['absolute_load_weighted_pad_tangent_speed']/ref['absolute_load_weighted_pad_tangent_speed'],
     slip_ratio_to_teacher=actual['slip']/ref['slip'],action_time_index_RMS=float(np.sqrt(np.mean(ad*ad))),
     target_time_index_RMS_rad=float(np.sqrt(np.mean(td*td))),relativefoot_time_index_RMS_xyz=np.sqrt(np.mean(fd*fd,axis=0)).tolist(),
     interpret='same saved elapsed index, not causal/aligned phase after physical divergence; coordinate difference is not pad velocity vector')
  comparisons[c['name']]=dict(prefix_exact=prefix,netXY_denominator_ratio=c['metrics']['net_XY_speed_denominator']/teacher['metrics']['net_XY_speed_denominator'],per_leg=byleg)
 neural={}
 for c,stamp in zip(cases[1:],['20261111T010218Z','20261111T013852Z']):
  neural[c['name']],model=neural_analysis(c,stamp)
 # Same causal teacher states, actual head mappings, to separate teacher fit from actual trajectory response.
 x=teacher['n']['X238'];u=teacher['n']['command2'];y=teacher['n']['Y42'];teacher_fit={}
 for stamp,name in [('20261111T010218Z','head2000'),('20261111T013852Z','head10000')]:
  io=load_io(stamp);c=next(c for c in cases if c['name']==name);model,_=io.load_checked(c['r']['checkpoint'],c['r']['checkpoint_sha256'])
  state=model.initial_state(1);pred=[];states=[]
  with torch.no_grad():
   for xx,uu in zip(x,u):
    mean,state=model.step(torch.tensor(xx[None],dtype=torch.float64),torch.tensor(uu[None],dtype=torch.float64),state)
    pred.append(mean[0].tanh().numpy());states.append(state[1][0].numpy())
   a=model.readout(torch.tensor(np.asarray(states),dtype=torch.float64)).tanh().numpy()
  err=np.asarray(pred)[960:]-y[960:];teacher_fit[name]={}
  for j,leg in enumerate(LEGS):
   e=err[:,j*7:(j+1)*7];loaded=teacher['t']['loads'][5799:15399:5,j]>=.1*W
   teacher_fit[name][leg]=dict(action_MSE=float(np.mean(e*e)),action_abs_p95=float(np.quantile(np.abs(e),.95)),action_abs_max=float(np.abs(e).max()),
     error_energy_loaded_fraction=float(np.sum((e*e)[loaded])/np.sum(e*e)),teacher_state_readout_abs_p95=float(np.quantile(np.abs(a[960:,j*7:(j+1)*7]),.95)))
 checks['all_whole_metrics_exact']=all(all(c['metrics']['summary_checks'].values()) for c in cases)
 checks['all_shared_prefixes_exact']=all(all(c['prefix_exact'].values()) for c in comparisons.values())
 checks['all_actual_checkpoint_outputs_states_exact']=all(all(v['strict_saved_replay'].values()) for v in neural.values())
 checks['no_physics_backend_loaded']=all(n not in sys.modules for n in ('mujoco','physics','student_wave'))
 bind(HERE/'ANALYSIS_PLAN.json');bind(Path(__file__))
 for p,h in INPUTS.items():
  if digest(p)!=h:raise ValueError('changed input during analysis '+p)
 checks['all_bound_inputs_unchanged']=True
 report=dict(checks=checks,all_passed=all(checks.values()),whole_window={c['name']:c['metrics'] for c in cases},comparisons=comparisons,
   readout_actual=neural,teacher_distribution_fit=teacher_fit,input_hashes=INPUTS,physics_integration_steps=0,optimizer_updates=0,
   scientific_limits=['No vector pad velocity is stored in scalar speed; cross-coordinate diagnosis is not established.',
   'Same time-index action/target/foot differences are descriptive; physical feedback differs after divergence.',
   'No optimizer, candidate model artifact, new physics, changed gate/window or body-infeasibility claim.'])
 with (HERE/'DIAGNOSIS.json').open('w') as f:json.dump(native(report),f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(dict(all_passed=report['all_passed'],checks=checks,slips={c['name']:c['metrics']['per_leg']['RH']['slip'] for c in cases},readout_RH={n:v['readout_by_leg']['RH'] for n,v in neural.items()}),indent=2))
 if not report['all_passed']:raise ValueError('diagnostic checks failed')

if __name__=='__main__':main()
