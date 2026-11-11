"""Array-only matched physical comparison; no neural/physical backend import."""
from pathlib import Path
import hashlib,json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
BASE=HERE.parent.parent
R=BASE/'20261110T235830Z'
NEW=BASE/'20261111T010218Z'
OLD=BASE/'20261110T191449Z/residual_candidate'
GROUPS=[('Parent500',R/'parent_control','RECOVERED_SUMMARY.json'),('Corrective1500',R/'evaluation','FINAL_SUMMARY.json'),('Original2-only1500',R/'two_only_evaluation','FINAL_SUMMARY.json'),('Positive-head2000',NEW/'evaluation','FINAL_SUMMARY.json')]
LEGS=['LF','LH','LM','RF','RH','RM'];WEIGHT=24.446354755878005
hashes={};arrays={};records=[];digest_cache={}

def digest(p):
 p=Path(p)
 if str(p) not in digest_cache:
  with p.open('rb') as f:digest_cache[str(p)]=hashlib.file_digest(f,'sha256').hexdigest()
 return digest_cache[str(p)]
def bind(p):
 p=Path(p);hashes[str(p)]=digest(p);return p
def text(p):return json.loads(bind(p).read_text())
def archive(p):
 p=bind(p)
 if str(p) not in arrays:
  with np.load(p,allow_pickle=False) as z:arrays[str(p)]={k:z[k].copy() for k in z.files}
 return arrays[str(p)]
def close(a,b):return bool(np.allclose(a,b,rtol=1e-12,atol=1e-12))
def yaw_q(q):
 w,x,y,z=q.T
 return np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))

def case(group,condition,run,auditpath,summary=None):
 result=text(run/'RESULT.json');audit=text(auditpath)
 if condition=='neutral' and group=='Parent500':
  assert audit['integrity_all_pass'] and audit['reports']['neutral']['integrity_pass']
  ar=audit['reports']['neutral'];check_count=len(ar['checks']);assert all(ar['checks'].values())
 else:
  assert audit['integrity_all_pass'] and all(audit['checks'].values())
  assert audit['checkpoint_sha256']==result['checkpoint_sha256'];check_count=len(audit['checks'])
 ident=text(run/'IDENTITY.json')
 canonical={k:v for k,v in ident.items() if k!='identity_sha256'}
 assert hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(',',':')).encode()).hexdigest()==ident['identity_sha256']==result['identity_sha256']
 for p,h in ident['files'].items():assert digest(p)==h['sha256'];hashes[p]=h['sha256']
 for p,h in result.get('source_hashes',{}).items():assert digest(p)==h;hashes[p]=h
 if summary:
  sc=summary['conditions'][condition]
  for key,filename in [('result_sha256',run/'RESULT.json'),('audit_sha256',auditpath)]:
   if key in sc:assert digest(filename)==sc[key]
 for key,name in [('trace','trace_1ms.npz'),('neural_decisions','neural_decisions.npz'),('geometry','geometry_1ms.npz'),('runtime_geometry','runtime_geometry.npz')]:
  if key in result:assert digest(run/name)==result[key]['sha256']
 t=archive(run/'trace_1ms.npz');d=archive(run/'neural_decisions.npz')
 assert len(t['time'])==15400 and len(d['X238'])==2880
 assert all(np.max(np.abs(t[k]))==0 for k in ('external_body_force_max','external_generalized_force_max'))
 mask=np.arange(5800,15400);times=t['time'][mask];com=t['com'][mask];loads=t['loads'][mask];speed=t['speed'][mask]
 duration=float(times[-1]-times[0]);delta=com[-1,:2]-com[0,:2];den=float(np.linalg.norm(delta)/duration);forward=float(delta[0]/duration)
 vt=(loads*speed).sum(0)/np.maximum(loads.sum(0),1e-12);slip=vt/max(den,1e-9)
 a=result['autonomous'];qyaw=yaw_q(t['qpos'][:,3:7]);yaw=float(qyaw[-1]-qyaw[5799])
 assert close(slip,[a['per_leg_slip_ratio'][leg] for leg in LEGS]);assert close(vt/den,slip)
 assert close(forward,a['forward_signed_native_per_s']) and close(den,a['forward_speed_native_per_s'])
 assert close(yaw,result['yaw_change_rad']) and close(loads.sum(1).mean()/WEIGHT,a['foot_support_mean'])
 assert close(com[:,2].min(),a['com_height_min']) and close(t['nonfoot'][mask].max(),a['nonfoot_force_max_native'])
 passed=bool(result.get('engineering_pass',result.get('candidate_engineering_pass')))
 out=dict(model=group,condition=condition,run=str(run),checkpoint_sha256=result['checkpoint_sha256'],engineering_pass=passed,gates=a['gates'],cycles=result['autonomous_cycles']['complete_per_leg'],all_six=result['autonomous_cycles']['all_legs_completed'],forward=forward,xy_denominator=den,max_slip=float(slip.max()),per_leg_slip=dict(zip(LEGS,map(float,slip))),per_leg_absolute_loaded_tangent_speed=dict(zip(LEGS,map(float,vt))),yaw_rad=yaw,COMmin=float(com[:,2].min()),nonfoot_peak=float(t['nonfoot'][mask].max()),footload_mean_over_weight=float(loads.sum(1).mean()/WEIGHT),audit_checks=check_count,identity_sha256=ident['identity_sha256'],result_sha256=digest(run/'RESULT.json'),audit_sha256=digest(auditpath))
 records.append(out);return out

summaries={}
for group,folder,summaryname in GROUPS:
 summary=text(folder/summaryname);summaries[group]=summary
 assert summary['status'] in ('COMPLETE','COMPLETE_RECOVERED_FROM_EXISTING_SAVED_RUNS','COMPLETE_RECOVERED') or summary.get('saved_trajectory_verification') is True
 if group=='Parent500':case(group,'neutral',OLD/'EVAL_RESIDUAL500_neutral',OLD/'ACTUAL_VERIFICATION.json')
 else:case(group,'neutral',folder/'EVAL_neutral',folder/'VERIFY_neutral.json',summary)
 for cond in ('left','right'):
  ap=folder/(('RECOVERED_VERIFY_' if group=='Parent500' else 'VERIFY_')+cond+'.json')
  case(group,cond,folder/('EVAL_'+cond),ap,summary)

# Shared actual prefix is exact; neutral body trajectories all match original authority.
original=archive(Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152754Z/NATIVE_ROLE_2500/trace_1ms.npz'))
reference=arrays[str(OLD/'EVAL_RESIDUAL500_neutral/trace_1ms.npz')]
for rec in records:
 t=arrays[str(Path(rec['run'])/'trace_1ms.npz')]
 rec['matched_5800_prefix_exact']=all(np.array_equal(t[k][:5800],reference[k][:5800]) for k in ('time','qpos','qvel','action','target','actualctrl','com','loads'))
 assert rec['matched_5800_prefix_exact']
 if rec['condition']=='neutral':
  rec['neutral_body_qpos_qvel_original_bitexact']=all(np.array_equal(t[k],original[k]) for k in ('qpos','qvel'))
  assert rec['neutral_body_qpos_qvel_original_bitexact']

# New right branch is literally unchanged from matched original2-only1500 run.
newrun=NEW/'evaluation/EVAL_right';parentrun=R/'two_only_evaluation/EVAL_right'
right_checks={}
for name in ('trace_1ms.npz','neural_decisions.npz','geometry_1ms.npz'):
 a=archive(newrun/name);b=archive(parentrun/name)
 for key in a:
  if key in b:right_checks[name+':'+key]=bool(np.array_equal(a[key],b[key]))
assert all(right_checks.values())

fig,axes=plt.subplots(2,2,figsize=(16,10));colors=['#777777','#c55a11','#1768a2','#209560'];x=np.arange(2);width=.19
for j,(group,folder,_) in enumerate(GROUPS):
 turn=[next(r for r in records if r['model']==group and r['condition']==c) for c in ('left','right')]
 axes[0,0].bar(x+(j-1.5)*width,[r['max_slip'] for r in turn],width,color=colors[j],label=group)
 for k,r in enumerate(turn):axes[0,0].text(k+(j-1.5)*width,r['max_slip']+.001,f'{r["max_slip"]:.5f}',ha='center',rotation=90,fontsize=8)
 axes[0,1].bar(x+(j-1.5)*width,[r['forward'] for r in turn],width,color=colors[j])
 axes[1,0].bar(x+(j-1.5)*width,np.degrees([r['yaw_rad'] for r in turn]),width,color=colors[j])
 for c,style in [('left','-'),('right','--')]:
  r=next(r for r in turn if r['condition']==c);t=arrays[str(Path(r['run'])/'trace_1ms.npz')];path=t['com'][5799:,:2]-t['com'][5799,:2]
  axes[1,1].plot(path[:,0],path[:,1],style,color=colors[j],label=group+' '+c)
axes[0,0].axhline(.5,color='red',ls='--',label='Original slip limit 0.5')
neutral=next(r for r in records if r['condition']=='neutral')['max_slip'];axes[0,0].axhline(neutral,color='black',ls=':',label=f'All neutral {neutral:.6f}')
axes[0,0].set_ylim(0,.64);axes[0,0].set_title('Full 9.6s maximum per-leg slip — zero baseline retained');axes[0,0].legend(fontsize=8)
axes[0,1].axhline(.05,color='red',ls='--');axes[0,1].set_title('Signed forward speed — original minimum 0.05');axes[0,1].set_ylabel('Native model units / s')
axes[1,0].axhline(np.degrees(.01),color='red',ls=':');axes[1,0].axhline(-np.degrees(.01),color='red',ls=':');axes[1,0].set_title('Actual thorax yaw change — sign and magnitude');axes[1,0].set_ylabel('Degrees')
for ax in (axes[0,0],axes[0,1],axes[1,0]):ax.set_xticks(x,['Left','Right']);ax.grid(axis='y',alpha=.25)
axes[1,1].set_title('Actual complete autonomous COM paths — matched world axes');axes[1,1].set_xlabel('COM delta X (native)');axes[1,1].set_ylabel('COM delta Y (native)');axes[1,1].legend(fontsize=7);axes[1,1].grid(alpha=.25)
fig.suptitle('MATCHED 1s COMMAND RAMPS — four fixed neural models; new left still FAIL\nNew positive-head2000 right is bitexact to original2-only1500 right. Fixed-condition engineering only; NO FOOD SUCCESS.',fontsize=14)
fig.tight_layout(rect=(0,0,1,.93));image=HERE/'ACTUAL_MATCHED_COMPARISON.png';fig.savefig(image,dpi=160);plt.close(fig)

# Validate every bound input again without digest cache; do not silently accept ongoing changes.
for p,h in hashes.items():
 with Path(p).open('rb') as f:now=hashlib.file_digest(f,'sha256').hexdigest()
 assert now==h,'input changed '+p
report=dict(status='ARRAY_ONLY_VERIFIED_MATCHED_PHYSICAL_COMPARISON',cases=records,right_new_vs_parent_all_shared_arrays_bitexact=right_checks,right_new_independent_audit_checks=next(r for r in records if r['model']=='Positive-head2000' and r['condition']=='right')['audit_checks'],input_hashes=hashes,npz_archives_read_once=len(arrays),physics_integration_steps=0,optimizer_updates=0,backend_models_constructed=0,source_sha256=digest(__file__),limitations=['Fixed conditions only; no arbitrary command switching/initial-state robustness or full native certification.','New left head changes parameterization, range and fitted weights; actual improvement does not isolate a causal factor.','Original failed labels/gates remain unchanged; no food behavior claim.'])
(HERE/'METRICS.json').write_text(json.dumps(report,indent=2)+'\n')
desktop=Path('/home/sensen/Desktop/engine修复验收/食物趋近实施-20261110/续轮2-匹配转向与神经读出物理对照-左转未达标.png')
shutil.copyfile(image,desktop);assert digest(image)==digest(desktop)
(HERE/'DELIVERY.json').write_text(json.dumps(dict(project_image=str(image),desktop_image=str(desktop),sha256=digest(image),desktop_sha256=digest(desktop),visual_review='pending actual image opening'),indent=2)+'\n')
print(json.dumps(dict(cases=len(records),plot=str(image),desktop=str(desktop),right_equal_checks=len(right_checks),new_right_audit_checks=report['right_new_independent_audit_checks'],values=[dict(model=r['model'],condition=r['condition'],slip=r['max_slip'],forward=r['forward'],pass_=r['engineering_pass']) for r in records]),indent=2))
