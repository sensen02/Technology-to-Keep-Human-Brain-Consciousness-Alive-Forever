"""Array-only same-checkpoint step/ramp comparison. No physical backend imports."""
from pathlib import Path
import hashlib,json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
SHA='17fd103492007dc4de4ae9832fa39e4098cef980c4bebb03425c99f7c2f093a8'
COMMANDS={'left':[0.,1.],'right':[0.,-1.],'speed085':[-.15,0.],'speed090':[-.1,0.]}
W=24.446354755878005

def digest(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def readj(p):return json.loads(Path(p).read_text())
def arr(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def main():
 sources={str(HERE/'compare.py'):digest(HERE/'compare.py')};data={};metrics={};checks={}
 for key,path,auditpath in [('neutral',ROOT/'continuation3000/RECOVERY_EVAL_neutral',ROOT/'continuation3000/RECOVERY_VERIFY_ACTUAL_neutral.json')]+[(mode+'_'+case,ROOT/('continuation3000/RECOVERY_EVAL_'+case if mode=='step' else 'ramp3000/EVAL_'+case),ROOT/('continuation3000/RECOVERY_VERIFY_ACTUAL_'+case+'.json' if mode=='step' else 'ramp3000/VERIFY_'+case+'.json')) for case in COMMANDS for mode in ('step','ramp')]:
  rp=path/'RESULT.json';r=readj(rp);audit=readj(auditpath)
  assert audit['integrity_all_pass'] and all(audit['checks'].values())
  assert r['checkpoint_sha256']==SHA
  for filename,record in [('trace_1ms.npz','trace'),('neural_decisions.npz','neural_decisions')]:assert digest(path/filename)==r[record]['sha256']
  for p in (rp,auditpath,path/'trace_1ms.npz',path/'neural_decisions.npz',path/'IDENTITY.json'):sources[str(p)]=digest(p)
  t=arr(path/'trace_1ms.npz');d=arr(path/'neural_decisions.npz');data[key]=(t,d,r)
  assert t['time'].shape==(15400,) and d['command'].shape==(2880,2)
  assert all(np.isfinite(v).all() for v in t.values()) and all(np.isfinite(v).all() for v in d.values())
  assert np.max(np.abs(t['external_body_force_max']))==0 and np.max(np.abs(t['external_generalized_force_max']))==0
  sel=slice(5800,15400);duration=t['time'][-1]-t['time'][5800];delta=t['com'][-1]-t['com'][5800]
  forward=float(delta[0]/duration);den=float(np.linalg.norm(delta[:2])/duration)
  loads=t['loads'][sel];vt=(loads*t['speed'][sel]).sum(0)/np.maximum(loads.sum(0),1e-12);slip=vt/max(den,1e-9)
  a=r['autonomous'];assert forward==a['forward_signed_native_per_s'] and float(slip.max())==a['slip_ratio_max']
  assert float(t['com'][sel,2].min())==a['com_height_min'] and float(t['nonfoot'][sel].max())==a['nonfoot_force_max_native']
  assert np.allclose(slip,list(a['per_leg_slip_ratio'].values()),rtol=0,atol=0)
  cmd=np.zeros((2880,2))
  if key!='neutral':
   mode,case=key.split('_',1);elapsed=np.arange(1920)*.005;u=np.clip(elapsed,0,1);blend=3*u*u-2*u*u*u
   cmd[960:]=np.asarray(COMMANDS[case])*(blend[:,None] if mode=='ramp' else 1.)
  err=float(np.max(np.abs(cmd-d['command'])));assert err<1e-15
  metrics[key]=dict(path=str(path),checkpoint_sha256=SHA,engineering_pass=r['engineering_pass'],gates=a['gates'],all6=r['autonomous_cycles']['all_legs_completed'],cycles=r['autonomous_cycles']['complete_per_leg'],slip_max=a['slip_ratio_max'],slip_per_leg=a['per_leg_slip_ratio'],signed_forward=forward,net_xy_speed_denominator=den,loadweighted_absolute_foot_speed=vt.tolist(),com_min=a['com_height_min'],nonfoot_peak_native=a['nonfoot_force_max_native'],foot_support_mean=a['foot_support_mean'],yaw=r['yaw_change_rad'],command_formula_max_error=err,integrity_audit=str(auditpath),audit_checks=len(audit['checks']))
  checks[key+'_fullmetrics_command_audit']=True
 base=data['neutral']
 prefix={};onset={}
 for key,(t,d,r) in data.items():
  pp={k:bool(np.array_equal(t[k][:5800],base[0][k][:5800])) for k in ('time','qpos','qvel','action','target','actualctrl','loads','com')}
  pp.update({'neural_'+k:bool(np.array_equal(d[k][:960],base[1][k][:960])) for k in ('time','X238','command','hidden_before','hidden_after','mean','policy_action','actually_applied_action','control_source')})
  assert all(pp.values());prefix[key]=pp
 for case in COMMANDS:
  st,sd,_=data['step_'+case];rt,rd,_=data['ramp_'+case]
  dif=np.max(np.abs(st['qpos']-rt['qpos']),axis=1);ix=np.flatnonzero(dif>1e-12)
  onset[case]=dict(first_qpos_difference_gt1e12_s_after_handover=None if not len(ix) else float(st['time'][ix[0]]-st['time'][5799]),first_1s_qpos_difference_rms=float(np.sqrt(np.mean((st['qpos'][5800:6800]-rt['qpos'][5800:6800])**2))),first_1s_qvel_difference_rms=float(np.sqrt(np.mean((st['qvel'][5800:6800]-rt['qvel'][5800:6800])**2))),descriptive_only='No onset subwindow acceptance or causal conclusion; full9.6s results remain decisive')
 fig,axes=plt.subplots(2,3,figsize=(19,10),constrained_layout=True)
 colors=['tab:blue','tab:orange'];x=np.arange(4);names=list(COMMANDS)
 for j,mode in enumerate(('step','ramp')):
  mm=[metrics[mode+'_'+c] for c in names];offset=(j-.5)*.34
  axes[0,0].bar(x+offset,[m['slip_max'] for m in mm],.34,label=mode,color=colors[j]);axes[0,0].set_yscale('log')
  axes[0,1].bar(x+offset,[m['signed_forward'] for m in mm],.34,label=mode,color=colors[j])
  axes[0,2].bar(x+offset,[m['com_min'] for m in mm],.34,label=mode,color=colors[j])
  axes[1,0].bar(x+offset,[m['nonfoot_peak_native']/W for m in mm],.34,label=mode,color=colors[j])
 for ax in axes.flat:ax.grid(alpha=.25)
 for ax in axes.flat[:4]:ax.set_xticks(x,names);ax.legend()
 axes[0,0].axhline(.5,color='red',ls='--',label='original limit .5');axes[0,0].set_title('Full9.6s maximum per-leg slip (log)')
 axes[0,1].axhline(.05,color='red',ls='--');axes[0,1].set_title('Full9.6s signed forward speed; original >=.05')
 axes[0,2].axhline(1.2,color='red',ls='--');axes[0,2].set_title('Full9.6s minimum COM; original >=1.20')
 axes[1,0].axhline(.2,color='red',ls='--');axes[1,0].set_title('Full9.6s nonfoot peak /W; strict zero reported')
 cmap=plt.get_cmap('tab10')
 for ci,case in enumerate(names):
  for mode,style in [('step','--'),('ramp','-')]:
   t,_,_=data[mode+'_'+case];rel=t['com'][5799:]-t['com'][5799]
   axes[1,1].plot(rel[:,0],rel[:,1],style,color=cmap(ci),label=case+' '+mode,alpha=.8)
   tm=t['time'][5799:]-t['time'][5799];axes[1,2].plot(tm,rel[:,2],style,color=cmap(ci),label=case+' '+mode,alpha=.8)
 axes[1,1].set_title('Actual complete autonomous COM paths; same axes');axes[1,1].set_xlabel('delta world X');axes[1,1].set_ylabel('delta world Y');axes[1,1].legend(fontsize=8)
 axes[1,2].set_title('Actual COM-height change over whole window');axes[1,2].set_xlabel('Actual seconds after handover');axes[1,2].legend(fontsize=8)
 fig.suptitle('Same checkpoint3000: STEP vs 1s SMOOTH RAMP — all command cases FAIL\nActual saved physics only; matched5800-prefix exact; no food qualification / no integration',fontsize=16)
 image=HERE/'ACTUAL_STEP_RAMP_COMPARISON.png';fig.savefig(image,dpi=160);plt.close(fig)
 assert all(digest(p)==h for p,h in sources.items())
 report=dict(status='VERIFIED_SAVED_ARRAY_COMPARISON_ALL_COMMAND_CASES_FAIL',checkpoint_sha256=SHA,full_window_s=9.6,metrics=metrics,prefix_checks=prefix,onset_descriptive=onset,checks=checks,sources=sources,physical_integration_steps=0,optimizer_steps=0,backend_imports=False,limitations=['No command onset causal attribution without further controlled evidence.','No short-window gates or omitted failure intervals.','All flags report original full-window criteria and existing independent audits.'])
 (HERE/'METRICS.json').write_text(json.dumps(report,indent=2)+'\n')
 desktop=Path('/home/sensen/Desktop/engine修复验收/食物趋近实施-20261110/续轮1-阶跃与渐变命令物理对照-均未达标.png')
 shutil.copyfile(image,desktop);assert digest(image)==digest(desktop)
 (HERE/'DELIVERY.json').write_text(json.dumps(dict(source=str(image),desktop=str(desktop),sha256=digest(image),matched=True),indent=2)+'\n')
 print(json.dumps(dict(metrics={k:{n:m[n] for n in ('engineering_pass','slip_max','signed_forward','com_min','nonfoot_peak_native','all6')} for k,m in metrics.items()},prefix_all_exact=True,onset=onset,image=str(image),desktop=str(desktop)),indent=2))

if __name__=='__main__':main()
