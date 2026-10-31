"""S131/P1/S2-S4 ONE bounded contact-aware QP; parent audits and executes.
Import inert. No neural runner, mj_step, trajectory, checkpoint or physical admission.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, csv, json
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from train_local_translation_candidate import integrate, BINS
from run_matched_state_counterfactual import ROOT, BASE, A, XML, sha, ah, js, dump, load, snapshot, error
from run_contact_aware_activation_response import committed_hashes, moment
from audit_muscle_reconstructability import xmltree, resolve, gain_bias

S130=BASE/'s130-contact-aware-activation-response/20261024T175612Z'
S125=BASE/'s125-local-translation-candidate/20261024T080500Z'
S119=BASE/'s119-training-constraints/20261024T070600Z'
SAMPLES=[77,100,105]
SCALE=.002/np.sqrt(15)
TOL=1e-9

def csvout(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main(out):
 assert out.is_absolute() and not out.exists(),'Fresh absolute output required'
 out.mkdir(parents=True)
 prereg=dict(created_utc=datetime.now(timezone.utc).isoformat(),task='S131 bounded OFFLINE ONLY contact-resolved instantaneous candidate',reference=str(A),source_chain=str(S130),formal_constraints=str(S119/'TRAINING_CONSTRAINTS.json'),samples=SAMPLES,allowed_pools=[3,15,38],entries=15,freeze='pool36, all other W entries, original zero mask; original S121 W not S126',weights_range=[0,4],variable='x=deltaW/(.002/sqrt(15))',scale=SCALE,objective='mean(three predicted delta contact-resolved qacc8)/1 + .05*sum(x*x)',objective_scale_native=1.,regularization=.05,constraint_row_scaling='positive reciprocal maxabs coefficient per nonzero raw row; zero rows factor1; solver sees scaled rows/bounds, all residual gates recomputed in ORIGINAL native raw units',hessian='.1*I strictly positive definite; unique constrained QP minimum',solver=dict(method='SLSQP',invocations=1,x0='zero',ftol=1e-12,maxiter=200,no_retry=True),predicted_constraints=dict(per_sample_delta_load_ratio_lower=0.,per_sample_abs_delta_qacc67_upper=1.,per_sample_all48_abs_delta_qacc_upper=2.,weight_l2_upper=.002,individual_weight_box=SCALE,all166_abs_delta_ctrl_upper=.005,raw_clip_branch='identical <=0, interior(0,1), >=1 on all166/all66 raw pools; raw linear constraints exact [0,1], postcheck strict branch'),tolerances=dict(load_lower_numeric=1e-9,safety_caps=1e-9,activation_reference_absolute=1e-12,baseline_absolute=1e-8,baseline_scaled_relative=1e-6,geometry_absolute=1e-10,geometry_scaled_relative=1e-10,both_baseline_tolerances_required=True),pre_forward_reject='no effective scaled free objective Jacobian (maxabs<=1e-15), solver unsuccessful/nonfinite, residual>1e-9, or predicted mean qacc8 descent<1e-8',actual_acceptance='each of 3 load ratios >= its historical S121 reference -1e-9; DOF6/7 absdelta<=1+1e-9; all48 absdelta<=2+1e-9; MEAN three delta qacc8 < -1e-8 (no each-point sign requirement); command/weight/mask/branch gates all pass',activation='Recompute actual 166 FIR commands, literal 25us RK4 full121x90 activation using S125 functions; no integration in 160 initialization ticks',forward='ONLY activation substituted at each selected frozen S121 qpos/qvel/ctrl/applied-force/time; nonlinear muscle force and full contact resolver; NOT state propagation',planned=dict(model_load=1,baseline_forward=3,candidate_forward=3,maximum_forward=6,step=0,neural_ticks=0),baseline='Every historical gate incl all48 qacc/passive/bias/actuator/constraint, raw contact/geometry/J/force, reconstructed M, efc_state vs validated S130; ALL3 baseline pass before ANY candidate forward; fresh MjData zero warmstart is not historical warmstart',physical_admission=False,S119='NOT TESTED / NOT SATISFIED',units='native; physical units NOT_ESTABLISHED',checkpoint_replacement=False,after_failure='retain all evidence; no relaxation/re-solve/scale/backtrack/new attempt')
 dump(out/'PREREG.json',prereg); prereg_hash=sha(out/'PREREG.json')
 ledger=dict(model_load=0,forward=0,baseline_forward=0,candidate_forward=0,step=0,neural_ticks=0,solver_invocations=0)
 dump(out/'CALL_LEDGER.json',ledger)
 inputs={Path(__file__).resolve(),ROOT/'train_local_translation_candidate.py',ROOT/'run_matched_state_counterfactual.py',ROOT/'run_contact_aware_activation_response.py',ROOT/'audit_muscle_reconstructability.py'}
 for root in [S130,S125,S119,A.parent]:
  inputs.update(p for p in root.rglob('*') if p.is_file())
  for p,h in committed_hashes(root/'INPUT_HASHES.json').items():
   assert sha(p)==h,'Input commitment mismatch '+p
   inputs.add(Path(p))
 xmlfiles=set();nodes=resolve(xmltree(XML,xmlfiles));inputs.update(xmlfiles)
 before={str(p):sha(p) for p in sorted(inputs)};dump(out/'INPUT_HASHES.json',before)
 assert js(S130/'PARENT_INDEPENDENT_CHECK_FINAL.json')['passed'] is True
 assert js(S130/'SUMMARY.json')['verified_activation_columns']==15
 z=load(A/'TRAJECTORY.npz');ini=load(A/'INITIALIZATION.npz');rows=load(A/'CONSTRAINT_ROWS.npz')
 chain=load(S130/'FIXED_STATE_CHAIN.npz');old=load(S125/'LOCAL_CHAIN.npz');historical=load(S130/'RAW_PROBES.npz')
 W=z['weights'];pool=z['actuator_pool'];indices=chain['weight_indices'];n=15
 assert W.shape==(66,5) and indices.shape==(15,2)
 assert np.array_equal(indices,np.array([(p,b) for p in [3,15,38] for b in range(5)]))
 assert np.array_equal(W,ini['weights']) and np.array_equal(W,old['reference_weights'])
 assert np.all(W[indices[:,0],indices[:,1]]>0)
 assert error(old['reference_act'],z['act'],1e-12,1e-12)['passed']
 source=js(A.parent/'PREREG.json')['source_hashes'];manifests=[p for p in source if p.endswith('/MANIFEST.json') and 's4-endpoints' in p];assert len(manifests)==1
 groups=[tuple(sorted(r['local_indices'])) for r in js(manifests[0])['rows']]
 pools=sorted(set(g for g in groups if g));lookup={g:i for i,g in enumerate(pools)}
 assert np.array_equal(pool,[lookup.get(g,-1) for g in groups])
 hist=np.zeros((80,66));features=[]
 for sp in np.concatenate((ini['spikes'],z['spikes'])):
  activity=np.array([sp[list(g)].mean()/.0005/100 for g in pools]);hist=np.concatenate((activity[None],hist))[:80]
  features.append(np.stack([hist[lo:hi].mean(0) for lo,hi in BINS],axis=1))
 features=np.array(features);assert error(features,np.concatenate((ini['features'],z['features'])),1e-12,1e-12)['passed']
 mapped=pool>=0
 def controls(w):
  raw=np.sum(features*w,axis=2);c=np.zeros((166,90));c[:,mapped]=np.clip(raw[:,pool[mapped]],0,1);return raw,c
 raw,c=controls(W)
 assert error(c[:160],ini['decoded'],1e-12,1e-12)['passed'] and error(c[160:],z['applied'],1e-12,1e-12)['passed']
 assert not ini['body_act'].any() and not z['ctrl'][0].any()
 for i in range(1,121):assert np.array_equal(z['ctrl'][i],c[160+(i-1)//20])
 prm=np.zeros((90,10));pars={k:np.zeros((90,nc)) for k,nc in [('lengthrange',2),('gainprm',10),('biasprm',10)]}
 for j,node in enumerate(nodes):
  v=list(map(float,node['resolved']['dynprm'].split()));prm[j,:len(v)]=v
  for k in pars:
   v=list(map(float,node['resolved'][k].split()));pars[k][j,:len(v)]=v
 a0,_=integrate(c[160:],prm,ini['body_act']);assert error(a0,z['act'],1e-12,1e-12)['passed']
 draw=np.zeros((166,66,n));dc=np.zeros((166,90,n))
 for j,(p,b) in enumerate(indices):draw[:,p,j]=features[:,p,b]
 rawact=np.zeros_like(c);rawact[:,mapped]=raw[:,pool[mapped]]
 dc[:,mapped]=draw[:,pool[mapped]]*((rawact[:,mapped]>0)&(rawact[:,mapped]<1))[:,:,None]
 _,da=integrate(c[160:],prm,ini['body_act'],dc[160:])
 assert error(da,old['activation_derivative'],1e-12,1e-12)['passed']
 Q=[];L=[]
 for i in SAMPLES:
  p=f'sample{i:03d}';assert bool(chain[p+'_valid'])
  assert error(da[i,[0,1,2,4,5]],chain[p+'_activation_derivative'],1e-12,1e-12)['passed']
  Q.append(chain[p+'_weight_J_qacc']);L.append(chain[p+'_weight_J_load_ratio'])
 Q=np.array(Q);L=np.array(L).reshape(3,n);assert Q.shape==(3,48,15) and np.all(np.isfinite(Q)) and np.all(np.isfinite(L))
 objective=Q[:,8,:].mean(0)*SCALE
 matrices=[];lower=[];upper=[]
 def constrain(mat,lo,hi):
  mat=np.asarray(mat).reshape(-1,n);matrices.append(mat);lower.extend(np.broadcast_to(lo,(len(mat),)));upper.extend(np.broadcast_to(hi,(len(mat),)))
 constrain(L*SCALE,0,np.inf);constrain(Q[:,6:8]*SCALE,-1,1);constrain(Q*SCALE,-2,2);constrain(dc*SCALE,-.005,.005)
 # Exact linear raw inequalities; actual classification must remain identical.
 constrain(draw*SCALE,np.where(raw<=0,-np.inf,np.where(raw>=1,1,0)).ravel()-raw.ravel(),np.where(raw<=0,0,np.where(raw>=1,np.inf,1)).ravel()-raw.ravel())
 mat=np.vstack(matrices);lo=np.array(lower);hi=np.array(upper)
 rowmax=np.max(np.abs(mat),axis=1);row_scale=np.ones(len(mat));row_scale[rowmax>0]=1/rowmax[rowmax>0]
 scaled_mat=mat*row_scale[:,None];scaled_lo=lo*row_scale;scaled_hi=hi*row_scale
 wb=W[indices[:,0],indices[:,1]];bounds=np.stack((np.maximum(-1,-wb/SCALE),np.minimum(1,(4-wb)/SCALE)),axis=1)
 assert np.all(lo<=0) and np.all(hi>=0) and np.all(bounds[:,0]<=0) and np.all(bounds[:,1]>=0),'Zero must be feasible'
 np.savez_compressed(out/'QP_PROBLEM.npz',objective=objective,A=mat,lower=lo,upper=hi,bounds=bounds,Q=Q,L=L,weight_indices=indices,scale=np.asarray(SCALE),row_scale=row_scale,scaled_A=scaled_mat,scaled_lower=scaled_lo,scaled_upper=scaled_hi)
 def finish(status,**extra):
  assert before=={p:sha(p) for p in before} and sha(out/'PREREG.json')==prereg_hash
  dump(out/'FROZEN_BEFORE_AFTER.json',dict(passed=True,inputs=len(before),prereg_sha256=prereg_hash))
  summary=dict(status=status,offline_candidate_saved=(out/'CANDIDATE_WEIGHTS.npz').exists(),physical_admission=False,S119='NOT TESTED / NOT SATISFIED',standing_or_walking_claim=False,units='NOT_ESTABLISHED',ledger=ledger,**extra)
  dump(out/'SUMMARY.json',summary)
  (out/'REPORT.zh-CN.md').write_text('# S131 接触感知有界离线候选\n\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n\n仅三个S121冻结机械状态的即时接触重解；不含状态演化或神经感觉闭环梯度。非物理预热、站立或行走验收，不替换S113/checkpoint。失败不放宽、不重解、不追加扫描。\n')
  dump(out/'OUTPUT_HASHES.json',{str(p):sha(p) for p in out.iterdir() if p.is_file() and p.name!='OUTPUT_HASHES.json'})
  print(json.dumps(summary));return summary
 if np.max(np.abs(objective))<=1e-15:return finish('REJECT_NO_EFFECTIVE_FREE_OBJECTIVE_JACOBIAN')
 ledger['solver_invocations']=1;dump(out/'CALL_LEDGER.json',ledger)
 result=minimize(lambda x:float(objective@x+.05*(x@x)),np.zeros(n),jac=lambda x:objective+.1*x,bounds=bounds,constraints=[LinearConstraint(scaled_mat,scaled_lo,scaled_hi)],method='SLSQP',options=dict(ftol=1e-12,maxiter=200,disp=False))
 x=result.x;np.savez_compressed(out/'QP_SOLUTION.npz',x=x,objective=objective,gradient=objective+.1*x,raw_A=mat,lower=lo,upper=hi,bounds=bounds,row_scale=row_scale,regularization=np.asarray(.05));finite=bool(np.all(np.isfinite(x)))
 if not finite:
  dump(out/'SOLVER.json',dict(success=False,status=int(result.status),message=str(result.message),finite=False));return finish('REJECT_NONFINITE_SOLVER')
 residual=float(max(0,np.max(lo-mat@x),np.max(mat@x-hi),np.max(bounds[:,0]-x),np.max(x-bounds[:,1])))
 predicted_descent=float(-objective@x)
 delta=np.zeros_like(W);delta[indices[:,0],indices[:,1]]=SCALE*x;wc=W+delta
 rawnew,cnew=controls(wc);anew,_=integrate(cnew[160:],prm,ini['body_act'])
 branch=lambda r:np.where(r<=0,-1,np.where(r>=1,1,0))
 gates=dict(weight_range=bool(np.all(wc>=0) and np.all(wc<=4)),weight_l2=bool(np.linalg.norm(delta)<=.002+TOL),weight_box=bool(np.max(np.abs(delta))<=SCALE+TOL),pool36=bool(np.array_equal(wc[36],W[36])),all_other_entries=bool(np.array_equal(wc[~np.isin(np.arange(66),[3,15,38])],W[~np.isin(np.arange(66),[3,15,38])])),zero_mask=bool(np.array_equal(wc==0,W==0)),ctrl166=bool(np.max(np.abs(cnew-c))<=.005+TOL),clip_branches=bool(np.array_equal(branch(rawnew),branch(raw))))
 csvout(out/'ALL_WEIGHT_DELTAS.csv',[dict(pool=p,bin=b,original=repr(float(W[p,b])),delta=repr(float(delta[p,b])),candidate=repr(float(wc[p,b])),allowed=bool(p in [3,15,38])) for p in range(66) for b in range(5)])
 csvout(out/'ALL166_COMMAND_SERIES.csv',[dict(tick=k,actuator=a,pool=int(pool[a]),reference=float(c[k,a]),candidate=float(cnew[k,a]),delta=float(cnew[k,a]-c[k,a])) for k in range(166) for a in range(90)])
 csvout(out/'ACTIVATION_SERIES.csv',[dict(sample=i,actuator=a,reference=float(a0[i,a]),candidate=float(anew[i,a]),delta=float(anew[i,a]-a0[i,a])) for i in range(121) for a in range(90)])
 np.savez_compressed(out/'OFFLINE_CANDIDATE.npz',reference_weights=W,proposed_weights=wc,delta=delta,reference_ctrl=c,candidate_ctrl=cnew,reference_raw=raw,candidate_raw=rawnew,reference_act=a0,candidate_act=anew,features=features,activation_derivative=da,predicted_delta_qacc=np.einsum('sdj,j->sd',Q,SCALE*x),predicted_delta_load=L@(SCALE*x))
 dump(out/'SOLVER.json',dict(success=bool(result.success),status=int(result.status),message=str(result.message),iterations=int(result.nit),solve_invocations=1,linear_max_violation=residual,predicted_mean_descent=predicted_descent,offline_gates=gates))
 if not (result.success and residual<=TOL and predicted_descent>=1e-8 and all(gates.values())):return finish('REJECT_BEFORE_FORWARD')
 import mujoco as mj
 def reserve(kind):
  ledger[kind]+=1
  if kind.endswith('_forward'):ledger['forward']+=1
  assert ledger['model_load']<=1 and ledger['forward']<=6 and ledger['baseline_forward']<=3 and ledger['candidate_forward']<=3
  dump(out/'CALL_LEDGER.json',ledger)
 for name in ['control','act_dyn','act_gain','act_bias','passive']:
  getter=getattr(mj,'get_mjcb_'+name,None);assert getter is None or getter() is None
 reserve('model_load');m=mj.MjModel.from_xml_path(str(XML));pre=snapshot(m);assert pre==js(A/'MODEL_PROVENANCE.json')['before']
 m.opt.integrator=mj.mjtIntegrator.mjINT_RK4;m.opt.timestep=.000025;post=snapshot(m);assert post==js(A/'MODEL_PROVENANCE.json')['after']
 assert (m.nq,m.nv,m.nu)==(49,48,90) and not any([m.nmocap,m.neq,m.nplugin,m.nuserdata]) and 'qacc_warmstart' not in z
 dump(out/'MODEL_AUDIT.json',dict(version=mj.__version__,before=pre,after=post,warmstart='fresh zero not saved historical'))
 fmap=js(S130/'FOOT_MAP.json');geom_leg={int(k):v for k,v in fmap['geom_leg'].items()};floor=fmap['floor'];weight=fmap['weight_native'];saved={}
 def probe(i,candidate=False):
  d=mj.MjData(m)
  for k in ['qpos','qvel','act','ctrl','qfrc_applied','xfrc_applied']:getattr(d,k)[:]=z[k][i]
  if candidate:d.act[:]=anew[i]
  d.time=float(z['time'][i]);assert not d.qacc_warmstart.any()
  frozen={k:getattr(d,k).copy() for k in ['qpos','qvel','act','ctrl','qfrc_applied','xfrc_applied']};time=d.time
  reserve('candidate_forward' if candidate else 'baseline_forward');mj.mj_forward(m,d)
  assert time==d.time and all(np.array_equal(getattr(d,k),v) for k,v in frozen.items())
  c={k:getattr(d,k).copy() for k in ['qacc','qfrc_constraint','qfrc_actuator','qfrc_passive','qfrc_bias','actuator_force','actuator_length','actuator_velocity','efc_force','efc_type','efc_id','efc_state','solver_niter']}
  c.update(**frozen,time=np.asarray(time),warmstart_before=np.zeros(48),warnings=np.asarray([w.number for w in d.warning]),warning_lastinfo=np.asarray([w.lastinfo for w in d.warning]),nefc=np.asarray(d.nefc),ncon=np.asarray(d.ncon),actuator_moment=moment(m,d))
  for k in ['geom1','geom2','dim','efc_address','dist','pos','frame','friction']:c['contact_'+k]=np.asarray([getattr(v,k) for v in d.contact])
  c['contact_frame']=c['contact_frame'].reshape(d.ncon,3,3);c['contact_force']=np.zeros((d.ncon,6));foot=np.zeros((6,3))
  for ci,co in enumerate(d.contact):
   mj.mj_contactForce(m,d,ci,c['contact_force'][ci]);g1,g2=int(co.geom1),int(co.geom2)
   if floor in [g1,g2]:
    fg=g2 if g1==floor else g1
    if fg in geom_leg:foot[geom_leg[fg]]+=(1 if g1==floor else -1)*(co.frame.reshape(3,3).T@c['contact_force'][ci,:3])
  c['foot_fz']=foot[:,2];c['load_ratio']=np.asarray(foot[:,2].sum()/weight)
  assert not mj.mj_isSparse(m);c['efc_J_dense']=d.efc_J.copy().reshape(d.nefc,48)
  M=np.empty((48,48));mj.mj_fullM(m,M,d.qM);c['mass']=M
  prefix=f'sample{i:03d}_'+('candidate' if candidate else 'baseline')
  saved.update({prefix+'_'+k:v for k,v in c.items()});np.savez_compressed(out/'RAW_PROBES.npz',**saved)
  return c
 bases=[];bg=[]
 for i in SAMPLES:
  cbase=probe(i);bases.append(cbase);p=f'sample{i:03d}';rp=p+'_';checks={}
  for k in ['qacc','qfrc_constraint','qfrc_actuator','qfrc_passive','qfrc_bias','actuator_force','actuator_length','actuator_velocity','actuator_moment']:checks[k]=error(cbase[k],z[k][i])
  for k in ['efc_type','efc_id','contact_geom1','contact_geom2','contact_dim','contact_efc_address']:checks[k]=dict(passed=bool(np.array_equal(cbase[k],rows[rp+k])))
  for k in ['contact_pos','contact_frame','efc_J_dense']:checks[k]=error(cbase[k],rows[rp+k],1e-10,1e-10)
  checks['efc_force']=error(cbase['efc_force'],rows[rp+'efc_force'])
  checks['contact_force']=error(cbase['contact_force'],z['contact_force_raw6'][i,:int(cbase['ncon'])])
  checks['foot_fz']=error(cbase['foot_fz'],z['foot_force'][i,:,2]);checks['load_ratio']=error(cbase['load_ratio'],z['load_ratio'][i])
  checks['warnings']=dict(passed=bool(np.array_equal(cbase['warnings'],z['warnings'][i]) and not cbase['warnings'].any()))
  checks['nefc']=dict(passed=int(cbase['nefc'])==int(rows['nefc'][i]));checks['ncon']=dict(passed=int(cbase['ncon'])==int(z['contact_ncon'][i]))
  for k in ['efc_state','contact_dist','contact_friction']:
   ref=historical[p+'_baseline_'+k];checks['S130_'+k]=dict(passed=bool(np.array_equal(cbase[k],ref))) if k=='efc_state' else error(cbase[k],ref,1e-10,1e-10)
  checks['mass']=error(cbase['mass'],historical[p+'_mechanics_mass'],1e-10,1e-10)
  checks['J_transpose_force']=error(cbase['efc_J_dense'].T@cbase['efc_force'],cbase['qfrc_constraint'],1e-10,1e-10)
  gains=np.zeros(90);bias=gains.copy()
  for j in range(90):gains[j],bias[j],*_=gain_bias(cbase['actuator_length'][j],cbase['actuator_velocity'][j],pars['lengthrange'][j],pars['gainprm'][j],pars['biasprm'][j])
  checks['analytic_force']=error(gains*cbase['act']+bias,cbase['actuator_force'],1e-10,1e-10)
  bg.append(dict(sample=i,passed=all(v['passed'] for v in checks.values()),checks=checks));dump(out/'BASELINE_GATES.json',bg)
 if not all(g['passed'] for g in bg):
  assert snapshot(m)==post;return finish('REJECT_HISTORICAL_BASELINE_NO_CANDIDATE_FORWARD')
 candidates=[probe(i,True) for i in SAMPLES];actual=[];dq=[];dl=[]
 for j,(i,b,cn) in enumerate(zip(SAMPLES,bases,candidates)):
  q=cn['qacc']-z['qacc'][i];l=float(cn['load_ratio']-z['load_ratio'][i]);dq.append(q);dl.append(l)
  actual.append(dict(sample=i,load_delta=l,max_abs_delta_qacc67=float(np.max(np.abs(q[6:8]))),max_abs_delta_qacc48=float(np.max(np.abs(q))),delta_qacc8=float(q[8]),load_pass=bool(l>=-TOL),qacc67_pass=bool(np.max(np.abs(q[6:8]))<=1+TOL),qacc48_pass=bool(np.max(np.abs(q))<=2+TOL),warnings_zero=bool(not cn['warnings'].any()),contact_identity_unchanged=bool(all(np.array_equal(cn[k],b[k]) for k in ['nefc','ncon','efc_type','efc_id','efc_state','contact_geom1','contact_geom2','contact_dim','contact_efc_address'])),J_transpose_force=error(cn['efc_J_dense'].T@cn['efc_force'],cn['qfrc_constraint'],1e-10,1e-10)))
 dq=np.array(dq);dl=np.array(dl);mean8=float(dq[:,8].mean())
 admitted=bool(all(gates.values()) and all(v['load_pass'] and v['qacc67_pass'] and v['qacc48_pass'] and v['warnings_zero'] and v['contact_identity_unchanged'] and v['J_transpose_force']['passed'] for v in actual) and mean8 < -1e-8)
 dump(out/'ACTUAL_GATES.json',dict(points=actual,mean_delta_qacc8=mean8,mean_descent_pass=mean8 < -1e-8,offline_candidate_saved=admitted))
 csvout(out/'ALL48_ACTUAL_VS_LINEAR.csv',[dict(sample=i,dof=d,reference=float(z['qacc'][i,d]),candidate=float(candidates[j]['qacc'][d]),actual_delta=float(dq[j,d]),linear_delta=float(Q[j,d]@(SCALE*x))) for j,i in enumerate(SAMPLES) for d in range(48)])
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(3,1,figsize=(11,11));t=np.array(SAMPLES)*.025
 for ax,ref,act,pred,label in [(axes[0],z['qacc'][SAMPLES,8],dq[:,8],np.einsum('sj,j->s',Q[:,8],SCALE*x),'qacc8 native'),(axes[1],z['load_ratio'][SAMPLES],dl,L@(SCALE*x),'load ratio')]:
  ax.plot(t,ref,'o-',label='historical reference');ax.plot(t,ref+act,'o-',label='actual nonlinear contact');ax.plot(t,ref+pred,'x--',label='linear prediction');ax.set_ylabel(label);ax.legend()
 axes[2].plot(t,np.max(np.abs(dq),axis=1),'o-',label='actual max |delta all48 qacc|');axes[2].plot(t,np.max(np.abs(np.einsum('sdj,j->sd',Q,SCALE*x)),axis=1),'x--',label='linear max |delta all48|');axes[2].axhline(2,color='red',label='cap2');axes[2].legend();axes[2].set_xlabel('frozen S121 sample time ms')
 fig.suptitle('S131 OFFLINE ONLY fixed-state contact response; NOT physical admission');fig.tight_layout();fig.savefig(out/'CONTACT_AWARE_CANDIDATE.png',dpi=150);plt.close(fig)
 assert snapshot(m)==post and before=={p:sha(p) for p in before} and sha(out/'PREREG.json')==prereg_hash
 if admitted:np.savez_compressed(out/'CANDIDATE_WEIGHTS.npz',weights=wc,reference_weights=W,delta=delta)
 return finish('OFFLINE_CANDIDATE_PASS' if admitted else 'REJECT_ACTUAL_NONLINEAR_GATES',mean_delta_qacc8=mean8)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path);args=p.parse_args()
 main(args.output or BASE/'s131-contact-aware-candidate'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
