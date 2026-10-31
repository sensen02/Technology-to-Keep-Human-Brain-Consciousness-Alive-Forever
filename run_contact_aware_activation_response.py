"""S130/P1/S2-S4: bounded instantaneous activation/contact response, never a trajectory.
Import inert. Parent source audit required before execution. No weight generation.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, json
import numpy as np
from run_matched_state_counterfactual import ROOT, BASE, A, XML, sha, ah, js, dump, load, snapshot, error
from audit_muscle_reconstructability import gain_bias, xmltree, resolve

SAMPLES=[77,100,105]
ACTS=[0,1,2,4,5]
H=1e-5
CHAIN=BASE/'s125-local-translation-candidate/20261024T080500Z'
S129=BASE/'s129-matched-state-counterfactual/20261024T174515Z'
S119=BASE/'s119-training-constraints/20261024T070600Z'
OUTPUTS=['qacc','qfrc_constraint','qfrc_actuator','actuator_force','foot_fz','load_ratio','contact_force','efc_force']
# Independent fixed, elementwise absolute AND scaled-relative gates; never adapted.
FD_TOL={k:(1e-6,1e-5) for k in OUTPUTS}
FD_TOL['qacc']=(1e-3,1e-5)

def committed_hashes(path):
 manifest=js(path)
 if 'before' in manifest:
  assert set(manifest)=={'before','after','all_unchanged'},'Unknown wrapped manifest schema'
  assert manifest['all_unchanged'] is True and manifest['before']==manifest['after'],'Historical before/after commitment failed'
  manifest=manifest['before']
 assert isinstance(manifest,dict) and manifest,'Empty manifest forbidden'
 assert all(Path(p).is_absolute() and isinstance(h,str) and len(h)==64 for p,h in manifest.items()),'Invalid manifest entries'
 return manifest

def moment(m,d):
 a=np.zeros((m.nu,m.nv))
 for j in range(m.nu):
  s=int(d.moment_rowadr[j]);n=int(d.moment_rownnz[j]);a[j,d.moment_colind[s:s+n]]=d.actuator_moment[s:s+n]
 return a

def main(out):
 assert not out.exists(),'Fresh output required'
 out.mkdir(parents=True)
 prereg=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='S130 instantaneous fixed-S121-state diagnostic, not trajectory or closed-loop gradient',samples=SAMPLES,time_ms=[1.925,2.5,2.625],actuators=ACTS,excluded_actuator3='unmapped, zero activation; no diagnostic or training claim',h=H,half_h=H/2,stencil='Independent central +/-h and +/-h/2, Richardson=(4*Dhalf-Dh)/3; report all, count only verified derivatives',planned=dict(model_load=1,baseline_forward=3,perturb_forward=60,maximum_forward=63,step=0,neural_ticks=0),baseline_tolerance=dict(absolute=1e-8,scaled_relative=1e-6,both_required=True,geometry_absolute=1e-10,geometry_relative=1e-10),fd_tolerance={k:dict(absolute=v[0],scaled_relative=v[1],both_required=True) for k,v in FD_TOL.items()},analytic_derivative_tolerance=dict(absolute=1e-6,scaled_relative=1e-5),differential_dynamics_tolerance=dict(absolute=1e-6,scaled_relative=1e-5),force_reconstruction_tolerance=dict(absolute=1e-10,scaled_relative=1e-10),gates='Any failed baseline: NO perturbations there. Both scales all signed probes must retain nefc/type/id/contact geom/dim/address AND efc_state; warnings zero. Branch change => NONSMOOTH, no derivative count. No threshold changes/retries.',warmstart='Historical warmstart absent: fresh MjData zero for EVERY call, empirical baseline reproduction only; not historical restoration',model='Original numeric before/after snapshots exact; only original RK4 dt25us setting, no tuning',chain='Only S125 reference dactivation/dW 15 columns after full reference_act==S121 check, W36 frozen; no new weight or weight-FD calls. Fixed mechanical state, not closed-loop gradient.',mass='Reconstructed full M saved at verified baseline; no historical M claimed',units='native NOT_ESTABLISHED',forbidden=['mj_step','neural ticks','weight writes','candidate','solver search','new trajectory'])
 dump(out/'PREREG.json',prereg)
 inputs={Path(__file__).resolve(),ROOT/'run_matched_state_counterfactual.py',ROOT/'audit_muscle_reconstructability.py'}
 for root in [A.parent,S129,CHAIN,S119]:
  inputs.update(p for p in root.rglob('*') if p.is_file())
  for p,h in committed_hashes(root/'INPUT_HASHES.json').items():
   assert sha(p)==h,'Frozen source mismatch '+p
   inputs.add(Path(p))
 xmlfiles=set();nodes=resolve(xmltree(XML,xmlfiles));inputs.update(xmlfiles)
 before={str(p):sha(p) for p in sorted(inputs)};dump(out/'INPUT_HASHES.json',before)
 z=load(A/'TRAJECTORY.npz');r=load(A/'CONSTRAINT_ROWS.npz');chain=load(CHAIN/'LOCAL_CHAIN.npz')
 chain_gate=error(chain['reference_act'],z['act'],1e-14,1e-14)
 assert chain_gate['passed'] and chain['activation_derivative'].shape==(121,90,15)
 assert np.array_equal(chain['reference_weights'],z['weights'])
 assert set(chain['weight_indices'][:,0])=={3,15,38} and not np.any(chain['weight_indices'][:,0]==36)
 assert np.max(np.abs(np.delete(chain['activation_derivative'],ACTS,axis=1)))==0
 assert np.all(z['act'][SAMPLES][:,ACTS]>H) and np.all(z['act'][SAMPLES][:,ACTS]<1-H)
 pars={}
 for key,n in [('lengthrange',2),('gainprm',10),('biasprm',10)]:
  a=np.zeros((90,n))
  for j,node in enumerate(nodes):
   v=list(map(float,node['resolved'][key].split()));a[j,:len(v)]=v
  pars[key]=a
 ledger=dict(model_load=0,forward=0,baseline_forward=0,perturb_forward=0,step=0,neural_ticks=0)
 def reserve(key):
  ledger[key]+=1
  assert ledger['model_load']<=1 and ledger['forward']<=63
  dump(out/'CALL_LEDGER.json',ledger)
 import mujoco as mj
 for name in ['control','act_dyn','act_gain','act_bias','passive']:
  getter=getattr(mj,'get_mjcb_'+name,None);assert getter is None or getter() is None
 reserve('model_load');m=mj.MjModel.from_xml_path(str(XML));pre=snapshot(m)
 assert pre==js(A/'MODEL_PROVENANCE.json')['before']
 m.opt.integrator=mj.mjtIntegrator.mjINT_RK4;m.opt.timestep=.000025
 post=snapshot(m);assert post==js(A/'MODEL_PROVENANCE.json')['after']
 dump(out/'MODEL_AUDIT.json',dict(version=mj.__version__,before=pre,after=post,warmstart_saved='qacc_warmstart' in z))
 assert not any([m.nmocap,m.neq,m.nplugin,m.nuserdata]) and 'qacc_warmstart' not in z
 legs=['lf','lm','lh','rf','rm','rh'];geom_leg={}
 for g in range(m.ngeom):
  gn=(m.geom(g).name or '').lower();bn=(m.body(int(m.geom_bodyid[g])).name or '').lower()
  for li,leg in enumerate(legs):
   if 'tars' in gn+bn and (gn.startswith(leg) or bn.startswith(leg)):geom_leg[g]=li
 floors=np.flatnonzero(m.geom_type==mj.mjtGeom.mjGEOM_PLANE);assert len(floors)==1;floor=int(floors[0])
 weight=float(m.body_mass.sum()*np.linalg.norm(m.opt.gravity));assert weight>0
 dump(out/'FOOT_MAP.json',dict(order=legs,geom_leg=geom_leg,floor=floor,weight_native=weight))
 saved={};gates=[];details=[];valid=0;chain_saved={'weight_indices':chain['weight_indices'],'samples':np.asarray(SAMPLES),'actuators':np.asarray(ACTS)}
 def call(i,act=None):
  d=mj.MjData(m)
  for key in ['qpos','qvel','act','ctrl','qfrc_applied','xfrc_applied']:getattr(d,key)[:]=z[key][i]
  if act is not None:d.act[:]=act
  d.time=float(z['time'][i]);warm=d.qacc_warmstart.copy();assert np.all(warm==0)
  frozen={key:getattr(d,key).copy() for key in ['qpos','qvel','act','ctrl']};time=d.time
  reserve('baseline_forward' if act is None else 'perturb_forward');reserve('forward');mj.mj_forward(m,d)
  assert time==d.time and all(np.array_equal(getattr(d,key),v) for key,v in frozen.items())
  c={key:getattr(d,key).copy() for key in ['qacc','qfrc_constraint','qfrc_actuator','qfrc_passive','qfrc_bias','actuator_force','actuator_length','actuator_velocity','efc_force','efc_type','efc_id','efc_state','solver_niter']}
  c.update(act=d.act.copy(),warmstart_before=warm,warnings=np.asarray([w.number for w in d.warning]),warning_lastinfo=np.asarray([w.lastinfo for w in d.warning]),nefc=np.asarray(d.nefc),ncon=np.asarray(d.ncon),actuator_moment=moment(m,d))
  for key in ['geom1','geom2','dim','efc_address','dist','pos','frame','friction']:
   c['contact_'+key]=np.asarray([getattr(x,key) for x in d.contact])
  # MuJoCo exposes contact.frame flattened; S121 stores explicit 3x3.
  c['contact_frame']=c['contact_frame'].reshape(d.ncon,3,3)
  c['contact_force']=np.zeros((d.ncon,6));foot=np.zeros((6,3))
  for ci,x in enumerate(d.contact):
   mj.mj_contactForce(m,d,ci,c['contact_force'][ci])
   g1,g2=int(x.geom1),int(x.geom2)
   if floor in [g1,g2]:
    fg=g2 if g1==floor else g1
    if fg in geom_leg:foot[geom_leg[fg]]+=(1 if g1==floor else -1)*(x.frame.reshape(3,3).T@c['contact_force'][ci,:3])
  c['foot_fz']=foot[:,2];c['load_ratio']=np.asarray(foot[:,2].sum()/weight)
  assert not mj.mj_isSparse(m),'Original S121 model expected dense constraint Jacobian'
  c['efc_J_dense']=d.efc_J.copy().reshape(d.nefc,48)
  return d,c
 def keep(prefix,c):
  for key,value in c.items():saved[prefix+'_'+key]=value
  np.savez_compressed(out/'RAW_PROBES.npz',**saved)
 identity=['nefc','ncon','efc_type','efc_id','efc_state','contact_geom1','contact_geom2','contact_dim','contact_efc_address']
 for i in SAMPLES:
  d,base=call(i);prefix=f'sample{i:03d}';keep(prefix+'_baseline',base);rp=prefix+'_'
  gate={k:error(base[k],z[k][i]) for k in ['qacc','qfrc_constraint','qfrc_actuator','qfrc_passive','qfrc_bias','actuator_force','actuator_length','actuator_velocity','actuator_moment']}
  gate['warnings']=dict(passed=bool(np.array_equal(base['warnings'],z['warnings'][i]) and not np.any(base['warnings'])))
  gate['nefc']=dict(passed=int(base['nefc'])==int(r['nefc'][i]))
  gate['ncon']=dict(passed=int(base['ncon'])==int(z['contact_ncon'][i]))
  for key in ['efc_type','efc_id','contact_geom1','contact_geom2','contact_dim','contact_efc_address']:
   gate[key]=dict(passed=bool(np.array_equal(base[key],r[rp+key])))
  for key in ['contact_pos','contact_frame']:gate[key]=error(base[key],r[rp+key],1e-10,1e-10)
  gate['efc_force']=error(base['efc_force'],r[rp+'efc_force'])
  gate['efc_J_dense']=error(base['efc_J_dense'],r[rp+'efc_J_dense'],1e-10,1e-10)
  gate['J_transpose_force']=error(base['efc_J_dense'].T@base['efc_force'],base['qfrc_constraint'],1e-10,1e-10)
  gate['contact_force']=error(base['contact_force'],z['contact_force_raw6'][i,:d.ncon])
  gate['foot_fz']=error(base['foot_fz'],z['foot_force'][i,:,2]);gate['load_ratio']=error(base['load_ratio'],z['load_ratio'][i])
  gains=np.empty(90);bias=np.empty(90)
  for j in range(90):gains[j],bias[j],*_=gain_bias(base['actuator_length'][j],base['actuator_velocity'][j],pars['lengthrange'][j],pars['gainprm'][j],pars['biasprm'][j])
  gate['analytic_force']=error(gains*z['act'][i]+bias,base['actuator_force'],1e-10,1e-10)
  ok=all(v['passed'] for v in gate.values());gates.append(dict(sample=i,passed=ok,checks=gate));dump(out/'BASELINE_GATES.json',gates)
  if not ok:continue
  M=np.empty((48,48));mj.mj_fullM(m,M,d.qM);keep(prefix+'_mechanics',dict(mass=M,gain=gains,bias=bias))
  J={key:[] for key in OUTPUTS};sample_valid=[]
  for act in ACTS:
   probes=[]
   for label,delta in [('plus_h',H),('minus_h',-H),('plus_half',H/2),('minus_half',-H/2)]:
    a=z['act'][i].copy();a[act]+=delta;_,c=call(i,a);keep(prefix+f'_act{act}_'+label,c);probes.append(c)
   stable=all(all(np.array_equal(c[k],base[k]) for k in identity) for c in probes)
   warnings=all(not np.any(c['warnings']) for c in probes)
   checks={};dh={};half={};rich={}
   for key in OUTPUTS:
    if all(c[key].shape==base[key].shape for c in probes):
     dh[key]=(probes[0][key]-probes[1][key])/(2*H);half[key]=(probes[2][key]-probes[3][key])/H;rich[key]=(4*half[key]-dh[key])/3
     checks[key]=error(dh[key],half[key],*FD_TOL[key])
    else:checks[key]=dict(passed=False,reason='shape change');rich[key]=np.full_like(base[key],np.nan)
   analytic_tau=base['actuator_moment'][act]*gains[act];analytic_force=np.zeros(90);analytic_force[act]=gains[act]
   checks['analytic_tau']=error(half['qfrc_actuator'],analytic_tau,1e-6,1e-5)
   checks['analytic_force']=error(half['actuator_force'],analytic_force,1e-6,1e-5)
   checks['differential_dynamics']=error(M@half['qacc'],half['qfrc_actuator']+half['qfrc_constraint'],1e-6,1e-5)
   # Per-probe analytic force, not just cancellation-prone differences.
   checks['probe_force']=dict(passed=all(error(gains*c['act']+bias,c['actuator_force'],1e-10,1e-10)['passed'] for c in probes))
   passed=stable and warnings and all(v['passed'] for v in checks.values());valid+=int(passed);sample_valid.append(passed)
   margin=dict(activation_lower=float(z['act'][i,act]-H),activation_upper=float(1-z['act'][i,act]-H),min_abs_efc_force=float(np.min(np.abs(base['efc_force']),initial=np.inf)),min_abs_contact_distance=float(np.min(np.abs(base['contact_dist']),initial=np.inf)),note='Margins are diagnostics, not proofs of smooth solver branches; identity and efc_state checked separately')
   details.append(dict(sample=i,actuator=act,passed=bool(passed),active_set_stable=stable,warnings_zero=warnings,label='VERIFIED_LOCAL' if passed else ('NONSMOOTH' if not stable else 'FD_OR_ANALYTIC_GATE_FAILED'),checks=checks,branch_margin=margin));dump(out/'FD_CHECKS.json',details)
   keep(prefix+f'_act{act}_derivatives',{**{'h_'+k:v for k,v in dh.items()},**{'half_'+k:v for k,v in half.items()},**{'richardson_'+k:v for k,v in rich.items()},'analytic_tau':analytic_tau,'analytic_force':analytic_force,'fixed_constraint_direct_qacc':np.linalg.solve(M,analytic_tau),'valid':np.asarray(passed)})
   for key in OUTPUTS:J[key].append(rich[key])
  # Do not expose an unverified weight chain as a valid gradient.
  chain_saved[prefix+'_activation_derivative']=chain['activation_derivative'][i,ACTS,:]
  chain_saved[prefix+'_valid']=np.asarray(all(sample_valid))
  if all(sample_valid):
   for key in OUTPUTS:
    jac=np.stack(J[key],axis=-1);chain_saved[prefix+'_activation_J_'+key]=jac
    chain_saved[prefix+'_weight_J_'+key]=jac@chain['activation_derivative'][i,ACTS,:]
 np.savez_compressed(out/'FIXED_STATE_CHAIN.npz',**chain_saved)
 assert snapshot(m)==post and before=={p:sha(p) for p in before}
 dump(out/'FROZEN_BEFORE_AFTER.json',dict(passed=True,inputs=len(before)))
 summary=dict(output=str(out),baseline_passed=sum(g['passed'] for g in gates),baseline_count=3,verified_activation_columns=valid,planned_columns=15,ledger=ledger,chain_reference_gate=chain_gate,full_closed_loop_gradient=False,new_weights=False,physical_preheat=False,units='NOT_ESTABLISHED')
 dump(out/'SUMMARY.json',summary)
 (out/'REPORT.zh-CN.md').write_text('# S130 固定机械状态接触响应诊断\n\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n\n仅三个历史S121状态局部activation偏导。RAW_PROBES保留所有原始正负探针；严格原baseline门失败处不执行FD；主动集改变标NONSMOOTH，不计导数。FD_CHECKS保存所有双尺度/解析力/动力学门失败，不放宽阈值。M为本次重建非历史保存。FIXED_STATE_CHAIN仅在该状态五列全过时连接S125参考15列dactivation/dW，W36冻结；不含状态演化、神经反馈、后续接触变化或完整闭环梯度。无候选/新W/轨迹推进；非预热、站立或行走成功。\n')
 dump(out/'OUTPUT_HASHES.json',{str(p):sha(p) for p in out.iterdir() if p.is_file()})
 print(json.dumps(summary))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path);args=p.parse_args();main(args.output or BASE/'s130-contact-aware-activation-response'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
