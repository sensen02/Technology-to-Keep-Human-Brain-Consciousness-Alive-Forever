"""S129/P1/S2-S4 bounded frozen-state diagnostic, NOT a physical trajectory.
No mj_step, neural tick, weight search, or checkpoint write. Import is inert.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, csv, hashlib, json, xml.etree.ElementTree as ET
import numpy as np
from audit_muscle_reconstructability import gain_bias, xmltree, resolve

ROOT=Path(__file__).resolve().parent
BASE=Path('/home/sensen/Desktop/engine修复验收')
A=BASE/'s121-constraint-row-recording/20261024T072500Z/init80ms'
B=BASE/'s127-joint-candidate-closed-loop/20261024T082500Z/init80ms'
XML=Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
RELATED=[BASE/'s126-fixed-contraction/20261024T081500Z',BASE/'s119-training-constraints/20261024T070600Z',BASE/'s128-closed-loop-failure-boundary/20261024T083500Z']
SELECTED=[0,77,100,105,106,113]
sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
ah=lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
js=lambda p: json.loads(Path(p).read_text())
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvout(p,rows):
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def snapshot(m):
 r={}
 for prefix,obj in [('model',m),('opt',m.opt)]:
  for k in dir(obj):
   if k.startswith('_'):continue
   x=getattr(obj,k)
   if isinstance(x,np.ndarray):r[prefix+'.'+k]=dict(shape=list(x.shape),dtype=str(x.dtype),sha256=ah(x))
   elif isinstance(x,(bool,int,float,np.integer,np.floating)):r[prefix+'.'+k]=x.item() if isinstance(x,np.generic) else x
 return r
def error(a,b,atol=1e-8,rtol=1e-6):
 a,b=np.asarray(a),np.asarray(b)
 if a.shape!=b.shape:return dict(passed=False,shape_a=list(a.shape),shape_b=list(b.shape))
 ab=float(np.max(np.abs(a-b),initial=0)); rel=float(np.max(np.abs(a-b)/np.maximum(1.,np.maximum(np.abs(a),np.abs(b))),initial=0))
 return dict(passed=ab<=atol and rel<=rtol,max_abs=ab,max_scaled_relative=rel)
def main(out):
 assert not out.exists(),'Fresh timestamp output only'
 out.mkdir(parents=True)
 xmlfiles=set();tree=xmltree(XML,xmlfiles);nodes=resolve(tree)
 inputs={Path(__file__).resolve(),ROOT/'audit_muscle_reconstructability.py',*xmlfiles}
 for r in [A.parent,B.parent,*RELATED]:inputs.update(p for p in r.rglob('*') if p.is_file())
 # Honor original commitments, including complete model asset/source identity.
 for r in [A.parent,B.parent]:
  for p,h in js(r/'INPUT_HASHES.json').items():
   assert sha(p)==h,'Frozen input changed '+p
   inputs.add(Path(p))
 before={str(p):sha(p) for p in sorted(inputs)}
 dump(out/'INPUT_HASHES.json',before)
 prereg=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='static matched-state diagnostic NOT trajectory/newW/physical preheat/autonomy',arms=[str(A),str(B)],model_xml=str(XML),nq=49,nv=48,nu=90,offline_samples=121,offline_actuators=90,selected_indices=SELECTED,selected_ms=[i*.025 for i in SELECTED],planned_model_loads=1,planned_baseline_forwards=12,maximum_swap_forwards=12,steps=0,neural_ticks=0,thresholds=dict(baseline_absolute=1e-8,baseline_scaled_relative=1e-6,both_required=True,force_offline_abs=1e-10,decomposition_abs=1e-10,contact_geometry_abs=1e-10),swap_gate='Every selected same-arm force, qfrc_actuator, qfrc_constraint, qacc, all raw contact forces, contact identity/address/dim and geometry must pass independently; failed sample gets NO swap. Missing warmstart left explicit fresh-data zero; never inferred or called historical.',restored_fields=['qpos','qvel','act','ctrl','time','qfrc_applied','xfrc_applied','mocap_pos if saved','mocap_quat if saved','eq_active if saved','qacc_warmstart if saved'],algebra='tauXY=sum_j moment(stateX,j)*[gain(stateX,j)*activationY_j+bias(stateX,j)]; direct=.5[(AB-AA)+(BB-BA)], state=.5[(BA-AA)+(BB-AB)]',mass_policy='Saved full mass only for all121; absent => no invented inverse-M. Independently matched forward may save reconstructed M at selected states, labeled reconstructed.',frozen_constraint_caveat='M^-1 delta tau holds constraint generalized force fixed algebraically; physical contact solver need not hold contact force fixed',no_weight_or_checkpoint_changes=True,units='native; physical units NOT_ESTABLISHED')
 prereg['retained_failed_attempt']=dict(path=str(BASE/'s129-matched-state-counterfactual/20261024T174425Z'),exit_code=1,ledger=dict(model_load=1,forward=2,step=0,neural_ticks=0),reason='mj_fullM binding argument order; fixed to installed API mj_fullM(m,M,d.qM)',historical_manifest_note='Failed directory unchanged; its script hash records pre-fix source, current script differs intentionally')
 dump(out/'PREREG.json',prereg)
 arms=[load(p/'TRAJECTORY.npz') for p in [A,B]];rows=[load(p/'CONSTRAINT_ROWS.npz') for p in [A,B]]
 assert all(z['qpos'].shape==(121,49) and z['qvel'].shape==(121,48) and z['act'].shape==(121,90) for z in arms)
 assert np.array_equal(arms[0]['time'],arms[1]['time'])
 inventory=[]
 for p,z in zip([A,B],arms):
  for k in ['qpos','qvel','act','ctrl','time','qfrc_applied','xfrc_applied','mocap_pos','mocap_quat','eq_active','qacc_warmstart','plugin_state','userdata','qM','mass_matrix']:
   inventory.append(dict(arm=str(p),field=k,saved=k in z,shape=list(z[k].shape) if k in z else None))
 dump(out/'FIELD_AUDIT.json',dict(fields=inventory,warmstart='NOT_SAVED unless field present; cold reconstruction empirical gate mandatory',mass='Full mass NOT_SAVED in source trajectories; Ma is not M',all_npz_inventory=[dict(arm=str(p),fields={k:list(v.shape) for k,v in z.items()}) for p,z in zip([A,B],arms)]))
 pars={}
 for k,n in [('lengthrange',2),('gainprm',10),('biasprm',10)]:
  a=np.zeros((90,n))
  for j,node in enumerate(nodes):
   v=list(map(float,node['resolved'][k].split()));a[j,:len(v)]=v
  for p in [A,B]:
   for phase in ['before','after']:assert js(p/'MODEL_PROVENANCE.json')[phase]['model.actuator_'+k]['sha256']==ah(a)
  pars[k]=a
 gains=[];biases=[];forces=np.empty((2,2,121,90));taus=np.empty((2,2,121,48));torques=np.empty((2,2,121,90,48));checks=[]
 for s,z in enumerate(arms):
  g=np.zeros((121,90));b=g.copy()
  for i in range(121):
   for j in range(90):g[i,j],b[i,j],*_=gain_bias(z['actuator_length'][i,j],z['actuator_velocity'][i,j],pars['lengthrange'][j],pars['gainprm'][j],pars['biasprm'][j])
  gains.append(g);biases.append(b)
  for a in range(2):
   forces[s,a]=g*arms[a]['act']+b
   torques[s,a]=forces[s,a,:,:,None]*z['actuator_moment'];taus[s,a]=torques[s,a].sum(axis=1)
  checks.append(dict(arm=s,force=error(forces[s,s],z['actuator_force'],1e-10,1e-10),torque=error(taus[s,s],z['qfrc_actuator'],1e-10,1e-10)))
 assert all(c['force']['passed'] and c['torque']['passed'] for c in checks)
 direct=.5*((taus[0,1]-taus[0,0])+(taus[1,1]-taus[1,0]));state=.5*((taus[1,0]-taus[0,0])+(taus[1,1]-taus[0,1]));total=taus[1,1]-taus[0,0]
 assert np.max(np.abs(direct+state-total))<1e-10
 np.savez_compressed(out/'OFFLINE_2X2.npz',time=arms[0]['time'],force=forces,torque=taus,signed_per_actuator_torque=torques,torque_AA=torques[0,0],torque_AB=torques[0,1],torque_BA=torques[1,0],torque_BB=torques[1,1],activation_effect=direct,state_effect=state,gain=np.asarray(gains),bias=np.asarray(biases),activation_direct=direct,state_FLV_moment=state,total=total)
 csvout(out/'ALL48_SHAPLEY.csv',[dict(sample=i,time_ms=float(arms[0]['time'][i]*1000),dof=d,tau_AA=taus[0,0,i,d],tau_AB=taus[0,1,i,d],tau_BA=taus[1,0,i,d],tau_BB=taus[1,1,i,d],activation_direct=direct[i,d],state_FLV_moment=state[i,d],total=total[i,d]) for i in range(121) for d in range(48)])
 dump(out/'OFFLINE_VERIFICATION.json',dict(checks=checks,shapley_residual=float(np.max(np.abs(direct+state-total))),mass_inverse='NOT_AVAILABLE: no saved full M'))
 ledger=dict(model_load=0,forward=0,step=0,neural_ticks=0,state_restores=0,activation_swaps=0)
 def reserve(k):ledger[k]+=1;dump(out/'CALL_LEDGER.json',ledger)
 import mujoco as mj
 for name in ['control','act_dyn','act_gain','act_bias','passive']:
  getter=getattr(mj,'get_mjcb_'+name,None)
  assert getter is None or getter() is None,'Callbacks forbidden'
 reserve('model_load');m=mj.MjModel.from_xml_path(str(XML));assert (m.nq,m.nv,m.nu)==(49,48,90)
 # Prove every numeric original model field, not just dimensions.
 pre=snapshot(m)
 for p in [A,B]:assert pre==js(p/'MODEL_PROVENANCE.json')['before'],'Original compiled model mismatch'
 m.opt.integrator=mj.mjtIntegrator.mjINT_RK4;m.opt.timestep=.000025
 post=snapshot(m)
 for p in [A,B]:assert post==js(p/'MODEL_PROVENANCE.json')['after'],'Runtime numeric model mismatch'
 dump(out/'MODEL_AUDIT.json',dict(mujoco_version=mj.__version__,complete_numeric_provenance_match=True,nmocap=m.nmocap,neq=m.neq,nplugin=m.nplugin,nuserdata=m.nuserdata,solver=int(m.opt.solver),iterations=int(m.opt.iterations),tolerance=float(m.opt.tolerance),missing_warmstart='Fresh MjData zero, NOT historical estimate; all baseline outputs must match',empty_state_fields=dict(mocap=m.nmocap==0,equality=m.neq==0,plugin=m.nplugin==0,userdata=m.nuserdata==0)))
 def restore(z,i,activation=None):
  d=mj.MjData(m)
  for key in ['qpos','qvel','act','ctrl','qfrc_applied','xfrc_applied','mocap_pos','mocap_quat','eq_active','qacc_warmstart','plugin_state','userdata']:
   if key in z:getattr(d,key)[:]=z[key][i]
  d.time=float(z['time'][i]);reserve('state_restores')
  if activation is not None:d.act[:]=activation;reserve('activation_swaps')
  return d
 def forward(d):
  frozen={k:getattr(d,k).copy() for k in ['qpos','qvel','act','ctrl']};t=d.time
  reserve('forward');mj.mj_forward(m,d)
  assert d.time==t and all(np.array_equal(getattr(d,k),v) for k,v in frozen.items())
 def capture(d):
  return dict(qacc=d.qacc.copy(),qfrc_actuator=d.qfrc_actuator.copy(),qfrc_constraint=d.qfrc_constraint.copy(),actuator_force=d.actuator_force.copy(),qacc_warmstart_after=d.qacc_warmstart.copy(),solver_niter=d.solver_niter.copy(),nefc=np.asarray(d.nefc),efc_type=d.efc_type.copy(),efc_id=d.efc_id.copy(),efc_force=d.efc_force.copy(),contact_force=np.asarray([np.zeros(6) for _ in range(d.ncon)]).reshape(d.ncon,6))
 results=[];response=[];saved={}
 for s,(z,r) in enumerate(zip(arms,rows)):
  for i in SELECTED:
   d=restore(z,i);warm_before=d.qacc_warmstart.copy();forward(d);c=capture(d);c['qacc_warmstart_before']=warm_before
   for ci in range(d.ncon):mj.mj_contactForce(m,d,ci,c['contact_force'][ci])
   gate={k:error(c[k],z[k][i]) for k in ['qacc','qfrc_actuator','qfrc_constraint','actuator_force']}
   pref=f'sample{i:03d}_';n=int(z['contact_ncon'][i]);identity=d.ncon==n
   for k in ['geom1','geom2','dim','efc_address']:
    identity=identity and np.array_equal(np.asarray([getattr(x,k) for x in d.contact]),r[pref+'contact_'+k])
   gate['contact_identity']=dict(passed=bool(identity))
   gate['efc_force']=error(d.efc_force,r[pref+'efc_force'])
   gate['efc_type_id']=dict(passed=bool(np.array_equal(d.efc_type,r[pref+'efc_type']) and np.array_equal(d.efc_id,r[pref+'efc_id'])))
   gate['contact_position']=error(np.asarray([x.pos for x in d.contact]).reshape(d.ncon,3),r[pref+'contact_pos'],1e-10,1e-10)
   gate['contact_frame']=error(np.asarray([x.frame for x in d.contact]).reshape(d.ncon,3,3),r[pref+'contact_frame'],1e-10,1e-10)
   gate['contact_force']=error(c['contact_force'],z['contact_force_raw6'][i,:n])
   ok=all(v['passed'] for v in gate.values());results.append(dict(state_arm=s,sample=i,time_ms=i*.025,passed=ok,checks=gate,swap_executed=ok))
   dump(out/'FORWARD_BASELINE_GATES.json',results)
   for k,v in c.items():saved[f'arm{s}_sample{i:03d}_baseline_{k}']=v
   if not ok:continue
   # Fresh restored baseline state/warmstart, change activation ONLY, no ctrl swap.
   other=restore(z,i,arms[1-s]['act'][i]);forward(other);cc=capture(other)
   for ci in range(other.ncon):mj.mj_contactForce(m,other,ci,cc['contact_force'][ci])
   assert other.ncon==d.ncon
   assert all((x.geom1,x.geom2,x.dim,x.efc_address)==(y.geom1,y.geom2,y.dim,y.efc_address) for x,y in zip(d.contact,other.contact))
   M=np.empty((48,48));mj.mj_fullM(m,M,d.qM)
   driving=np.linalg.solve(M,cc['qfrc_actuator']-c['qfrc_actuator'])
   saved[f'arm{s}_sample{i:03d}_reconstructed_mass']=M
   saved[f'arm{s}_sample{i:03d}_fixed_constraint_driving_qacc']=driving
   for k,v in cc.items():saved[f'arm{s}_sample{i:03d}_swap_{k}']=v
   for dof in range(48):response.append(dict(state_arm=s,sample=i,time_ms=i*.025,dof=dof,delta_tau=float(cc['qfrc_actuator'][dof]-c['qfrc_actuator'][dof]),fixed_constraint_driving_qacc=float(driving[dof]),actual_solver_delta_qacc=float(cc['qacc'][dof]-c['qacc'][dof]),delta_qfrc_constraint=float(cc['qfrc_constraint'][dof]-c['qfrc_constraint'][dof])))
 np.savez_compressed(out/'STATIC_COUNTERFACTUAL.npz',**saved)
 if response:csvout(out/'STATIC_RESPONSE_ALL48.csv',response)
 assert snapshot(m)==post
 assert before=={p:sha(p) for p in before},'Frozen inputs changed'
 dump(out/'FROZEN_BEFORE_AFTER.json',dict(passed=True,inputs=len(before),sha256=before))
 summary=dict(output=str(out),offline_verified=True,baseline_selected=12,baseline_passed=sum(r['passed'] for r in results),static_swaps=ledger['activation_swaps'],ledger=ledger,all121_mass_inverse_available=False,diagnostic_only=True,physical_preheat=False,new_weights=False,autonomy=False)
 dump(out/'SUMMARY.json',summary)
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(3,1,figsize=(11,9),sharex=True)
 for a,dof in zip(ax,[6,7,8]):
  a.plot(arms[0]['time']*1000,total[:,dof],label='observed total delta tau');a.plot(arms[0]['time']*1000,direct[:,dof],label='matched-state activation direct');a.plot(arms[0]['time']*1000,state[:,dof],label='state FLV + moment');a.set_ylabel('native DOF'+str(dof));a.legend()
 ax[-1].set_xlabel('frozen historical time (ms), not new trajectory');fig.suptitle('S129 exact offline 2x2 Shapley; static contact swaps separately gated');fig.tight_layout();fig.savefig(out/'MATCHED_STATE_SHAPLEY.png',dpi=140);plt.close(fig)
 report='# S129 matched-state counterfactual diagnostic\n\n'+json.dumps(summary,indent=2)+'\n\nAll121 x90 exact muscle arithmetic verified against both historical arms. Symmetric state/activation decomposition is algebraically exact, not a full closed-loop gradient. Selected forward baselines independently gated before activation-only swap. Missing historical warmstart explicitly not reconstructed; failed gates prohibit contact-response claims at those samples. Fresh-state writes are diagnostic interventions, not autonomy or integration. Full saved mass unavailable; reconstructed mass only at baseline-verified selected samples. M-inverse direct driving freezes constraint generalized force algebraically, while actual contact solver can change contact forces. No new W or baseline checkpoint write; S127 rejected and S126 not promoted. Physical units NOT_ESTABLISHED; physical preheat/walking NOT achieved.\n'
 (out/'REPORT.zh-CN.md').write_text(report)
 dump(out/'OUTPUT_HASHES.json',{str(p):sha(p) for p in out.iterdir() if p.is_file()})
 print(json.dumps(summary))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path);a=p.parse_args();main(a.output or BASE/'s129-matched-state-counterfactual'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
