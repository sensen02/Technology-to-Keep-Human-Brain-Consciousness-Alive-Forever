"""Bounded saved-state mj_forward force ledger. Never steps or runs neurons."""
from pathlib import Path
import os,json,hashlib,sys
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_one_second_motor_pair import sha,dump
from run_body_timestep_diagnostic import OUT as INPUT
import engine.canonical_physics as cp
OUT=Path('/home/sensen/Desktop/engine修复验收/s2-early-force-balance/20261027T160000Z')
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((INPUT/'MANIFEST_SHA256.json').read_text());assert all(sha(INPUT/k)==v for k,v in manifest.items())
 dump(OUT/'RESTORE_CONTRACT.json',dict(input_manifest_sha256=sha(INPUT/'MANIFEST_SHA256.json'),model=str(cp.CANONICAL_XML),model_sha256=sha(cp.CANONICAL_XML),restore=['qpos','qvel','act','ctrl','time','xfrc_applied'],not_saved=['qacc_warmstart','historical solver workspace','qfrc_applied (runner leaves default zero)','mocap state (model no mocap)','historical contact frames'],calls='5 forwards per dt, no step or neural',selection='initial,5ms,10ms, crossing90pct immediately preceding and at first crossing; dt25 same physical times nearest row',tolerance='1e-10 kinematic/activation-dependent muscle values expected; solver/contact 1e-6 absolute diagnostic not equality guarantee; all actual errors retained',equation='M qacc = qfrc_applied + Jbody^T xfrc + qfrc_actuator + qfrc_passive - qfrc_bias + qfrc_constraint'))
 fine=np.load(INPUT/'dt12p5us/normal/BODY.npz');cross=int(np.flatnonzero(fine['thorax_position'][:,2]<.9*fine['thorax_position'][0,2])[0]);targets=[0.,.005,.010,float(fine['time'][cross-1]),float(fine['time'][cross])]
 rows=[];details={};forwards=0
 for tag,dt in [('dt12p5us',.0000125),('dt25us',.000025)]:
  m=cp.load_model();m.opt.timestep=dt;assert m.nmocap==0 and m.neq==0
  z=np.load(INPUT/tag/'normal/BODY.npz');saved_contacts=json.loads((INPUT/tag/'normal/CONTACTS.json').read_text());mass=float(m.body_mass.sum())
  geomleg={}
  for g in range(m.ngeom):
   name=(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or '').lower();body=(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g])) or '').lower()
   for leg in ['lf','lm','lh','rf','rm','rh']:
    if 'tars' in name+body and (name.startswith(leg) or body.startswith(leg)):geomleg[g]=leg
  for target in targets:
   idx=int(np.argmin(abs(z['time']-target)))
   if tag=='dt25us' and target==targets[-2]:idx=int(np.searchsorted(z['time'],target,side='right')-1)
   d=mujoco.MjData(m);d.qpos[:]=z['qpos'][idx];d.qvel[:]=z['qvel'][idx];d.act[:]=z['activation'][idx];d.ctrl[:]=z['command'][idx];d.time=z['time'][idx];d.xfrc_applied[:]=z['external_force'][idx]
   mujoco.mj_forward(m,d);forwards+=1
   M=np.empty((m.nv,m.nv));mujoco.mj_fullM(m,M,d.qM)
   # Generalized contributions; root free translation lies at first three dofs.
   terms={'negative_bias':-d.qfrc_bias.copy(),'passive':d.qfrc_passive.copy(),'muscle':d.qfrc_actuator.copy(),'constraint':d.qfrc_constraint.copy(),'explicit_generalized':d.qfrc_applied.copy(),'explicit_cartesian':np.zeros(m.nv)}
   for b in range(1,m.nbody):mujoco.mj_applyFT(m,d,d.xfrc_applied[b,:3],d.xfrc_applied[b,3:],d.xipos[b],b,terms['explicit_cartesian'])
   residual=M@d.qacc-sum(terms.values());floorforce=np.zeros(3);ground=[];perleg={};normal_errors=[];position_errors=[]
   for k in range(d.ncon):
    con=d.contact[k];a,b=int(con.geom1),int(con.geom2);pa=m.geom_type[a]==mujoco.mjtGeom.mjGEOM_PLANE;pb=m.geom_type[b]==mujoco.mjtGeom.mjGEOM_PLANE
    if not(pa or pb):continue
    f=np.zeros(6);mujoco.mj_contactForce(m,d,k,f);world=con.frame.reshape(3,3).T@f[:3];onfly=world if pa else -world;floorforce+=onfly;foot=b if pa else a;leg=geomleg.get(foot,'NONFOOT');perleg[leg]=perleg.get(leg,0)+1
    matches=[r for r in saved_contacts[idx] if int(r[1])==a and int(r[2])==b]
    match=min(matches,key=lambda r:np.linalg.norm(np.array(r[4:7])-con.pos)) if matches else None
    if match is not None:normal_errors.append(abs(f[0]-match[7]));position_errors.append(float(np.max(abs(np.array(match[4:7])-con.pos))))
    ground.append(dict(geom1=a,geom2=b,leg=leg,frame=con.frame,local_force=f,force_world_on_fly=onfly,normal_world=con.frame[:3],position=con.pos))
   # COM acceleration from rigid-body Jacobian derivatives; no acceleration forward stage missing.
   comacc=np.zeros(3)
   for b in range(1,m.nbody):
    jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);jdp=np.zeros_like(jp);jdr=np.zeros_like(jp)
    mujoco.mj_jacBodyCom(m,d,jp,jr,b);mujoco.mj_jacDot(m,d,jdp,jdr,d.xipos[b],b)
    comacc+=m.body_mass[b]*(jp@d.qacc+jdp@d.qvel)/mass
   # Gravity-only generalized vector and velocity bias are distinct, no extra forward needed.
   gravity=np.zeros(m.nv)
   for b in range(1,m.nbody):
    jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jacBodyCom(m,d,jp,jr,b);gravity+=jp.T@(m.body_mass[b]*m.opt.gravity)
   details[tag+'_'+str(idx)]=dict(terms=terms,gravity_generalized=gravity,velocity_bias=terms['negative_bias']-gravity,Mqacc=M@d.qacc,residual=residual,qacc=d.qacc,contacts=ground,solver_niter=d.solver_niter)
   row=dict(tag=tag,index=idx,target_time=target,time=float(d.time),height=float(d.xpos[m.body('Thorax').id,2]),root_acc=d.qacc[:6],com_acc=comacc,ground_force_world=floorforce,gravity_force=mass*m.opt.gravity,com_external_balance_residual=mass*comacc-floorforce-mass*m.opt.gravity,ground_contacts=perleg,max_activation=float(d.act.max()),max_muscle_force=float(np.max(abs(d.actuator_force))),force_saved_max_error=float(np.max(abs(d.actuator_force-z['muscle_force'][idx]))),thorax_saved_error=float(np.max(abs(d.xpos[m.body('Thorax').id]-z['thorax_position'][idx]))),contact_normal_saved_max_error=max(normal_errors,default=0),contact_position_saved_max_error=max(position_errors,default=0),equation_residual_max=float(np.max(abs(residual))),muscle_root_generalized_force=d.qfrc_actuator[:6],root_acc_contribution={k:np.linalg.solve(M,v)[:6] for k,v in terms.items()})
   foot_error=[]
   for leg in perleg:
    if leg=='NONFOOT':continue
    ids=[g for g,v in geomleg.items() if v==leg];j=['lf','lm','lh','rf','rm','rh'].index(leg);foot_error.append(np.max(abs(d.geom_xpos[ids].mean(0)-z['foot_position'][idx,j])))
   row['foot_saved_error']=float(max(foot_error,default=0));rows.append(row)
 dump(OUT/'FORCE_LEDGER.json',rows);dump(OUT/'GENERALIZED_TERMS.json',details)
 fig,axs=plt.subplots(2,2,figsize=(13,9))
 for tag in ('dt12p5us','dt25us'):
  rs=[r for r in rows if r['tag']==tag];t=[r['time']*1000 for r in rs]
  axs[0,0].plot(t,[r['height'] for r in rs],'o-',label=tag);axs[0,1].plot(t,[r['com_acc'][2] for r in rs],'o-',label=tag)
  axs[1,0].plot(t,[r['ground_force_world'][2]/abs(r['gravity_force'][2]) for r in rs],'o-',label=tag)
  axs[1,1].plot(t,[r['root_acc_contribution']['muscle'][2] for r in rs],'o-',label=tag+' muscle M^-1f');axs[1,1].plot(t,[r['root_acc_contribution']['negative_bias'][2] for r in rs],':',label=tag+' -bias M^-1f')
 for ax,title in zip(axs.flat,['Thorax height / model units','Whole-body COM vertical acceleration / model units/s2','World ground vertical force / model weight','Instantaneous root z accel contribution / NOT removal experiment']):ax.set_title(title);ax.set_xlabel('physical ms');ax.grid(alpha=.2);ax.legend(fontsize=7)
 fig.suptitle('Saved-state forward only: early free-body force balance / no unique causal claim');fig.tight_layout();fig.savefig(OUT/'EARLY_FORCE_BALANCE.png',dpi=140);plt.close(fig);Path('/home/sensen/Desktop/S2_EARLY_FORCE_BALANCE.png').write_bytes((OUT/'EARLY_FORCE_BALANCE.png').read_bytes())
 assert all(sha(INPUT/k)==v for k,v in manifest.items())
 dump(OUT/'EXECUTION.json',dict(command='MUJOCO_GL=egl '+sys.executable+' '+str(Path(__file__)),exit_code=0,model_loads=2,mj_forward=forwards,mj_step=0,neural_step=0,source_manifest_unchanged=True));(OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(json.dumps(rows,default=lambda x:x.tolist()));print('saved-state force ledger complete')
if __name__=='__main__':main()
