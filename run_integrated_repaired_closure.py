"""One explicit derived repaired configuration, one live free-body neural rollout."""
from pathlib import Path
import os,json,sys,time,traceback,xml.etree.ElementTree as ET
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,mujoco
from unittest.mock import patch
import run_neural_muscle_closure as c
from run_command1ms_single import Observer
from run_one_second_motor_pair import sha,dump
from engine.embodied.runtime_pool_guard import load_runtime_pool
from engine.embodied.initial_pose_contract import model_limit_report
from flygym.compose.fly.musculoskeletal import _load_mjcf
P=Path(__file__).parent
O=Path('/home/sensen/Desktop/engine修复验收/s4-integrated-repaired-closure/20261028T032000Z')
S=Path('/home/sensen/Desktop/engine修复验收/s2-stance-s3b/20261023T060000Z')
DURATION=1.;DT=.0000125

def identities(m):
 joints={};geom={}
 for leg in c.LEGS:
  ids=[j for j in range(m.njnt) if m.joint(j).name.lower().startswith('joint_'+leg)];tib=[j for j in ids if 'tibia' in m.joint(j).name.lower()];assert len(ids)==7 and len(tib)==1
  joints[leg]=dict(q=[int(m.jnt_qposadr[j]) for j in ids],tib_q=int(m.jnt_qposadr[tib[0]]),tib_v=int(m.jnt_dofadr[tib[0]]))
 for g in range(m.ngeom):
  nm=m.geom(g).name.lower();bn=m.body(int(m.geom_bodyid[g])).name.lower();legs=[leg for leg in c.LEGS if 'tars' in nm+bn and(nm.startswith(leg) or bn.startswith(leg))]
  if len(legs)==1:geom[g]=legs[0]
 assert set(geom.values())==set(c.LEGS)
 return joints,geom

def foot_load(m,q,geom):
 d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_forward(m,d);feet=np.zeros(6);non=0
 for k in range(d.ncon):
  co=d.contact[k];a,b=int(co.geom1),int(co.geom2);pa=m.geom_type[a]==mujoco.mjtGeom.mjGEOM_PLANE;pb=m.geom_type[b]==mujoco.mjtGeom.mjGEOM_PLANE
  if not(pa or pb):continue
  g=b if pa else a;f=np.zeros(6);mujoco.mj_contactForce(m,d,k,f);fz=float(((co.frame.reshape(3,3).T@f[:3])*(1 if pa else -1))[2])
  if g in geom:feet[c.LEGS.index(geom[g])]+=fz
  else:non+=fz
 return d,feet,non

def main():
 O.mkdir(parents=True,exist_ok=False)
 sources=[Path(__file__),Path(c.__file__),Path(c.cp.CANONICAL_XML),S/'STANCE2_CALIBRATED.npz',S/'S3V2_MODEL_a_springref_stance.xml',c.POOLS,P/'engine/embodied/runtime_pool_guard.py',P/'engine/embodied/motor_output.py',P/'engine/neural_cond.py',P/'engine/embodied/sensory_channels.py',P/'run_command1ms_single.py',P/'run_quarter_second_pair.py',P/'data/flywire/banc_connectome.npz']
 hashes={str(p):sha(p) for p in sources}
 dump(O/'PREREGISTRATION.json',dict(source_hashes=hashes,one_rollout=True,duration_s=DURATION,dt_body_s=DT,dt_neural_s=.0005,dt_sensory_command_s=.005,seed=0,base='canonical65d allphysics retained',changes=['S3V2 saved stance','required joint ranges from matching stance XML','42 springref aligned to chosen stance','folded LW/RW body transforms only','all90 LR rescaled by newL/oldL to preserve canonical-keyframe fractional position, width ratio and physical force law'],excluded=['185 NOT_ACCEPTED attachments','old10xgain','oldactivationtau','oldcontactstiffness','external controller','CPG','PD','rootpin','act preload','neural background','new connectivity','outcome gain search'],initial_seating='static rootz only bounded +/-0.1 model units, up to40bisection forwards targetfootFz=weight; if bracket fails use original saved stance and disclose unbalanced load; never afterrelease',success_gate='last200ms nonfoot fraction0, COM height>=0.8 initial, meanfootFz/weight>=0.8; gate is support screening not walking certification',scope='Manual engineering composite, not physiological validation; one change bundle not causal factor test'))
 counts=dict(model_load=0,forward=0,step=0,neural_step=0,network_instances=0,rollouts=0);real_fw=mujoco.mj_forward;real_st=mujoco.mj_step;real_ns=c.ConductanceNetwork.step;real_ni=c.ConductanceNetwork.__init__;obs=None;error=None;start=time.time()
 def fw(*a,**kw):counts['forward']+=1;return real_fw(*a,**kw)
 def st(*a,**kw):counts['step']+=1;return real_st(*a,**kw)
 def ns(*a,**kw):counts['neural_step']+=1;return real_ns(*a,**kw)
 def ni(*a,**kw):counts['network_instances']+=1;return real_ni(*a,**kw)
 try:
  with patch.object(mujoco,'mj_forward',fw),patch.object(mujoco,'mj_step',st),patch.object(c.ConductanceNetwork,'step',ns),patch.object(c.ConductanceNetwork,'__init__',ni):
   # Builder-only reference model, no dynamics.
   old=c.cp.load_model();counts['model_load']+=1;od=mujoco.MjData(old);mujoco.mj_resetDataKeyframe(old,od,0);mujoco.mj_forward(old,od);Lold=od.actuator_length.copy();oldlr=old.actuator_lengthrange.copy();q=np.load(S/'STANCE2_CALIBRATED.npz')['qpos'].copy()
   tree=ET.parse(c.cp.CANONICAL_XML);root=tree.getroot();fixed=ET.parse(S/'S3V2_MODEL_a_springref_stance.xml').getroot();fj={j.get('name'):j for j in fixed.iter('joint') if j.get('name')};changes=[]
   for j in root.iter('joint'):
    n=j.get('name')
    if not n or n not in fj or j.get('type')=='free':continue
    adr=int(old.jnt_qposadr[old.joint(n).id]);previous=dict(j.attrib);j.set('range',fj[n].get('range'));j.set('springref',format(float(q[adr]),'.17g'));changes.append(dict(kind='joint',name=n,before=previous,after=dict(j.attrib)))
   for b in root.iter('body'):
    if b.get('name') not in ('LWing','RWing'):continue
    source=next(x for x in fixed.iter('body') if x.get('name')==b.get('name'));before=dict(b.attrib)
    for attr in ('pos','quat'):b.set(attr,source.get(attr))
    changes.append(dict(kind='wingbody',name=b.get('name'),before=before,after=dict(b.attrib)))
   # FlyGym resolves existing relative mesh filenames through its verified cache.
   root.find('option').set('timestep',format(DT,'.17g'))
   model_path=O/'INTEGRATED_MODEL.xml';tree.write(model_path,encoding='utf-8',xml_declaration=True)
   m=_load_mjcf(str(model_path)).compile();counts['model_load']+=1;m.opt.timestep=DT;assert m.nmesh==71 and m.nq==49 and m.nv==48 and m.nu==90 and m.neq==0
   d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_forward(m,d);scale=d.actuator_length/Lold;newlr=oldlr*scale[:,None];assert np.isfinite(newlr).all() and(scale>0).all()
   for i,a in enumerate(root.findall('./actuator/*')):a.set('lengthrange',' '.join(format(float(v),'.17g') for v in newlr[i]))
   # LR only updates existing compiled parameter, then persist identical XML values.
   m.actuator_lengthrange[:]=newlr;tree.write(model_path,encoding='utf-8',xml_declaration=True)
   joints,geom=identities(m);weight=float(m.body_mass.sum()*abs(m.opt.gravity[2]));originalq=q.copy();seat=[]
   def sample(dz):
    qq=originalq.copy();qq[2]+=dz;dd,ff,nn=foot_load(m,qq,geom);seat.append(dict(dz=dz,foot_fz=float(ff.sum()),nonfoot_fz=nn));return qq,dd,ff,nn
   lo,hi=-.1,.1;ql,dl,fl,nl=sample(lo);qh,dh,fh,nh=sample(hi);bracketed=bool(fl.sum()>=weight>=fh.sum())
   if bracketed:
    for _ in range(35):
     mid=(lo+hi)/2;qq,dd,ff,nn=sample(mid)
     if ff.sum()>weight:lo=mid
     else:hi=mid
    q,d,feet,non=sample((lo+hi)/2)
   else:q,d,feet,non=sample(0.)  # never deepen or search beyond the declared bound
   keys=root.find('keyframe')
   if keys is None:keys=ET.SubElement(root,'keyframe')
   for k in list(keys):keys.remove(k)
   ET.SubElement(keys,'key',name='integrated_repaired_stance',qpos=' '.join(format(float(v),'.17g') for v in q))
   tree.write(model_path,encoding='utf-8',xml_declaration=True)
   # Independent final-file compile verifies persisted timestep, LR and keyframe.
   final_model=_load_mjcf(str(model_path)).compile();counts['model_load']+=1
   np.testing.assert_array_equal(final_model.actuator_lengthrange,m.actuator_lengthrange)
   np.testing.assert_allclose(final_model.key_qpos[0],q,rtol=0,atol=1e-15)
   assert final_model.opt.timestep==DT
   compiled_invariants={}
   for field in ['actuator_gainprm','actuator_biasprm','actuator_dynprm','actuator_acc0','actuator_length0','body_mass','body_inertia','geom_solref','geom_solimp','geom_friction']:
    np.testing.assert_array_equal(getattr(final_model,field),getattr(m,field));compiled_invariants[field]=True
   m=final_model
   assert np.all(feet>0) and abs(non)<1e-9
   if bracketed:assert abs(feet.sum()/weight-1)<1e-4
   gate=c.cp.stance_facts(m,d).as_dict();gate['initial_scalar_joint_ranges']=model_limit_report(m,q);assert gate['initial_scalar_joint_ranges']['within_declared_scalar_ranges'];c.require_six_foot_initial_contacts(m,q,geom)
   d.act[:]=0;mujoco.mj_forward(m,d);f0=d.actuator_force.copy();d.act[:]=1;mujoco.mj_forward(m,d);f1=d.actuator_force.copy();assert np.all(abs(f1-f0)>1e-9),'some muscles have zero active capacity in selected stance'
   beforefrac=(Lold-oldlr[:,0])/(oldlr[:,1]-oldlr[:,0]);afterfrac=(d.actuator_length-newlr[:,0])/(newlr[:,1]-newlr[:,0]);np.testing.assert_allclose(beforefrac,afterfrac,rtol=0,atol=1e-12)
   dump(O/'MODEL_CHANGES.json',dict(changes=changes,lr_scale=scale,old_lr=oldlr,new_lr=newlr,model_sha256=sha(model_path),unmodified_fields=['mass','inertia','all231sitepositions','tendonorder','actuator_gainprm','actuator_biasprm','actuator_dynprm','contactsolref','friction'],engineering_only=True));dump(O/'STATIC_PREFLIGHT.json',dict(gate=gate,weight_native=weight,foot_fz=feet,nonfoot_fz=non,initial_rootz_shift=float(q[2]-originalq[2]),seating_samples=seat,static_load_bracketed=bracketed,initial_foot_load_ratio=float(feet.sum()/weight),fallback='original saved stance unchanged' if not bracketed else None,compiled_invariants=compiled_invariants,all90_active_nonzero=True,operating_fraction_max_error=float(abs(beforefrac-afterfrac).max()),counts=counts.copy()))
   np.savez_compressed(O/'INITIAL.npz',qpos=q,qvel=np.zeros(m.nv),activation=np.zeros(m.na),command=np.zeros(m.nu),prior_stance_qpos=originalq)
   print('STATIC_PREFLIGHT',json.dumps(dict(foot_fz=feet.tolist(),root_shift=float(q[2]-originalq[2]),model_sha=sha(model_path),counts=counts)),flush=True)
   print('READY_FOR_ROLLOUT',flush=True)
   np.random.seed(0);built=c.build_neural_tier(c.TierConfig(background_mean_nA=0,background_sd_nA=0));manifest=c.build_motor_manifest(load_runtime_pool(c.POOLS),built['tier']['root_ids'],[m.actuator(i).name for i in range(m.nu)]);groups,_=c.channel_index_map(int(built['n']),built['tier']['root_ids']);dump(O/'MAPPING.json',manifest)
   np.random.seed(0);obs=Observer(m,geom);d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_forward(m,d);obs(m,d);ev={};counts['rollouts']+=1
   records,tr,cmd,spikes=c.episode(m,q,joints,geom,built,manifest,groups,'normal',DURATION,stride=1,full_detail_intervals=0,push=None,external_touch=False,evidence=ev,body_observer=obs,body_dt_s=DT)
   dump(O/'RECORDS.json',records);np.savez_compressed(O/'INTERVALS.npz',trajectory=tr,commands=cmd,spikes=spikes,**{k:v for k,v in ev.items() if isinstance(v,np.ndarray)})
 except Exception as e:error=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc())
 finally:
  if obs is not None:obs.save(O/'BODY.npz');dump(O/'CONTACTS.json',obs.contacts)
  dump(O/'EXECUTION.json',dict(command=sys.executable+' '+str(Path(__file__)),exit_code=0 if error is None else 1,counts=counts,error=error,runtime_s=time.time()-start));dump(O/'SOURCE_UNCHANGED.json',dict(passed=all(sha(Path(k))==v for k,v in hashes.items())));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
 if error:raise RuntimeError(error)
 print('COMPLETE',O,flush=True)
if __name__=='__main__':main()
