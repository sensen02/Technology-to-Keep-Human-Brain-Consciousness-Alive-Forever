"""Run separately under each model's existing runtime; never steps a simulation."""
import sys, json, hashlib, importlib.util
from pathlib import Path
import numpy as np
import mujoco as mj
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
OUT=Path(__file__).resolve().parent
OFF=ROOT/'experiments/flybody-teacher/20261107T174640Z/official-flybody'
LEGACY=Path('/home/sensen/Desktop/engine修复验收/flybody-teacher/20261107T174640Z/official_straight_10s_v2')
mode=sys.argv[1]
counts={k:0 for k in ['mj_step','mj_step1','mj_step2','mj_forward','mj_jac','mj_resetDataKeyframe']}
for key in counts:
    original=getattr(mj,key)
    def wrap(*args,_key=key,_original=original,**kwargs):
        counts[_key]+=1
        if _key in ('mj_step','mj_step1','mj_step2'): raise RuntimeError('No-step audit forbids '+_key)
        return _original(*args,**kwargs)
    setattr(mj,key,wrap)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,o):p.write_text(json.dumps(o,indent=2,ensure_ascii=False,allow_nan=False,default=lambda v:v.tolist() if hasattr(v,'tolist') else str(v))+'\n')
def name(kind,i):return mj.mj_id2name(m,getattr(mj.mjtObj,'mjOBJ_'+kind.upper()),int(i)) or ''
if mode=='official':
    assert mj.__version__=='3.1.6'
    source=LEGACY/'MODEL.mjb';m=mj.MjModel.from_binary_path(str(source))
    identity=json.loads((LEGACY/'MODEL_IDENTITY.json').read_text())
    actionnames=identity['action_names'];thorax=m.body('walker/thorax').id
    sourcefiles=[source,LEGACY/'MODEL_IDENTITY.json',OFF/'flybody/fruitfly/assets/fruitfly.xml',OFF/'flybody/fruitfly/fruitfly.py',OFF/'flybody/fly_envs.py',OFF/'flybody/tasks/base.py',OFF/'flybody/tasks/task_utils.py']
else:
    assert mj.__version__=='3.9.0',mj.__version__
    entry=ROOT/'experiments/s5-ann-walk/20261107T123634Z/run_residual.py'
    spec=importlib.util.spec_from_file_location('audit_current_entry',entry);entrymod=importlib.util.module_from_spec(spec);spec.loader.exec_module(entrymod)
    n=entrymod.n
    from incremental_actuator import load_full_model,DEFAULT_MODEL_PATH
    source=DEFAULT_MODEL_PATH;m=load_full_model();thorax=m.body('Thorax').id
    actionnames=[name('actuator',i) for i in range(m.nu)]
    sourcefiles=[source,entry]
    for mod in list(sys.modules.values())+[entrymod,n]:
        p=getattr(mod,'__file__',None)
        if p and Path(p).is_file() and (str(ROOT) in str(p)) and str(p).endswith('.py'):sourcefiles.append(Path(p))
    # Dynamic importlib modules are not always registered in sys.modules.
    walk=entrymod
    while hasattr(walk,'b'):
        walk=walk.b
        if getattr(walk,'__file__',None):sourcefiles.append(Path(walk.__file__))
    dump(OUT/'current_runtime_contract.json',dict(model_path=source,neural_runtime=n.__file__,gait=n.GAIT,physics_dt=n.DT,command_dt=n.CDT,policy_dt=n.PDT,rate_cap=n.CAP,gain_scale=n.IncrementalActuator.scale))
real=lambda i: mode=='current' or name('body',i).startswith('walker/')
rows={}
for kind,number,prefix,fields in [
 ('joint',m.njnt,'jnt',['type','bodyid','qposadr','dofadr','pos','axis','limited','range','stiffness','margin','solref','solimp']),
 ('actuator',m.nu,'actuator',['trntype','trnid','dyntype','gaintype','biastype','actadr','actnum','dynprm','gainprm','biasprm','gear','ctrllimited','ctrlrange','forcelimited','forcerange','lengthrange']),
 ('body',m.nbody,'body',['parentid','rootid','jntnum','jntadr','pos','quat','ipos','iquat','mass','inertia','geomnum','geomadr']),
 ('geom',m.ngeom,'geom',['bodyid','type','dataid','pos','quat','size','contype','conaffinity','condim','friction','solref','solimp','margin','gap','rgba'])]:
    table=[]
    for i in range(number):
        r={'id':i,'name':name(kind,i)}
        for f in fields:
            arr=getattr(m,prefix+'_'+f,None)
            if arr is not None:r[f]=np.asarray(arr[i]).tolist()
        if kind=='joint':
            adr=int(m.jnt_qposadr[i]);dof=int(m.jnt_dofadr[i])
            r.update(qpos0=float(m.qpos0[adr]),qpos_spring=float(m.qpos_spring[adr]),damping=float(m.dof_damping[dof]),armature=float(m.dof_armature[dof]),frictionloss=float(m.dof_frictionloss[dof]),body_name=name('body',m.jnt_bodyid[i]))
        if kind=='actuator':
            short=r['name'].split('/')[-1];r['action_index']=actionnames.index(short) if mode=='official' else i
            trnid=int(m.actuator_trnid[i,0]);target_kind={0:'joint',1:'joint',2:'slidercrank',3:'tendon',4:'site',5:'body'}[int(m.actuator_trntype[i])]
            r['target_kind']=target_kind;r['target_name']=name(target_kind,trnid)
        table.append(r)
    rows[kind+'s']=table
rows['options']={k:np.asarray(getattr(m.opt,k)).tolist() for k in ['timestep','gravity','integrator','cone','solver','iterations','noslip_iterations','tolerance','disableflags','enableflags']}
rows['counts']={k:int(getattr(m,k)) for k in ['nq','nv','nu','na','njnt','nbody','ngeom','nmesh','neq','ntendon','nkey']}
rows['counts']['real_hinges']=sum(m.jnt_type[i]==3 and real(int(m.jnt_bodyid[i])) for i in range(m.njnt))
rows['counts']['real_mass']=float(sum(m.body_mass[i] for i in range(m.nbody) if real(i)))
rows['tendons']=[]
for i in range(m.ntendon):
    wraps=[]
    for w in range(m.tendon_adr[i],m.tendon_adr[i]+m.tendon_num[i]):
        typ=int(m.wrap_type[w]);oid=int(m.wrap_objid[w])
        wraps.append(dict(type=typ,objid=oid,parameter=float(m.wrap_prm[w]),name=name('joint' if typ==1 else 'site' if typ==3 else 'geom',oid) if oid>=0 else ''))
    rows['tendons'].append(dict(id=i,name=name('tendon',i),wraps=wraps,range=m.tendon_range[i],stiffness=float(m.tendon_stiffness[i]),damping=float(m.tendon_damping[i])))
rows['action_order']=actionnames
rows['runtime']={'python':sys.executable,'mujoco':mj.__version__,'model_source':str(source),'model_source_sha256':sha(source)}
rows['meshes']=[]
for i in range(m.nmesh):
    vertices=m.mesh_vert[m.mesh_vertadr[i]:m.mesh_vertadr[i]+m.mesh_vertnum[i]]
    rows['meshes'].append(dict(id=i,name=name('mesh',i),vertices=len(vertices),faces=int(m.mesh_facenum[i]),compiled_vertices_sha256=hashlib.sha256(vertices.tobytes()).hexdigest(),aabb_min=vertices.min(0),aabb_max=vertices.max(0)))
poses={};fd=[]
for pose in ['qpos0','key0']:
    d=mj.MjData(m)
    if pose=='key0':mj.mj_resetDataKeyframe(m,d,0)
    mj.mj_forward(m,d)
    base=d.qpos.copy();R=d.xmat[thorax].reshape(3,3).copy();origin=d.xpos[thorax].copy()
    poses[pose]=dict(qpos=base,thorax_world_pos=origin,thorax_world_rotation=R,
        body_positions_thorax=(d.xpos-origin)@R,body_rotations_thorax=np.einsum('ij,bjk->bik',R.T,d.xmat.reshape(-1,3,3)),
        joint_axes_thorax=d.xaxis@R,joint_anchors_thorax=(d.xanchor-origin)@R)
    for j in range(m.njnt):
        if m.jnt_type[j]!=mj.mjtJoint.mjJNT_HINGE or not real(int(m.jnt_bodyid[j])):continue
        # Select a distal descendant (furthest qpos0 origin) to test nontrivial lever arm.
        body=int(m.jnt_bodyid[j]);desc=[]
        for b in range(1,m.nbody):
            a=b
            while a and a!=body:a=int(m.body_parentid[a])
            if a==body:desc.append(b)
        target=max(desc,key=lambda b:np.linalg.norm(d.xpos[b]-d.xanchor[j]))
        # Body-local offset also tests a terminal body that has no distal origin.
        local=np.array([.001,.002,.003]);point=d.xpos[target]+d.xmat[target].reshape(3,3)@local
        jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mj.mj_jac(m,d,jp,jr,point,target)
        analytic=jp[:,m.jnt_dofadr[j]].copy();values=[]
        for delta in [-1e-6,1e-6]:
            scratch=mj.MjData(m);scratch.qpos[:]=base;scratch.qpos[m.jnt_qposadr[j]]+=delta
            mj.mj_forward(m,scratch)
            assert scratch.time==0
            values.append(scratch.xpos[target]+scratch.xmat[target].reshape(3,3)@local)
        numerical=(values[1]-values[0])/2e-6
        fd.append(dict(pose=pose,joint=name('joint',j),target_body=name('body',target),local_probe=local,
            analytic=analytic,numerical=numerical,max_abs_error=float(np.max(np.abs(analytic-numerical)))))
rows['poses']=poses
rows['finite_difference']=dict(epsilon_rad=1e-6,rows=fd,max_abs_error=max(r['max_abs_error'] for r in fd),passed=all(r['max_abs_error']<1e-7 for r in fd))
rows['measured_calls']=counts;rows['environment_steps']=0;rows['training_updates']=0
rows['source_hashes']={str(p.resolve()):sha(p) for p in sorted(set(sourcefiles))}
assert rows['finite_difference']['passed']
assert counts['mj_step']==counts['mj_step1']==counts['mj_step2']==0
assert rows['counts']['nu']==(59 if mode=='official' else 42)
dump(OUT/(mode+'_model.json'),rows)
print(json.dumps(dict(model=mode,counts=rows['counts'],calls=counts,fd_max=rows['finite_difference']['max_abs_error']),default=lambda x:int(x)))
