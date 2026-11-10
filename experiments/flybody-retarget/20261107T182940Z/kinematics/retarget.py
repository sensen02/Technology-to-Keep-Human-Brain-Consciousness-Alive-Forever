"""Preregistered no-step two-runtime extraction and bounded retargeting."""
import os,sys,json,hashlib
from pathlib import Path
import numpy as np
import mujoco as mj
P=Path(__file__).resolve().parent
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
MAP=ROOT/'experiments/flybody-teacher/20261107T180844Z/mapping'
SRC=Path('/home/sensen/Desktop/engine修复验收/flybody-teacher/20261107T180844Z/measured_longest_07718')
MJB=Path('/home/sensen/Desktop/engine修复验收/flybody-teacher/20261107T174640Z/official_straight_10s_v2/MODEL.mjb')
LEGS=['LF','LH','LM','RF','RH','RM']; SUF=['T1_left','T3_left','T2_left','T1_right','T3_right','T2_right']
LAND=['coxa','femur_center','tibia','tarsus1','tarsus2','tarsus3','tarsus4','claw_origin','material_foot']
C={k:0 for k in ['mj_forward','mj_kinematics','mj_comPos','mj_jac','mj_resetDataKeyframe','mj_step','mj_step1','mj_step2']}
for k in C:
    original=getattr(mj,k)
    def call(*a,_k=k,_o=original,**kw):
        C[_k]+=1
        if _k.startswith('mj_step'):raise RuntimeError('Step forbidden')
        return _o(*a,**kw)
    setattr(mj,k,call)
def dump(name,o):
    (P/name).write_text(json.dumps(o,indent=2,allow_nan=False,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rot(q):
    a=np.zeros(9);mj.mju_quat2Mat(a,np.array(q));return a.reshape(3,3)
def frame(a):
    x=(a[0]+a[3]-a[1]-a[4])/2;x=x/np.linalg.norm(x)
    y=a[:3].mean(0)-a[3:].mean(0);y-=x*np.dot(x,y);y/=np.linalg.norm(y)
    return np.column_stack([x,y,np.cross(x,y)])
def landmarks(m,d,ids,local,th):
    rr=d.xmat[th].reshape(3,3)
    xyz=d.xpos[ids]+np.einsum('...ij,...j->...i',d.xmat[ids].reshape(6,9,3,3),local)
    return (xyz-d.xpos[th])@rr

def official():
    assert mj.__version__=='3.1.6'
    m=mj.MjModel.from_binary_path(str(MJB));d=mj.MjData(m);mj.mj_resetDataKeyframe(m,d,0);mj.mj_forward(m,d)
    th=m.body('walker/thorax').id;ids=[];local=np.zeros((6,9,3))
    for l,s in enumerate(SUF):
        names=['coxa','femur','tibia','tarsus','tarsus2','tarsus3','tarsus4','claw','claw']
        ids.append([m.body('walker/'+n+'_'+s).id for n in names])
        g=m.geom('walker/tarsal_claw_'+s+'_collision').id;b=ids[-1][-1]
        # Lowest surface point on capsule in fixed key0 thorax vertical direction.
        Rb=d.xmat[b].reshape(3,3);Rg=rot(m.geom_quat[g]);down=Rb.T@d.xmat[th].reshape(3,3)@np.array([0.,0.,-1.])
        axis=Rg[:,2];local[l,-1]=m.geom_pos[g]+axis*m.geom_size[g,1]*np.sign(np.dot(axis,down))+m.geom_size[g,0]*down
    ids=np.array(ids);base=landmarks(m,d,ids,local,th)
    identity=json.loads((SRC/'MODEL_IDENTITY.json').read_text());jn=identity['mocap_joint_names'];assert len(jn)==103
    qadd=[m.joint(n).qposadr[0] for n in jn[1:]]
    body=np.load(SRC/'BODY.npz');ref=np.load(SRC/'REFERENCE.npz')
    assert body['qpos'].shape==(1112,m.nq) and ref['qpos'].shape==(1176,109)
    payload=dict(leg_names=np.array(LEGS),landmark_names=np.array(LAND),landmark_body_ids=ids,landmark_body_local=local,key0_landmarks=base,anatomical_frame=frame(base[:,0]),mocap_joint_names=np.array(jn))
    for tag,qs in [('teacher',body['qpos']),('measured',ref['qpos'])]:
        values=[];roots=[]
        for q in qs:
            d.qpos[:]=m.qpos0
            if tag=='teacher':d.qpos[:]=q
            else:d.qpos[:7]=q[:7];d.qpos[qadd]=q[7:]
            mj.mj_forward(m,d);assert d.time==0
            values.append(landmarks(m,d,ids,local,th));roots.append(d.qpos[:7].copy())
        payload[tag+'_landmarks']=np.array(values);payload[tag+'_root_qpos']=np.array(roots)
    payload['teacher_time_s']=body['time']-body['time'][0];payload['measured_time_s']=np.arange(1176)*.002
    np.savez_compressed(P/'SOURCE_LANDMARKS.npz',**payload)
    dump('OFFICIAL_EXTRACTION.json',dict(runtime=mj.__version__,calls=C,source_hashes={str(p):sha(p) for p in [MJB,SRC/'BODY.npz',SRC/'REFERENCE.npz',SRC/'MODEL_IDENTITY.json']},model_loads=1,units='official cm',root_reference_only=True))
    print(json.dumps(C),flush=True)

def current():
    from scipy.optimize import least_squares
    assert mj.__version__=='3.9.0'
    sys.path.insert(0,str(ROOT/'experiments/s5-ann-incremental/20261107T010000Z'))
    from incremental_actuator import load_full_model,DEFAULT_MODEL_PATH
    cfg=json.loads((P/'PREREGISTRATION.json').read_text());s=cfg['solver'];m=load_full_model();d=mj.MjData(m)
    mj.mj_resetDataKeyframe(m,d,0);mj.mj_forward(m,d);baseq=d.qpos.copy();th=m.body('Thorax').id
    jids=m.actuator_trnid[:,0];qa=m.jnt_qposadr[jids];da=m.jnt_dofadr[jids];bounds=m.jnt_range[jids].copy();q0=baseq[qa].copy()
    names=np.array([m.joint(int(j)).name for j in jids]);assert names.tolist()==json.loads((MAP/'STUDENT_INTERFACE.json').read_text())['output']['ordered_joint_names']
    ids=[];local=np.zeros((6,9,3));material=[]
    for l,leg in enumerate(LEGS):
        ids.append([m.body(leg+n).id for n in ['Coxa','Trochanter' if leg[1]=='F' else 'Femur','Tibia','Tarsus1','Tarsus2','Tarsus3','Tarsus4','Tarsus5','Tarsus5']])
        b=ids[-1][-1];gg=[g for g in range(m.ngeom) if m.geom_bodyid[g]==b and m.geom_type[g]==mj.mjtGeom.mjGEOM_MESH];assert len(gg)==1
        g=gg[0];mi=m.geom_dataid[g];v=m.mesh_vert[m.mesh_vertadr[mi]:m.mesh_vertadr[mi]+m.mesh_vertnum[mi]].astype(float)
        vb=v@rot(m.geom_quat[g]).T+m.geom_pos[g];vw=vb@d.xmat[b].reshape(3,3).T+d.xpos[b]
        vi=int(np.argmin(((vw-d.xpos[th])@d.xmat[th].reshape(3,3))[:,2]));local[l,-1]=vb[vi]
        material.append(dict(leg=leg,geom=m.geom(g).name,compiled_mesh_vertex=vi,body_local=local[l,-1]))
    ids=np.array(ids);base=landmarks(m,d,ids,local,th);length=np.linalg.norm(np.diff(base,axis=1),axis=-1);L=length.sum(1)
    src=np.load(P/'SOURCE_LANDMARKS.npz');R=frame(base[:,0])@src['anatomical_frame'].T
    assert abs(np.linalg.det(R)-1)<1e-12 and np.max(abs(R.T@R-np.eye(3)))<1e-12
    source_lengths=np.linalg.norm(np.diff(src['key0_landmarks'],axis=1),axis=-1)
    geometry=dict(registration_source_to_current_rotation=R,determinant=np.linalg.det(R),source_anatomical_frame=src['anatomical_frame'],current_anatomical_frame=frame(base[:,0]),current_coxa_anchors=base[:,0],current_segment_lengths_native=length,source_segment_lengths_cm=source_lengths,current_div_source_nominal_cm10=length/(10*source_lengths),current_leg_lengths_native=L,current_material_points=material,landmark_body_ids=ids,landmark_body_local=local,current_root_scratch_qpos=baseq[:7],joint_names=names,joint_limits_rad=bounds,qa=qa,da=da,source_hashes={str(DEFAULT_MODEL_PATH):sha(DEFAULT_MODEL_PATH),str(P/'PREREGISTRATION.json'):sha(P/'PREREGISTRATION.json'),str(P/'SOURCE_LANDMARKS.npz'):sha(P/'SOURCE_LANDMARKS.npz')})
    dump('GEOMETRY.json',geometry)
    for tag,filename,kind in [('teacher','TARGETS.npz','teacherderived_NOT_rawanimal'),('measured','MEASURED_TARGETS.npz','measured_reference_engineering_retarget')]:
        raw=src[tag+'_landmarks'][:1112];u=np.diff(raw,axis=2);u/=np.linalg.norm(u,axis=-1,keepdims=True);u=u@R.T
        target=np.concatenate([np.broadcast_to(base[:,0,None],(1112,6,1,3)),np.broadcast_to(base[:,0,None],(1112,6,1,3))+np.cumsum(u*length[None,:,:,None],axis=2)],axis=2)
        Q=np.zeros((1112,42));ach=np.zeros_like(target);status=np.zeros((1112,6),int);nfev=np.zeros_like(status)
        for t in range(1112):
            for l,leg in enumerate(LEGS):
                ix=np.arange(l*7,l*7+7);prev=q0[ix] if t==0 else Q[t-1,ix];lo=bounds[ix,0].copy();hi=bounds[ix,1].copy()
                if t:lo=np.maximum(lo,prev-.06);hi=np.minimum(hi,prev+.06)
                cache={}
                def evaluate(q):
                    if 'q' in cache and np.array_equal(cache['q'],q):return cache['r'],cache['j']
                    d.qpos[:]=baseq;d.qpos[qa[ix]]=q;mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
                    rr=d.xmat[th].reshape(3,3);world=d.xpos[ids[l]]+np.einsum('nij,nj->ni',d.xmat[ids[l]].reshape(9,3,3),local[l]);pos=(world-d.xpos[th])@rr
                    jac=[]
                    for b,p in zip(ids[l],world):
                        jp=np.zeros((3,m.nv));mj.mj_jac(m,d,jp,None,p,int(b));jac.append(rr.T@jp[:,da[ix]])
                    jac=np.array(jac);delta=np.diff(pos,axis=0);lens=np.linalg.norm(delta,axis=1);dirs=delta/lens[:,None];dj=np.diff(jac,axis=0)
                    du=np.einsum('nij,njk->nik',(np.eye(3)[None]-dirs[:,:,None]*dirs[:,None,:])/lens[:,None,None],dj)
                    r=np.concatenate([((pos-target[t,l])/L[l]).ravel(),(.05*(dirs-u[t,l])).ravel(),.02*(q-prev),.002*(q-q0[ix])])
                    J=np.concatenate([(jac/L[l]).reshape(-1,7),(.05*du).reshape(-1,7),.02*np.eye(7),.002*np.eye(7)],axis=0)
                    cache.update(q=q.copy(),r=r,j=J,pos=pos);return r,J
                res=least_squares(lambda q:evaluate(q)[0],np.clip(prev,lo+1e-12,hi-1e-12),jac=lambda q:evaluate(q)[1],bounds=(lo,hi),method='trf',max_nfev=s['max_nfev_first'] if t==0 else s['max_nfev_subsequent'],ftol=s['ftol'],xtol=s['xtol'],gtol=s['gtol'])
                Q[t,ix]=res.x;evaluate(res.x);ach[t,l]=cache['pos'];status[t,l]=res.status;nfev[t,l]=res.nfev
            if t%100==0:print(tag,t,'calls',C['mj_kinematics'],flush=True)
        normerr=np.linalg.norm(ach-target,axis=-1)/L[None,:,None]
        np.savez_compressed(P/filename,time_s=np.arange(1112)*.002,targets_rad=Q,ordered_joint_names=names,first_pose_targets_rad=Q[0],joint_limits_rad=bounds,leg_names=np.array(LEGS),landmark_names=np.array(LAND),target_landmarks=target,current_achieved_landmarks=ach,target_segment_directions=u,current_segment_directions=np.diff(ach,axis=2)/length[None,:,:,None],source_root_qpos=src[tag+'_root_qpos'][:1112],source_frame_ids=np.arange(1112),source_id=np.array('teacher_actual_physical_longest_07718' if tag=='teacher' else 'measured_longest_07718'),reference_id=np.array('measured_longest_07718'),source_kind=np.array(kind),normalized_landmark_error=normerr,leg_lengths_native=L,solver_status=status,solver_nfev=nfev,registration_source_to_current_rotation=R)
        dump(tag.upper()+'_QUALITY.json',dict(max_normalized_error=normerr.max(),rms_normalized_error=np.sqrt(np.mean(normerr**2)),per_leg_max=normerr.max(axis=(0,2)),per_leg_rms=np.sqrt(np.mean(normerr**2,axis=(0,2))),holdout_max=normerr[-222:].max(),holdout_rms=np.sqrt(np.mean(normerr[-222:]**2)),max_rate=np.max(abs(np.diff(Q,axis=0)))/.002,geometry_gate=bool(normerr.max()<=.1 and np.sqrt(np.mean(normerr**2))<=.05),max_nfev_terminations=int(np.sum(status==0)),finite_inbounds=bool(np.isfinite(Q).all() and np.all(Q>=bounds[:,0]-1e-10) and np.all(Q<=bounds[:,1]+1e-10))))
        print('SAVED',str(P/filename),flush=True)
        dump('CURRENT_EXECUTION.json',dict(runtime=mj.__version__,calls=C,model_compiles=1,completed_sources=['teacher'] if tag=='teacher' else ['teacher','measured']))
if __name__=='__main__':
    {'official':official,'current':current}[sys.argv[1]]()
