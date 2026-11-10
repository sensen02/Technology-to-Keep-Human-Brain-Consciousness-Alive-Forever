"""Independent serial-tree FK and central-difference Jacobian validation."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
MAP=ROOT/'experiments/flybody-teacher/20261107T180844Z/mapping'
def load(n):return json.loads((P/n).read_text())
def rot(q):
    w,x,y,z=np.asarray(q)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def axisrot(v,t):
    x,y,z=v;K=np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])
    return np.eye(3)+np.sin(t)*K+(1-np.cos(t))*K@K
def fk(model,q):
    nb=model['counts']['nbody'];p=np.zeros((nb,3));r=np.tile(np.eye(3),(nb,1,1))
    for b in model['bodys'][1:]:
        i=b['id'];pa=b['parentid'];p[i]=p[pa]+r[pa]@b['pos'];r[i]=r[pa]@rot(b['quat'])
        for ji in range(b['jntadr'],b['jntadr']+b['jntnum']):
            j=model['joints'][ji];a=j['qposadr']
            if j['type']==0:p[i]=q[a:a+3];r[i]=rot(q[a+3:a+7]);continue
            assert j['type']==3
            anchor=p[i]+r[i]@j['pos'];r[i]=r[i]@axisrot(j['axis'],q[a]-j['qpos0']);p[i]=anchor-r[i]@j['pos']
    return p,r
G=load('GEOMETRY.json');model=json.loads((MAP/'current_model.json').read_text());official=json.loads((MAP/'official_model.json').read_text())
ids=np.array(G['landmark_body_ids']);local=np.array(G['landmark_body_local']);qa=np.array(G['qa']);th=next(b['id'] for b in model['bodys'] if b['name']=='Thorax');source=np.load(P/'SOURCE_LANDMARKS.npz')
checks={};fk_calls=0
for tag,fn in [('teacher','TARGETS.npz'),('measured','MEASURED_TARGETS.npz')]:
    z=np.load(P/fn);Q=z['targets_rad'];bounds=np.array(G['joint_limits_rad']);assert Q.shape==(1112,42)
    assert z['ordered_joint_names'].tolist()==G['joint_names'];assert np.array_equal(z['first_pose_targets_rad'],Q[0]);assert np.all(np.isfinite(Q))
    assert np.max(bounds[:,0]-Q)<=1e-10 and np.max(Q-bounds[:,1])<=1e-10
    rates=np.diff(Q,axis=0)/np.diff(z['time_s'])[:,None];assert np.max(abs(rates))<=30+1e-7
    # Exact piecewise linear interpolation onto 1ms preserves limits and finite differences.
    one_ms=np.empty((2223,42));one_ms[::2]=Q;one_ms[1::2]=(Q[:-1]+Q[1:])/2
    assert np.max(abs(np.diff(one_ms,axis=0))/.001)<=30+1e-7
    achieved=[]
    for qj in Q:
        q=np.array(model['poses']['key0']['qpos']);q[qa]=qj;p,r=fk(model,q);fk_calls+=1
        xyz=p[ids]+np.einsum('...ij,...j->...i',r[ids],local);achieved.append((xyz-p[th])@r[th])
    achieved=np.array(achieved);e=np.max(abs(achieved-z['current_achieved_landmarks']));assert e<1e-9
    # Rebuild targets independently from source directions, frozen current lengths and registered axes.
    raw=source[tag+'_landmarks'][:1112];dirs=np.diff(raw,axis=2);dirs/=np.linalg.norm(dirs,axis=-1,keepdims=True);dirs=dirs@np.array(G['registration_source_to_current_rotation']).T
    target=np.empty_like(raw);target[:,:,0]=G['current_coxa_anchors']
    for i in range(8):target[:,:,i+1]=target[:,:,i]+dirs[:,:,i]*np.array(G['current_segment_lengths_native'])[None,:,i,None]
    te=np.max(abs(target-z['target_landmarks']));assert te<1e-12
    ne=np.linalg.norm(achieved-target,axis=-1)/np.array(G['current_leg_lengths_native'])[None,:,None]
    assert np.max(abs(ne-z['normalized_landmark_error']))<1e-12
    checks[tag]=dict(independent_fk_max_abs_error=float(e),independent_target_max_abs_error=float(te),max_rate_rad_s=float(np.max(abs(rates))),finite_limits_rate_pass=True,geometry_gate=bool(ne.max()<=.1 and np.sqrt(np.mean(ne**2))<=.05),max_normalized_landmark_error=float(ne.max()),rms_normalized_landmark_error=float(np.sqrt(np.mean(ne**2))),holdout_max=float(ne[-222:].max()))
# Independently reconstruct source FK at fixed samples using official saved model and actual qpos/name mapping.
srcdir=Path('/home/sensen/Desktop/engine修复验收/flybody-teacher/20261107T180844Z/measured_longest_07718');body=np.load(srcdir/'BODY.npz');ref=np.load(srcdir/'REFERENCE.npz');jn=source['mocap_joint_names'];jd={j['name']:j for j in official['joints']};oth=next(b['id'] for b in official['bodys'] if b['name']=='walker/thorax');oid=source['landmark_body_ids'];olocal=source['landmark_body_local'];sourceerr=0.
for tag,qs in [('teacher',body['qpos']),('measured',ref['qpos'])]:
    for t in [0,111,555,889,1111]:
        q=np.array(official['poses']['key0']['qpos'])
        if tag=='teacher':q[:]=qs[t]
        else:
            q[:7]=qs[t,:7]
            for k,n in enumerate(jn[1:]):q[jd[str(n)]['qposadr']]=qs[t,7+k]
        p,r=fk(official,q);fk_calls+=1;v=(p[oid]+np.einsum('...ij,...j->...i',r[oid],olocal)-p[oth])@r[oth]
        sourceerr=max(sourceerr,float(np.max(abs(v-source[tag+'_landmarks'][t]))))
assert sourceerr<1e-9
# Independent current runtime Jacobian: every42 joint, fixed5 saved teacher poses.
import mujoco as mj
assert mj.__version__=='3.9.0';counts={k:0 for k in ['mj_forward','mj_kinematics','mj_comPos','mj_jac','mj_step','mj_step1','mj_step2']}
for key in counts:
    original=getattr(mj,key)
    def wrap(*a,_key=key,_o=original,**kw):
        counts[_key]+=1
        if _key.startswith('mj_step'):raise RuntimeError('no step')
        return _o(*a,**kw)
    setattr(mj,key,wrap)
sys.path.insert(0,str(ROOT/'experiments/s5-ann-incremental/20261107T010000Z'))
from incremental_actuator import load_full_model
m=load_full_model();d=mj.MjData(m);z=np.load(P/'TARGETS.npz');dof=np.array(G['da']);jerr=0.;mjerr=0.
for t in [0,111,555,889,1111]:
    d.qpos[:]=model['poses']['key0']['qpos'];d.qpos[qa]=z['targets_rad'][t];mj.mj_kinematics(m,d);mj.mj_comPos(m,d);base=d.qpos.copy();R=d.xmat[th].reshape(3,3).copy()
    for i in range(42):
        leg=i//7;b=int(ids[leg,-1]);point=d.xpos[b]+d.xmat[b].reshape(3,3)@local[leg,-1];jp=np.zeros((3,m.nv));mj.mj_jac(m,d,jp,None,point,b);an=jp[:,dof[i]].copy();vv=[]
        for shift in [-1e-6,1e-6]:
            d.qpos[:]=base;d.qpos[qa[i]]+=shift;mj.mj_kinematics(m,d);vv.append(d.xpos[b]+d.xmat[b].reshape(3,3)@local[leg,-1])
        jerr=max(jerr,float(np.max(abs(an-(vv[1]-vv[0])/2e-6))));d.qpos[:]=base;mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
    xyz=d.xpos[ids]+np.einsum('...ij,...j->...i',d.xmat[ids].reshape(6,9,3,3),local);got=(xyz-d.xpos[th])@R;mjerr=max(mjerr,float(np.max(abs(got-z['current_achieved_landmarks'][t]))))
    assert d.time==0
assert jerr<1e-6 and mjerr<1e-9
for ledger in ['OFFICIAL_EXTRACTION.json','GEOMETRY.json']:
    for path,h in load(ledger)['source_hashes'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
out=dict(passed=True,datasets=checks,source_sample_fk_max_error=sourceerr,independent_serial_fk_calls=fk_calls,jacobian_fd_max_abs_error=jerr,current_runtime_saved_fk_max_error=mjerr,current_runtime_calls=counts,current_model_compiles=1,step_calls=0,training_updates=0,root_reference_not_deployed=True)
(P/'VERIFICATION.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
