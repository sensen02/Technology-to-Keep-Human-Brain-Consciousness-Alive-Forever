"""Audit model-supported full joint readout without modifying archived inputs."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MUJOCO_GL','egl')
import sys,json,hashlib
from pathlib import Path
import numpy as np
R=Path('/run/media/sensen/Data2/cell_wound_prototype');sys.path.insert(0,str(R))
O=Path('/home/sensen/Desktop/engine修复验收/s5-neural-joint-torque/20261105T064731Z')
OLD=Path('/home/sensen/Desktop/engine修复验收/s5-neural-joint-torque/20261105T062800Z')
SRC=Path('/home/sensen/Desktop/engine修复验收/s5-drive-chain-diagnosis/20261105T054144Z/baseline/EVAL/RUN')
P=Path('/home/sensen/Desktop/engine修复验收/s5-joint-torque-plant/20261105T060730Z')
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.tolist() if hasattr(v,'tolist') else str(v))+'\n')
def main():
    O.mkdir(parents=True,exist_ok=False)
    protected=[SRC/'MODEL.xml',SRC/'MAPPING.json',P/'MODEL_JOINT_TORQUE.xml',OLD/'COVERAGE.json',OLD/'WARM_P0/BODY.npz',Path(__file__)]
    before={str(p):sha(p) for p in protected}
    dump(O/'AUDIT_PREREGISTRATION.json',dict(source_hashes=before,method='A[j,a]=torque_j(activation e_a)-torque_j(activation zero), full 42x90 at keyframe and 25 actual warmup states. Fixed readout coefficient sign(sum_a_in_pool A[j,a]) at keyframe; no dominant-joint truncation.',threshold='per joint 1e-8 * max absolute full pooled coefficient, with absolute floor 1e-12; report 1e-10/1e-6 sensitivity',sign_changes='25 states from last .2s assisted window; diagnostic only, no runtime geometry/PD readout',scope='recover existing muscle effects via existing pools; no new neurons/synapses; numerical model attribution, not anatomy validation',importance='核心必需',units='NOT_ESTABLISHED'))
    import mujoco
    from flygym.compose.fly.musculoskeletal import _load_mjcf
    from scipy.optimize import nnls
    m=_load_mjcf(SRC/'MODEL.xml').compile();d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0)
    jm=_load_mjcf(P/'MODEL_JOINT_TORQUE.xml').compile()
    joint_names=[jm.joint(int(j)).name for j in jm.actuator_trnid[:,0]]
    va=np.array([m.jnt_dofadr[m.joint(n).id] for n in joint_names])
    old=js(OLD/'COVERAGE.json');pools=old['pools'];lookup={tuple(v):i for i,v in enumerate(pools)}
    rows=js(SRC/'MAPPING.json')['rows'];M=np.zeros((90,66))
    for a,r in enumerate(rows):
        key=tuple(r['local_indices'])
        if key:M[a,lookup[key]]=1
    def matrix(q,v):
        d.qpos[:]=q;d.qvel[:]=v;d.act[:]=0;mujoco.mj_forward(m,d);base=d.qfrc_actuator.copy();A=np.zeros((42,90))
        for a in range(90):
            d.act[:]=0;d.act[a]=1;mujoco.mj_forward(m,d);A[:,a]=(d.qfrc_actuator-base)[va]
        d.act[:]=np.linspace(0,1,90);mujoco.mj_forward(m,d)
        error=float(np.max(abs((d.qfrc_actuator-base)[va]-A@d.act)))
        assert error<1e-9
        return A,error
    A,error=matrix(m.key_qpos[0],np.zeros(m.nv));C=A@M
    def signs(C,rel=1e-8):
        tol=np.maximum(np.max(abs(C),axis=1)*rel,1e-12)
        return np.where(abs(C)>tol[:,None],np.sign(C),0)
    S=signs(C);S0=np.asarray(old['sign_matrix']);warm=np.load(OLD/'WARM_P0/BODY.npz')
    indices=np.flatnonzero(warm['time']>=1.1-1e-10);indices=indices[np.linspace(0,len(indices)-1,25).round().astype(int)]
    traces=[];matrices=[];signseq=[]
    for i in indices:
        a,e=matrix(warm['qpos'][i],warm['qvel'][i]);c=a@M;s=signs(c);matrices.append(a);signseq.append(s)
        target=warm['ctrl'][i]/jm.actuator_ctrlrange[:,1]
        r0=nnls(S0,target,maxiter=10000)[1];r1=nnls(S,target,maxiter=10000)[1]
        traces.append(dict(time=float(warm['time'][i]),affine_error=e,dominant_cone_relative_residual=float(r0/max(np.linalg.norm(target),1e-12)),full_cone_relative_residual=float(r1/max(np.linalg.norm(target),1e-12))))
    seq=np.asarray(signseq)
    def counts(s):return dict(uncovered=int((~np.any(s,axis=1)).sum()),bidirectional=int((np.any(s>0,axis=1)&np.any(s<0,axis=1)).sum()),one_direction=int((np.any(s>0,axis=1)^np.any(s<0,axis=1)).sum()),nonzero_coefficients=int(np.count_nonzero(s)),rank=int(np.linalg.matrix_rank(s)))
    per=[]
    for j,name in enumerate(joint_names):
        per.append(dict(joint=name,old_positive=int((S0[j]>0).sum()),old_negative=int((S0[j]<0).sum()),full_positive=int((S[j]>0).sum()),full_negative=int((S[j]<0).sum()),all_muscles_positive=int((A[j]>1e-12).sum()),all_muscles_negative=int((A[j]<-1e-12).sum()),coefficient_nonzero_in_warm_states=int(np.any(seq[:,j]!=0,axis=0).sum()),opposite_to_fixed_sign_entries=int(np.any(seq[:,j]*S[j]<0,axis=0).sum())))
    np.savez_compressed(O/'FULL_MAPPING.npz',muscle_matrix=A,muscle_to_pool=M,pool_matrix=C,sign_matrix=S,dominant_sign_matrix=S0,warm_muscle_matrices=matrices,warm_sign_matrices=seq)
    report=dict(old=counts(S0),full=counts(S),all_muscles=counts(signs(A)),threshold_sensitivity={str(e):counts(signs(C,e)) for e in [1e-10,1e-8,1e-6]},joints=per,samples=traces,keyframe_affine_error=error,sign_flipping_entries=int(np.any(seq*S[None,:,:]<0,axis=0).sum()),new_coefficients=int(((S!=0)&(S0==0)).sum()),changed_existing_signs=int((S*S0<0).sum()),cone_note='Necessary unconstrained nonnegative coefficient cone only; independent instantaneous coefficients are more flexible than shared w/g and neural activity. It does not prove realizable closed-loop control.',units='NOT_ESTABLISHED')
    dump(O/'MAPPING_AUDIT.json',report)
    dump(O/'FULL_COVERAGE.json',dict(pools=pools,sign_matrix=S,joints=[dict(joint=name,positive_pools=int((S[j]>0).sum()),negative_pools=int((S[j]<0).sum()),limit=float(jm.actuator_ctrlrange[j,1])) for j,name in enumerate(joint_names)],**counts(S)))
    after={str(p):sha(p) for p in protected};assert before==after;dump(O/'AUDIT_INPUT_HASHES_AFTER.json',after)
    print(json.dumps({k:report[k] for k in ['old','full','all_muscles','new_coefficients','changed_existing_signs','sign_flipping_entries']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
