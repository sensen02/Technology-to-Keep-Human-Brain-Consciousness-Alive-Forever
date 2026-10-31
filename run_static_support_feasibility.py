"""S165 bounded static support feasibility (native-soft audit + rigid upper bound).
One bounded run, no mj_step and no neural integration. Native forward is used only to
verify saved S154 states and to evaluate fixed-qpos activation endpoints. Rigid LP is
explicitly a relaxation, never a native-soft feasibility claim.
"""
from pathlib import Path
import json, hashlib, csv
import numpy as np
from scipy.optimize import linprog
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
SRC=Path('/home/sensen/Desktop/engine修复验收/s154-activation-rate/20261024T215751775489Z')
XML=Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
OUT=Path('/home/sensen/Desktop/engine修复验收/s165-static-support')
STATES=(12,60); ATOL=1e-9

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def jdump(p,x): Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def arr(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def main():
    if OUT.exists(): raise RuntimeError('output exists; one fresh run only')
    OUT.mkdir(parents=True); jdump(OUT/'PREREG.json',dict(scope='S165/P1 bounded static support',states=list(STATES),bounds=[0.,.5,1.],dt=25e-6,tau_act=.001,tau_deact=.0004,neural_ticks=0,mj_step=0,activation_application='diagnostic copies only; no optimized activation is applied to a trajectory',native_soft='forward/contact forces are solver-determined and not LP variables',rigid_relaxation='contact efc rows at actual qpos, nonnegative native pyramidal edge forces; active joint-limit rows retained as nonnegative variables; negative-gap rows only'))
    z=arr(SRC/'init80ms/TRAJECTORY.npz'); r=arr(SRC/'init80ms/CONSTRAINT_ROWS.npz')
    import mujoco as mj
    if mj.__version__!='3.9.0': raise RuntimeError(mj.__version__)
    m=mj.MjModel.from_xml_path(str(XML)); d=mj.MjData(m)
    # S154 numeric runtime contract: apply RK4 and 25 us after native load, before forward.
    m.opt.integrator=mj.mjtIntegrator.mjINT_RK4; m.opt.timestep=25e-6
    if (m.nu!=90 or m.nv!=48): raise RuntimeError('model mismatch')
    native=[]; relax=[]
    def restore(i):
        for n in ('time','qpos','qvel','act','ctrl','qfrc_applied','xfrc_applied'): 
            if n=='time': d.time=float(z[n][i])
            else:getattr(d,n)[:]=z[n][i]
        d.qacc_warmstart[:]=z['native_qacc_warmstart'][i]
    def dense_rows(i):
        p=f'sample{i:03d}_'; return r[p+'efc_J_dense'],r[p+'efc_type'],r[p+'efc_id'],r[p+'contact_efc_address']
    for i in STATES:
        restore(i); mj.mj_forward(m,d)
        M=np.zeros((48,48)); mj.mj_fullM(m,M,d.qM); Ma=M@d.qacc
        checks={k:float(np.max(np.abs(getattr(d,k)-z[k][i]))) for k in ('qacc','qfrc_actuator','qfrc_passive','qfrc_constraint','qfrc_bias')}
        checks['Ma']=float(np.max(np.abs(Ma-z['Ma'][i])))
        J,typ,eid,addr=dense_rows(i)
        checks['JTF']=float(np.max(np.abs(J.T@d.efc_force-d.qfrc_constraint)))
        checks['residual']=float(np.max(np.abs(Ma-(d.qfrc_actuator+d.qfrc_passive+d.qfrc_constraint-d.qfrc_bias))))
        native.append(dict(state=i,time_s=float(d.time),checks=checks,contacts=int(z['contact_ncon'][i]),max_gap=float(np.min(z['contact_dist'][i,:z['contact_ncon'][i]]))) )
        if max(checks.values())>ATOL: jdump(OUT/'NATIVE_DYNAMICS_FAILED.json',native); return
        # Save full verification inputs before any endpoint diagnostic.
        extra={n:np.asarray(getattr(d,n)).copy() for n in ('efc_R','efc_aref','efc_D','efc_frictionloss') if hasattr(d,n)}
        np.savez_compressed(OUT/f'NATIVE_STATIC_INPUTS_{i:03d}.npz',qpos=d.qpos.copy(),qvel=d.qvel.copy(),act=d.act.copy(),ctrl=d.ctrl.copy(),qacc=d.qacc.copy(),qfrc_actuator=d.qfrc_actuator.copy(),qfrc_passive=d.qfrc_passive.copy(),qfrc_constraint=d.qfrc_constraint.copy(),qfrc_bias=d.qfrc_bias.copy(),Ma=Ma,J=J,efc_force=d.efc_force.copy(),efc_type=typ,efc_id=eid,contact_efc_address=addr,contact_dist=z['contact_dist'][i],contact_dim=z['contact_dim'][i],contact_frame=z['contact_frame'][i],contact_geom_ids=z['contact_geom_ids'][i],contact_friction=z['contact_friction'][i],contact_margin=z['contact_dist'][i]+0.0,**extra)
        for b in (0.,.5,1.):
            restore(i); d.act[:]=b; d.ctrl[:]=b; mj.mj_forward(m,d)
            # Retain only actual floor contacts with negative gap; include active joint limits.
            J,typ,eid,addr=dense_rows(i); keep=[]
            n=int(z['contact_ncon'][i])
            for c in range(n):
                a=int(addr[c]); dim=int(z['contact_dim'][i,c]); rows=dim if dim in (1,3,4,6) else {3:4,4:6,6:6}.get(dim,1)
                # S154 contact_efc_address points to first row; condim3 is native pyramid 4 rows.
                if a>=0 and z['contact_dist'][i,c]<=0: keep.extend(range(a,a+rows))
            keep.extend([q for q,t in enumerate(typ) if int(t)==3])
            keep=sorted(set(q for q in keep if 0<=q<J.shape[0]))
            rhs=-(d.qfrc_actuator+d.qfrc_passive-d.qfrc_bias)
            A=J[keep].T if keep else np.zeros((48,0)); bounds=[(0,None)]*len(keep)
            sol=linprog(np.zeros(len(keep)),A_eq=A,b_eq=rhs,bounds=bounds,method='highs')
            residual=float(np.max(np.abs(A@sol.x-rhs))) if sol.success else None
            relax.append(dict(state=i,bound=b,status=int(sol.status),message=sol.message,success=bool(sol.success),nvars=len(keep),residual=residual,negative_gap_contacts=sum(int(z['contact_dist'][i,c]<0) for c in range(n))))
    jdump(OUT/'NATIVE_DYNAMICS.json',native); jdump(OUT/'RIGID_RELAXATION_LP.json',relax)
    jdump(OUT/'SUMMARY.json',dict(status='COMPLETED',native_soft_conclusion='native qacc/contact rows verified at saved states; native-soft force is solver-determined, not freely assignable',rigid_relaxation_conclusion='qualified upper-bound distribution-room only; does not certify native-soft static feasibility or repair penetrated gaps',native=native,rigid_relaxation=relax,source_sha256={str(SRC/'init80ms/TRAJECTORY.npz'):sha(SRC/'init80ms/TRAJECTORY.npz'),str(SRC/'init80ms/CONSTRAINT_ROWS.npz'):sha(SRC/'init80ms/CONSTRAINT_ROWS.npz'),str(XML):sha(XML)}))
    (OUT/'REPORT.zh-CN.md').write_text('# S165 bounded static support feasibility\n\nNative-soft forward verification passed before endpoint diagnostics. The LP is a rigid-contact relaxation and is reported only as a qualified upper bound; no optimized activation was applied to any trajectory.\n')
    print(OUT)
if __name__=='__main__': main()
