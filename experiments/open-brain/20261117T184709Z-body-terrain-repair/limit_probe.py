"""Static finite-limit constraint test, separate from free-root teacher physics."""
from body_model import *
b=Body(adhesion=True);b.restore(119)
qpos=b.d.qpos.copy();results=[]
for j in range(1,b.m.njnt):
    dof=int(b.m.jnt_dofadr[j]);adr=int(b.m.jnt_qposadr[j])
    for side,offset,expected in ((0,-.05,1.),(1,.05,-1.)):
        b.d.qpos[:]=qpos;b.d.qvel[:]=0.;b.d.qpos[adr]=b.m.jnt_range[j,side]+offset
        mujoco.mj_forward(b.m,b.d)
        rows=np.where((b.d.efc_type[:b.d.nefc]==mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)&(b.d.efc_id[:b.d.nefc]==j))[0]
        checks=[]
        for row in rows:
            start=int(b.d.efc_J_rowadr[row]);n=int(b.d.efc_J_rownnz[row]);columns=b.d.efc_J_colind[start:start+n]
            match=np.where(columns==dof)[0]
            value=float(b.d.efc_J[start+match[0]]) if len(match) else 0.
            checks.append(dict(jacobian=value,force=float(b.d.efc_force[row]),restoring=bool(value*expected>0 and b.d.efc_force[row]>0)))
        results.append(dict(joint=mujoco.mj_id2name(b.m,mujoco.mjtObj.mjOBJ_JOINT,j),side=side,checks=checks,
                            pass_=bool(checks and all(c['restoring'] for c in checks))))
out=dict(status='STATIC_LIMIT_CONSTRAINT_CHECK',pass_=all(r['pass_'] for r in results),conditions=len(results),
         results=results,physics_steps=0,forward_calls=len(results),biological_ranges='UNKNOWN_ENGINEERING_GUARDS',
         source_sha256={str(Path(__file__).resolve()):sha(__file__),str(HERE/'JOINT_LIMITS.json'):sha(HERE/'JOINT_LIMITS.json')})
write(HERE/'LIMIT_PROBE.json',out);print(out['pass_'],out['conditions'],'joint limit constraint directions')
