"""Observe native adhesion transmission and its contact forces without mutation."""
from body_model import mujoco,np


def pad_effects(b,tick):
    qforce=np.zeros(b.m.nv);wrench=np.zeros((6,6));com=b.d.xipos[b.feet].copy();contact_rows=[]
    contacts=[[] for _ in range(6)]
    body_to_leg={int(body):i for i,body in enumerate(b.feet)}
    for c in b.d.contact[:b.d.ncon]:
        if c.exclude or c.efc_address<0:continue
        for geom,sign in ((c.geom1,1.),(c.geom2,-1.)):
            leg=body_to_leg.get(int(b.m.geom_bodyid[geom]))
            if leg is not None:contacts[leg].append((c.pos.copy(),sign*c.frame[:3].copy(),float(c.dist)))
    for i,a in enumerate(b.adhesion_ids):
        adr=int(b.d.moment_rowadr[a]);n=int(b.d.moment_rownnz[a]);cols=b.d.moment_colind[adr:adr+n]
        values=b.d.actuator_moment[adr:adr+n]*b.d.actuator_force[a]
        np.add.at(qforce,cols,values)
        if contacts[i]:
            force_per=float(b.d.actuator_force[a])/len(contacts[i])
            for pos,normal,dist in contacts[i]:
                force=force_per*normal;torque=np.cross(pos-com[i],force)
                wrench[i,:3]+=force;wrench[i,3:]+=torque
                contact_rows.append([tick,i,*pos,*force,*normal,dist])
    reconstructed=np.zeros(b.m.nv)
    for i,body in enumerate(b.feet):
        mujoco.mj_applyFT(b.m,b.d,wrench[i,:3],wrench[i,3:],com[i],int(body),reconstructed)
    return qforce,wrench,com,float(np.max(np.abs(reconstructed-qforce))),contact_rows
