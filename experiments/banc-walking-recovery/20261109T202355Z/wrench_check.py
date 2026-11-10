from pathlib import Path
import json
import numpy as np
import mujoco as mj
from core_v2 import Core
from frame_audit import rc,T,E
HERE=Path(__file__).resolve().parent

def main():
    c=Core();c.reset();m,d=c.plant,c.d
    c.tick(np.zeros(rc.NJ))
    # Construct identical arbitrary vertical forces at actual contact points using both APIs.
    expected=np.zeros(m.nv);forces=np.zeros_like(d.xfrc_applied)
    for ci in range(d.ncon):
        ct=d.contact[ci];ga,gb=int(ct.geom1),int(ct.geom2)
        if not(c.planes[ga] or c.planes[gb]):continue
        g=gb if c.planes[ga] else ga
        if c.geom_leg[g]<0:continue
        b=int(m.geom_bodyid[g]);f=np.array([0.,0.,-.123])
        mj.mj_applyFT(m,d,f,np.zeros(3),ct.pos,b,expected)
        forces[b,:3]+=f;forces[b,3:]+=np.cross(ct.pos-d.xipos[b],f)
    actual=np.zeros(m.nv)
    for b in range(1,m.nbody):
        mj.mj_applyFT(m,d,forces[b,:3],forces[b,3:],d.xipos[b],b,actual)
    err=float(np.max(np.abs(actual-expected)));assert err<1e-10
    out=dict(label='COM-referenced adhesion fix, exact generalized-force check',wrench_error=err,teacher_v2={})
    print('wrench generalized-force error',err,flush=True)
    # E.score constructs its own Core; replace module dependency inside isolated diagnostic only.
    original=E.rc.Core;E.rc.Core=Core
    try:
        for l in range(rc.NL):
            r=E.score(l,T.teacher_action);out['teacher_v2'][rc.LEG_ORDER[l]]=r
            print(rc.LEG_ORDER[l],r['passed_implemented'],r['advance_implemented'],r['support_min'],flush=True)
    finally:E.rc.Core=original
    (HERE/'WRENCH_CHECK.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
