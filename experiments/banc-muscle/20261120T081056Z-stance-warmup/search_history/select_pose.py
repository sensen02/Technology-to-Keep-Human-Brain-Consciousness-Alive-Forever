"""Choose a non-intersecting six-foot stance; quantify temporary joint trim after native muscle allocation."""
import itertools
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import mujoco as mj
from solve import Equilibrium, HERE, body, load_pose
from balance_root import solve
from scipy.spatial import ConvexHull

LEG_PREFIXES=('LF','LM','LH','RF','RM','RH')
SEGMENTS=('Coxa','Femur','Trochanter','Tibia','Tarsus')


def geometry(eq,d):
    m=eq.m
    geoms=[g for g in range(m.ngeom) if m.geom(g).name[:2] in LEG_PREFIXES
           and any(k in m.geom(g).name for k in SEGMENTS)]
    # Check every pair from different legs, including pairs excluded by the collision mask.
    clearance=min(mj.mj_geomDistance(m,d,i,j,1.,None) for i,j in itertools.combinations(geoms,2)
                  if m.geom(i).name[:2]!=m.geom(j).name[:2])
    local=(d.geom_xpos[eq.foot_geoms]-d.xpos[eq.obs.thorax])@d.xmat[eq.obs.thorax].reshape(3,3)
    min_side=float(min(local[:,1]*eq.signs))
    contacts=[c for c in d.contact if eq.obs.floor in c.geom]
    points=np.array([c.pos[:2] for c in contacts])
    if len(points)<3:
        return dict(admissible=False)
    hull=ConvexHull(points)
    margin=float(-max(hull.equations[:,:2]@d.subtree_com[0,:2]+hull.equations[:,2]))
    loads=eq.obs.measure(d)['ground_load_w']
    depth=min(c.dist for c in contacts)
    legal=bool(np.all(d.qpos[7:]>=m.jnt_range[1:,0]-1e-10) and np.all(d.qpos[7:]<=m.jnt_range[1:,1]+1e-10))
    ok=clearance>=0 and min_side>.10 and margin>.05 and depth>=-.012 and loads[-1]==0 and legal
    ok=bool(ok and min(loads[:6])>.01 and abs(sum(loads[:6])-1)<.001 and d.xmat[eq.obs.thorax,8]>=.95)
    return dict(admissible=ok,leg_mesh_clearance_mm=clearance,side_margin_mm=min_side,
                support_margin_mm=margin,depth_mm=float(depth),legal=legal)


def choose():
    eq=Equilibrium(body(1.))
    reference=np.r_[eq.reference,np.full(eq.m.nu,.25)]
    eligible=[]
    audit=[]
    with TemporaryDirectory() as temp:
        source=Path(temp)/'seed.npz'
        out=Path(temp)/'pose.json'
        for branch in ('mid/pitch','high/pitch','low/escape','widened/stage2'):
            target=np.load(HERE/(branch+'.npz'))['x']
            for k in np.linspace(0.,1.,21):
                x=reference*(1-k)+target*k
                np.savez(source,x=x,contact_ms=1.)
                solve(source,out)
                stance=json.loads(out.read_text())
                d=load_pose(eq.m,stance['qpos'])
                d.act[:]=stance['tonic_activation']
                d.ctrl[:]=d.act
                d.qfrc_applied[6:]=stance['warm_teacher_hinge_trim']
                mj.mj_forward(eq.m,d)
                check=geometry(eq,d)
                check.update(branch=branch,k=float(k),trim_norm=float(np.linalg.norm(d.qfrc_applied[6:])),
                             max_equilibrium_accel=float(max(abs(d.qacc))))
                audit.append(check)
                if check['admissible'] and check['max_equilibrium_accel']<1e-6:
                    eligible.append((check['trim_norm'],stance,check))
    _,stance,check=min(eligible,key=lambda row:row[0])
    (HERE/'diagnostic_trim_pose.json').write_text(json.dumps(stance,indent=2)+'\n')
    (HERE/'POSE_SEARCH.json').write_text(json.dumps(dict(selected=check,candidates=audit),indent=2)+'\n')
    print(check)


if __name__=='__main__':
    choose()
