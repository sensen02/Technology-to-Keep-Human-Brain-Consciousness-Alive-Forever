"""Kinematic screening with previously integrated, unperturbed configurations."""
import json
from itertools import product

import mujoco as mj
import numpy as np

from geometry import HERE, LEG, OBSTACLE, SEGMENTS, make_obstacle_model

archive = HERE.parents[1] / '20261116T000536Z/PROTO_nominal/TRACE.npz'
scenario={'center':[.83, -.744, .14],'half_size':[.012,.009,.14]}
m=make_obstacle_model(scenario)
d=mj.MjData(m)
g=m.geom(OBSTACLE).id
ids=[i for i in range(m.ngeom) if int(m.geom_bodyid[i]) > 0 and (m.geom_contype[i] or m.geom_conaffinity[i])]
names=[m.geom(i).name for i in ids]
si=[names.index(name) for name in SEGMENTS]
with np.load(archive,allow_pickle=False) as old:
    stride=5
    qpos=old['qpos'][::stride].copy()
    load=old['loads'][::stride,3]
data=np.empty((len(qpos),len(ids),3))
ranges=np.empty((len(qpos),len(ids),2,3))
mesh={}
for k,i in enumerate(ids):
    mid=m.geom_dataid[i]
    if mid>=0 and m.geom_type[i]==mj.mjtGeom.mjGEOM_MESH:
        mesh[i]=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]].copy()
for t,q in enumerate(qpos):
    d.qpos[:]=q
    mj.mj_kinematics(m,d)
    data[t]=d.geom_xpos[ids]
    for k,i in enumerate(ids):
        if i in mesh:
            xyz=mesh[i] @ d.geom_xmat[i].reshape(3,3).T + d.geom_xpos[i]
            ranges[t,k]=xyz.min(0),xyz.max(0)
        else:
            ranges[t,k]=d.geom_xpos[i]-m.geom_rbound[i],d.geom_xpos[i]+m.geom_rbound[i]
pool=[]
for x,y,h in product(np.arange(.735, .91, .005),np.arange(-.770,-.695,.005),np.arange(.195,.326,.005)):
    center=np.array([x,y,h/2])
    half=np.array([.009,.008,h/2])
    low=center-half
    high=center+half
    inside=np.all(ranges[:,:,0] <= high, axis=2)&np.all(ranges[:,:,1]>=low,axis=2)
    a=np.flatnonzero(inside[:,si].any(1))
    if len(a)==0 or not 2315<=a[0]*stride<=2510 or load[a[0]]>.05*sum(m.body_mass)*9801:
        continue
    early=inside.copy()
    early[430:560,si]=False
    count=early.sum()
    score=(count, int(abs(a[0]*stride-2420)), int(len(a)))
    pool.append((score,center,half))
pool.sort(key=lambda row:row[0])
candidates=[]
for score,center,half in pool[:600]:
    low=center-half
    high=center+half
    fits=np.all(ranges[:,:,0] <= high, axis=2)&np.all(ranges[:,:,1]>=low,axis=2)
    idx,which=np.nonzero(fits)
    hits=[]
    for t in np.unique(idx):
        d.qpos[:]=qpos[t]
        mj.mj_kinematics(m,d)
        d.geom_xpos[g]=center
        m.geom_size[g]=half
        for k in which[idx==t]:
            gap=mj.mj_geomDistance(m,d,ids[k],g,.02,None)
            if gap<0:
                hits.append((int(t*stride),names[k],float(gap)))
    if not hits or hits[0][1] not in SEGMENTS or not 2300 <= hits[0][0] <= 2530:
        continue
    other=[hit for hit in hits if hit[1] not in SEGMENTS]
    later=[hit for hit in hits if hit[0]>3000]
    timing=hits[0][0]
    score=[len(later),len(other),abs(timing-2400),len(hits)]
    candidates.append(dict(score=score,center=center.tolist(),half_size=half.tolist(),first_hit=hits[0],
                           last_hit=hits[-1],other_first=other[:3],
                           hit_names=sorted(set(hit[1] for hit in hits))))
candidates.sort(key=lambda c:c['score'])
(HERE/'SCAN.json').write_text(json.dumps(candidates,indent=2)+'\n')
print('Screen pools',len(pool),len(candidates))
for c in candidates[:18]:
    print(c)
