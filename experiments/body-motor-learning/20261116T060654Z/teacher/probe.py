from pathlib import Path
import sys
import mujoco as mj
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / '20261116T000536Z'))
from walking_bench import stand

c=stand.physics.Core(adhesion=False)
m,d=c.plant,c.d
q=np.load(HERE.parents[1] / '20261116T000536Z/PROTO_nominal/TRACE.npz')['qpos']
labels = ['RFTibia_geom','RFTarsus1_geom','RFTarsus2_geom','RFTarsus3_geom','RFTarsus4_geom','RFTarsus5_pad0']
for t in range(1300,4001,100):
    d.qpos[:]=q[t]
    mj.mj_forward(m,d)
    values=[]
    for lab in labels:
        g=m.geom(lab).id
        mid=m.geom_dataid[g]
        if mid >= 0 and int(m.geom_type[g])==int(mj.mjtGeom.mjGEOM_MESH):
            v=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]
            xyz=v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
        else:
            xyz=d.geom_xpos[g][None]+m.geom_size[g,0]*np.r_[np.eye(3),-np.eye(3)]
        values.append(np.round([*xyz.min(0),*xyz.max(0)],3).tolist())
    print(t, ' '.join(str(v) for v in values))
