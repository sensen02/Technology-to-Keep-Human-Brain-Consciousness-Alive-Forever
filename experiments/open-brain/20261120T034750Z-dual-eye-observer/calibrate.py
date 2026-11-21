"""Export a rigid marker plate and the static asset's two eye extrinsics."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
root=Path('/run/media/sensen/Data2/cell_wound_prototype')
sys.path.insert(0,str(root/'experiments/open-brain/20261117T184709Z-body-terrain-repair'))
from body_model import Body,mujoco as mj
b=Body()
spec=b.fly.vision_config_path
import yaml
config=yaml.safe_load(spec.read_text())
model=b.sim.world.mjcf_root.copy()
for name,sensor in config['sensors'].items():
    model.body('nmf/'+sensor['parent']).add_camera(name=name,pos=sensor['rel_pos'],
        euler=sensor['orientation'],fovy=config['fovy_per_eye'])
m=model.compile()
d=mj.MjData(m)
mj.mj_forward(m,d)
head=m.geom('nmf/c_head').id
hb=m.geom_bodyid[head]
r=d.geom_xmat[head].reshape(3,3)
p=d.geom_xpos[head].copy()
def local(pos):
    return (pos-p)@r
head_geom=[]
for i in range(m.ngeom):
    body=m.geom_bodyid[i]
    while body and body!=hb:body=m.body_parentid[body]
    if body==hb:head_geom.append(i)
mid=m.geom_dataid[head]
v=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]
# Place the removable beads on distinct dorsal and frontal surfaces of the static head.
directions=np.array([[.75,.65,.85],[.85,-.65,.82],[-.78,.80,.7],[-.78,-.74,.8],
                     [1.,.08,-.38],[.3,.04,1.]])
directions/=np.linalg.norm(directions,axis=1,keepdims=True)
pts=[]
for direction in directions:
    gd=direction@r
    pts.append((v[np.argmax(v@gd)]+gd*.037).tolist())
eyes=[]
for name in ('l_eye_cam','r_eye_cam'):
    c=m.camera(name).id
    eyes.append(dict(name=name[0].upper(),pos_mm=local(d.cam_xpos[c]).tolist(),
                     rot=(r.T@d.cam_xmat[c].reshape(3,3)).tolist()))
omit=[]
for n in config['hidden_segments']:
    g=m.geom('nmf/'+n).id
    omit.append(g)
out=dict(head_geom=int(head),head_children=head_geom,omit_eye_view=omit,tag_radius_mm=.043,
         tags_mm=pts,colors=[[237,31,38],[20,220,36],[32,128,242],[240,205,19],[225,24,224],[22,224,218]],
         eyes=eyes,fov_y_deg=157,image_hw=[512,450],
         head_forward=(r.T@[1.,0.,0.]).tolist(),head_up=(r.T@[0.,0.,1.]).tolist(),
         eye_source=str(spec),eye_source_sha256=hashlib.sha256(spec.read_bytes()).hexdigest())
Path('rig.json').write_text(json.dumps(out,indent=2)+'\n')
