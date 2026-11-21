"""Generate two physical trials with identical saved motor drives. Keep reference private."""
import argparse
import json
from pathlib import Path
from floor import *
from scipy.spatial.transform import Rotation

def run(kind,folder):
    f=Floor(kind)
    m,d,b=f.m,f.d,f.fly
    folder.mkdir(exist_ok=True)
    ids=np.array([m.geom(b.m.geom(int(b.m.body_geomadr[foot])).name).id for foot in b.feet])
    # Renderable exterior surfaces only. Ankle/foot observations are not solver contact labels.
    gids=[m.geom(b.m.geom(i).name).id for i in range(1,b.m.ngeom) if b.m.geom(i).name]
    groups={m.geom(b.m.geom(i).name).id:k for i,k in b.geom_leg.items()}
    leg_load=[]; poses=[]; surface_z=[]; positions=[]; ground_total=[]
    vertex_ids=[]; all_verts=[]; all_faces=[]
    for gid in gids:
        mid=m.geom_dataid[gid]
        if m.geom_type[gid]!=mj.mjtGeom.mjGEOM_MESH:
            continue
        start=m.mesh_vertadr[mid]; count=m.mesh_vertnum[mid]
        vertices=m.mesh_vert[start:start+count].copy()
        all_verts.append(vertices)
        vertex_ids.append(gid)
        fa=m.mesh_faceadr[mid]; fc=m.mesh_facenum[mid]
        all_faces.append(m.mesh_face[fa:fa+fc].copy())
    np.savez_compressed(folder/'render_mesh.npz',gids=vertex_ids,vertices=np.vstack(all_verts),
                        vcounts=list(map(len,all_verts)),faces=np.vstack(all_faces),
                        fcounts=list(map(len,all_faces)),foot_ids=ids)
    w=np.zeros(6); sub=[]; thrust=[]
    with np.load(TRACK) as src:
        motor,pad=src['control'],src['pads']
    for tick in range(6300):
        d.ctrl[b.actuators]=motor[tick]
        d.ctrl[b.adhesion_ids]=pad[tick]
        f.step()
        # Independent scoring stream, includes opposing native adhesive tractions on the tiles.
        force=np.zeros(7)
        for n,c in enumerate(d.contact[:d.ncon]):
            if c.efc_address<0: continue
            if c.geom1 in f.grid: other,sg=c.geom2,1
            elif c.geom2 in f.grid: other,sg=c.geom1,-1
            else: continue
            mj.mj_contactForce(m,d,n,w)
            force[groups.get(other,6)]+=sg*(c.frame.reshape(3,3).T@w[:3])[2]
        if f.moving:
            for leg,act in enumerate(b.adhesion_ids):
                a=d.moment_rowadr[act]; n=d.moment_rownnz[act]
                cols=d.moment_colind[a:a+n]
                tile=cols>=f.v[0]
                force[leg]-=d.actuator_force[act]*d.actuator_moment[a:a+n][tile].sum()
            thrust.append(-np.sum(d.qfrc_constraint[f.v]+d.qfrc_actuator[f.v]))
        else:
            for leg,act in enumerate(b.adhesion_ids):
                start=d.moment_rowadr[act]; end=start+d.moment_rownnz[act]
                cols=d.moment_colind[start:end]
                force[leg]+=d.actuator_force[act]*d.actuator_moment[start:end][cols==2].sum()
        sub.append(force)
        if (tick+1)%20: continue
        leg_load.append(np.mean(sub,axis=0)); sub=[]
        if f.moving:
            surface_z.append(f.depth().copy())
            ground_total.append(np.mean(thrust));thrust=[]
        positions.append(d.xpos[b.thorax].copy())
        poses.append(np.column_stack((d.geom_xpos[gids],d.geom_xmat[gids])))
    np.savez_compressed(folder/'private_walk.npz',forces=leg_load,total=ground_total,body=positions)
    np.savez_compressed(folder/'optical_scene.npz',transforms=poses,gids=gids,time_ms=2*np.arange(1,len(poses)+1))
    if surface_z: np.savez_compressed(folder/'tile_motion.npz',depth_um=surface_z)
    print(kind,positions[0],positions[-1],np.percentile(np.sum(leg_load,axis=1),[0,50,100]),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('kind',choices=['sand','insert'])
    p.add_argument('output',type=Path)
    a=p.parse_args()
    run(a.kind,a.output)
