"""Optical marker and pressure-map transducers on the same archived body replay."""
import mujoco
from pathlib import Path
from paths import *

# Four calibrated orthographic views. Channel-tagged fiducials; integer pixel centers.
ANGLES = np.deg2rad([15,105,195,285])
VIEWS = np.array([[[-np.sin(a),np.cos(a),0],
                   [-.5*np.cos(a),-.5*np.sin(a),np.sqrt(.75)]] for a in ANGLES])
P = VIEWS.reshape(8,3)
PIXEL_MM = .012
GRID_MM = .125


def acquire(model, trace, rng):
    d = mujoco.MjData(model)
    # Three exterior thoracic fiducials; tibia base, tarsus base and distal fifth tarsomere.
    ids = [model.body('nmf/'+leg+'_'+part).id for leg in LEGS for part in ('tibia','tarsus1','tarsus5')]
    thorax = model.body('nmf/c_thorax').id
    body_tags = np.array([[0,0,.18],[.4,0,.18],[0,.25,.18]])
    with np.load(trace) as z:
        poses=z['qpos'][49::50]
        load=z['normal_load'].reshape(-1,50,6).mean(axis=1)
    markers = []
    for q in poses:
        d.qpos[:] = q
        mujoco.mj_kinematics(model,d)
        points = d.xpos[ids].copy()
        # Place distal markers at distal end of the visible tarsal mesh.
        for i in range(2,len(ids),3):
            g=model.body_geomadr[ids[i]]
            mid=model.geom_dataid[g]
            v=model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid]+model.mesh_vertnum[mid]]
            positions=v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
            points[i]=positions[np.argmax(np.linalg.norm(positions-d.xpos[ids[i]],axis=1))]
        markers.append(np.vstack([body_tags@d.xmat[thorax].reshape(3,3).T+d.xpos[thorax],points]))
    markers=np.asarray(markers)
    pixels=np.rint(markers@P.T/PIXEL_MM+rng.normal(0,.22,(len(markers),21,8))).astype(np.int16)
    # Externally sampled, unlabeled taxels. Only compressive normal load is represented.
    # One resultant per foot; the acquisition model places it at the marked distal contact.
    pads=np.rint(markers[:,5::3,:2]/GRID_MM).astype(int)
    maps=[]
    for t,foot in enumerate(pads):
        cells,idx=np.unique(foot,axis=0,return_inverse=True)
        force=np.maximum(0,np.bincount(idx,weights=load[t])+rng.normal(0,.10,len(cells)))
        force=np.rint(force/.1)*.1
        maps.append(np.column_stack((cells,force)))
    return pixels,maps


def decode(pixels,maps):
    observed=pixels.astype(float)@np.linalg.pinv(P).T*PIXEL_MM
    forces=np.zeros((len(pixels),6))
    for t,frame in enumerate(maps):
        feet=observed[t,5::3,:2]
        dist=np.linalg.norm(frame[:,None,:2]*GRID_MM-feet[None],axis=2)
        for row, leg in enumerate(dist.argmin(axis=1)):
            if dist[row,leg]<.20 and frame[row,2]>=.5:
                forces[t,leg]+=frame[row,2]
    return observed.astype('f'),forces.astype('f')


if __name__=='__main__':
    for seed in SEEDS:
        folder=ARCHIVE/f'seed{seed}'
        for lane,mode in enumerate(read(folder/'INPUTS.json')['modes']):
            out=HERE/'data'/f'{seed}_{mode}'
            out.mkdir(exist_ok=True,parents=True)
            model=mujoco.MjModel.from_binary_path(str(folder/(mode+'.mjb')))
            pixels,maps=acquire(model,folder/mode/'TRACE.npz',np.random.default_rng(seed+lane*150))
            pos,force=decode(pixels,maps)
            np.savez_compressed(out/'external.npz',pixel_xy=pixels,taxels=np.vstack(maps),
                                offsets=np.r_[0,np.cumsum(list(map(len,maps)))],markers_mm=pos,force_uN=force)
