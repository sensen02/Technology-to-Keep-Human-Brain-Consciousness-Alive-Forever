"""Frozen physical actor with replayed neck commands in independent, smooth contact terrains."""
import functools,sys,json,copy,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import mujoco
import torch
OUT=Path(__file__).resolve().parent
BASE=OUT.parent
OLD=BASE/'20261120T155833Z-body-surrogate'
sys.path.insert(0,str(BASE/'20261117T202200Z-body-stop-repair'))
from policy_repair import read as read_actor
from shared import Body,advance,phase_state,setup,feature,observation
from proprioception import extra_live
import body_model
# The observation module has its own dependency named paths.
sys.path.insert(0,str(OLD))
from observe import acquire,decode
from fast_load import FootLoad
from flygym.compose.world import FlatGroundWorld
PHYSICAL=BASE/'20261117T211320Z-body-command-calibration/FIT/epoch040.pt'
def plan(j):
    if j<3000:
        mode=('intact','block_an','block_dn','dropout_an','dropout_dn','tear_an')[(j//2)%6]
        return [10901,11311][j%2],mode,'train'
    return [(11701,'intact'),(11903,'clean_on')][j%2]+('test',)
def terrain(j):
    rng=np.random.default_rng(20261+j)
    x=np.arange(-6,24.01,.25);y=np.arange(-20,20.01,.25)
    xx,yy=np.meshgrid(x,y)
    phase=rng.uniform(0,2*np.pi,3)
    period=rng.uniform(1.8,4.7,3)
    # A flat landing zone permits reuse of the validated physical initial state.
    ramp=np.clip((np.hypot(xx,yy)-2)/1.5,0,1)
    scale=[.035,.055,.045,.065][(j//2)%4]
    height=scale*ramp**2*(3-2*ramp)*(.5+.29*np.sin(2*np.pi*(xx+.25*yy)/period[0]+phase[0])+.15*np.cos(2*np.pi*(yy-.32*xx)/period[1]+phase[1])+.06*np.sin(2*np.pi*(xx-yy)/period[2]+phase[2]))
    return x,y,height,scale
class Ground(FlatGroundWorld):
    def __init__(self,j):
        super().__init__()
        x,y,height,scale=terrain(j)
        self.mjcf_root.add_hfield(name='terrain_relief',size=(np.ptp(x)/2,np.ptp(y)/2,scale,.1),nrow=len(y),ncol=len(x),userdata=(height/scale).ravel())
        g=self.ground_geom
        g.type=mujoco.mjtGeom.mjGEOM_HFIELD;g.hfieldname='terrain_relief'
        g.pos=((x[0]+x[-1])/2,(y[0]+y[-1])/2,0)
def archive(seed):
    if seed==11903:return Path('/home/sensen/.codex/worktrees/4528/cell_wound_prototype/experiments/open-brain/20261118T052426Z-event-anchored-repair/FORMAL/seed11903')
    return BASE/'20261118T051014Z-neck-fault-assay/FORMAL'/f'seed{seed}'
@functools.cache
def commands(seed,mode):
    with np.load(archive(seed)/mode/'TRACE.npz') as z:
        return z['command']

@functools.cache
def physical_actor():
    return read_actor(PHYSICAL)

def run(j):
    dest=OUT/'episodes'/f'terrain{j:02d}'
    if (dest/'receipt.json').is_file():return
    dest.mkdir(parents=True,exist_ok=True)
    seed,mode,part=plan(j)
    cmd=commands(seed,mode)
    assert cmd.shape==(20000,2)
    x,y,h,scale=terrain(j)
    with patch.object(body_model,'Terrain',lambda _:Ground(j)):
        body=Body(seed=seed)
    body.m.hfield_data[:]=(h/scale).ravel()
    body.restore(seed);setup(body,'terminal')
    foot_load=FootLoad(body)
    actor=physical_actor()
    cpg=copy.deepcopy(body.motor.cpg_network);freq=cpg.intrinsic_freqs.copy()
    poses=np.empty((len(cmd),body.m.nq));loads=np.empty((len(cmd),6))
    centers=np.empty((len(cmd),3));up=np.empty(len(cmd))
    for t,c in enumerate(cmd):
        advance(cpg,freq,c)
        if not t%10:
            ph=phase_state(cpg)
            obs=observation(body)
            state=np.r_[feature(body.d.qpos,body.d.qvel,obs.stumbling_contact_forces,c,ph,body.weight),extra_live(body)].astype('f')
            targets,pads,_=actor.act(state,ph)
        body.apply(targets,pads)
        poses[t]=body.d.qpos;foot_load.record(loads[t])
        centers[t]=body.d.xpos[body.thorax];up[t]=body.d.xmat[body.thorax,8]
    assert np.isfinite(poses).all() and not body.d.warning.number.any()
    with tempfile.TemporaryDirectory(dir=dest) as tmp:
        path=Path(tmp)/'trace.npz'
        np.savez(path,qpos=poses,normal_load=loads)
        pixels,maps=acquire(body.m,path,np.random.default_rng(5021+j))
    markers,force=decode(pixels,maps)
    assert np.isfinite(markers).all() and len(markers)==400
    np.savez_compressed(dest/'external.npz',markers_mm=markers,force_uN=force,pixels=pixels,taxels=np.vstack(maps),offsets=np.r_[0,np.cumsum(list(map(len,maps)))])
    np.savez_compressed(dest/'terrain.npz',x=x,y=y,z=h)
    foot=markers[:,5::3,:2].reshape(-1,2)
    assert np.all((foot[:,0]>x[0])&(foot[:,0]<x[-1])&(foot[:,1]>y[0])&(foot[:,1]<y[-1]))
    ix=np.rint((foot[:,0]-x[0])/.25).astype(int);iy=np.rint((foot[:,1]-y[0])/.25).astype(int)
    supported=force.ravel()>0
    row=dict(terrain=j,seed=seed,mode=mode,split=part,steps=len(cmd),realized_relief_um=float(1000*h.max()),supported_foot_max_surface_um=float(1000*h[iy[supported],ix[supported]].max()),travel_mm=float(np.linalg.norm(centers[-1,:2]-centers[0,:2])),tail_rms_mm_s=float(np.sqrt((np.diff(centers[-2001:,:2],axis=0)**2).sum(1).mean())/.0001),min_upright=float(up.min()))
    temp=dest/'receipt.tmp'
    temp.write_text(json.dumps(row,indent=2))
    temp.replace(dest/'receipt.json')
    print(row,flush=True)
if __name__=='__main__':
    torch.set_num_threads(1)
    with torch.inference_mode():
        for arg in sys.argv[1:]:run(int(arg))
