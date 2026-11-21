"""Add a read-only, unlabelled taxel transducer to the frozen physical food teacher."""
from pathlib import Path
import fcntl, sys, json, time
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import mujoco
OUT=Path(__file__).resolve().parent
WORK=Path('/home/sensen/.codex/worktrees/4528/cell_wound_prototype/experiments/open-brain')
ODOR=WORK/'20261119T190005Z-odor-sac'
sys.path.insert(0,str(ODOR))
import live
seed=int(sys.argv[1])
dest=OUT/f'aligned-{seed}'
lease=(OUT/'logs'/f'{seed}.lock').open('w')
fcntl.flock(lease,fcntl.LOCK_EX)
if (dest/'RESULT.json').is_file() and (dest/'floor.npz').is_file():
    raise SystemExit(0)
if dest.exists():
    dest.rename(OUT/'logs'/f'partial-{seed}-{time.time_ns()}')

from cohort import scenario
fields=scenario(seed)
original_course=live.course
def varied_course(seed,stage,side):
    config=original_course(seed,stage,side)
    return replace(config,goal_x=fields.goal_x,goal_y=fields.goal_y,strength=fields.strength)
live.course=varied_course

base_pose=live.docking.start_pose
def angled_pose(env):
    q=base_pose(env)
    angle=np.deg2rad(fields.yaw_deg)/2
    q[3:7]=(np.cos(angle),0.,0.,np.sin(angle))
    with env.physics.reset_context():env.physics.data.qpos[:]=q
    return q
live.docking.start_pose=angled_pose

class Sensor(live.WallContacts):
    def __init__(self,phy):
        super().__init__(phy)
        m=phy.model.ptr
        # Capture contacts *before* dm_control's final mj_step1 rebuilds their indices.
        self.data_addr=phy.data.ptr._address
        self.origin=phy.time()
        self.last=self.origin
        self.bucket=0;self.duration=0.0
        original=mujoco.mj_step1
        def observe_step1(model,data):
            if data._address==self.data_addr and data.time>self.last+1e-10:
                self.after_solve(model,data)
            return original(model,data)
        mujoco.mj_step1=observe_step1
        mujoco.mj_saveModel(m,str(dest/'scene.mjb'),None)
        self.floor=m.geom('groundplane').id
        self.count=0;self.cells={};self.rows=[];self.offset=[0]
        self.weight_native=float(m.body_mass.sum()*abs(m.opt.gravity[2]))
        self.active_frame_steps=25 # 5 ms, with four native physics steps per query
        self.w=np.zeros(6)
        self.rng=np.random.default_rng(seed+974)
        global recorder
        recorder=self
    def after_solve(self,m,d):
        # No leg IDs in the sensor. Use only forces actually delivered to the ground.
        elapsed=float(d.time-self.last);self.last=float(d.time);self.duration+=elapsed
        for cidx,c in enumerate(d.contact[:d.ncon]):
            if self.floor not in (c.geom1,c.geom2) or c.efc_address<0:continue
            mujoco.mj_contactForce(m,d,cidx,self.w)
            # Contact frame x points from geom1 to geom2, reported wrench acts on geom2.
            sign=1 if c.geom1==self.floor else -1
            vertical=max(0.,sign*(c.frame.reshape(3,3).T@self.w[:3])[2])
            xy=tuple(np.rint(10*c.pos[:2]/.125).astype(int))
            self.cells[xy]=self.cells.get(xy,0.)+vertical*elapsed
        self.count+=1
        tick=int(round((d.time-self.origin)/.005))
        if tick>self.bucket and abs(d.time-self.origin-tick*.005)<1e-8:
            assert abs(self.duration-.005)<1e-8
            self.bucket=tick
            for (x,y),v in self.cells.items():
                # CGS native force: 1 dyne = 10 microNewtons.
                value=round(max(0.,10*v/self.duration+self.rng.normal(0,.10)),1)
                if value>=.5:self.rows.append([x,y,value])
            self.offset.append(len(self.rows));self.cells={}; self.duration=0.

# Head RGB was diagnostic in the odor-only teacher. The separate external
# marker cameras are generated from the complete physical trace in prepare.py.
live.docking.observation=lambda *args: (np.zeros(1),np.zeros((1,1,3),dtype=np.uint8),None)
live.WallContacts=Sensor
live.run(SimpleNamespace(out=dest,policy=ODOR/'TRAIN_DETOUR/POLICY.npz',seed=seed,stage='open',side=(1 if fields.goal_y>0 else -1),sense='odor',duration=6.4,assisted=True))
np.savez_compressed(dest/'floor.npz',cells=np.asarray(recorder.rows).reshape(-1,3),offsets=recorder.offset,pitch_mm=.125,threshold_uN=.5,weight_native=recorder.weight_native)
