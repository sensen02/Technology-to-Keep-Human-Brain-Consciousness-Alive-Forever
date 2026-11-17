"""Bounded foot-extension feasibility from the recorded late plateau posture."""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np
import torch

from ramp_hfield import HERE, Capture, scene
from ramp_course import LEGS, W, previous, stan, turn

ORIGIN=HERE/'RAMP_OVERLAY_LONG'
DURATION=1800


class Hold:
    def __init__(self,body,teacher,lift):
        self.body=body
        self.teacher=teacher
        self.baseline=teacher.rest_pad_bottom.copy()
        self.initial=body.d.subtree_com[body.th,2]
        self.lift=lift
        self.last_snapshot={}
        self.correction=0.

    def sample(self,t):
        desired=self.initial+self.lift*previous.blend(t/.65)
        error=float(desired-self.body.d.subtree_com[self.body.th,2])
        self.correction=float(np.clip(2.*error,0.,.044))*previous.blend(t/.25)
        self.teacher.rest_pad_bottom[:]=self.baseline-self.correction
        self.last_snapshot=dict(absolute_goal=float(desired),bounded_foot_extension=self.correction)
        return dict(target_xy=self.teacher.rest[:,:2].copy(),target_z=np.zeros(6),stance=np.ones(6,dtype=bool),
                    source_time_s=None,phase='plateau_body_height_feedback')


def run(output,lift):
    reference=turn.read(ORIGIN/'RESULT.json')
    archive=ORIGIN/'TRACE.npz'
    assert reference['files']['TRACE.npz']==stan.sha(archive)
    output.mkdir()
    plan=reference['plan']
    body,roles,_,_=scene(plan,True)
    with np.load(archive,allow_pickle=False) as records:
        body.d.qpos[:]=records['qpos'][-1]
        body.d.qvel[:]=records['qvel'][-1]
        body.d.ctrl[:]=records['ctrl'][-1]
        body.actuator.target_rad[:]=records['target'][-1]
        body.d.time=float(records['time'][-1])
    mj.mj_forward(body.plant,body.d)
    saved=body.d.qpos.copy()
    record=Capture(body,DURATION,roles)
    record.lo=plan['upper_x']+.005
    record.hi=10.
    body.recorder=record
    teacher=turn.make_teacher(body,'arc_left',.12,.115)
    reference=Hold(body,teacher,lift)
    teacher.reference=reference
    action=np.zeros(42)
    samples=[]
    initial_height=float(body.d.subtree_com[body.th,2])
    ident=stan.identity.capture(body,dict(scope='diagnostic physical hold from archived late plateau qpos qvel motors',
                         height_request_native=lift,terminal_snapshot=True),extra_paths=[Path(__file__),archive,ORIGIN/'RESULT.json'])
    stan.put(output/'RUN_IDENTITY.json',ident)
    for tick in range(DURATION):
        if tick%5==0:
            action,_=teacher.action(body,tick*.001)
        measured=body.tick(action)
        samples.append(np.r_[body.d.subtree_com[body.th],measured['loads']/W,reference.correction])
        if np.any(body.d.xfrc_applied) or np.any(body.d.qfrc_applied):
            raise RuntimeError('External force is forbidden in the diagnostic')
    assert record.count==DURATION*80 and not record.poses[:,7].any()
    np.testing.assert_array_equal(record.endpoints[0],saved)
    tail=record.forces[-16000:]
    field=record.slope_values[-16000:]
    z=np.asarray(samples)
    vertical=(z[-200:,:3]-z[-201:-1,:3])/.001/stan.LENGTH
    speed=float(np.sqrt(np.square(vertical).sum(1).mean()))
    plateau_load=dict(zip(LEGS,(field[:,:6].mean(0)/W).tolist()))
    correction=dict(native_steps=record.count,actual_terminal_com_delta=float(z[-200:,2].mean()-initial_height),
                    requested_height_gain=lift,plateau_per_foot_normal_W=plateau_load,
                    total_leg_vertical_W=dict(zip(LEGS,(tail[:,:6].mean(0)/W).tolist())),
                    end_stationary_speed_bodylength_s=speed,peak_nonfoot_W=float(record.forces[:,12].max()/W),
                    terminal_nonfoot_W=float(tail[:,12].max()/W), all_native_extra_force_zero=True,
                    max_foot_extension=float(z[:,-1].max()),scope='physical probe initialized near final standing with archived q and motor targets')
    np.savez_compressed(output/'PROBE.npz',sampled=z,native_pose=record.poses,native_force=record.forces,upper=record.slope_values)
    correction['array_sha256']=stan.sha(output/'PROBE.npz')
    stan.identity.verify_runtime(ident,body)
    stan.put(output/'RESULT.json',correction)
    print(correction,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--lift',type=float,required=True)
    a=p.parse_args()
    assert 0.<=a.lift<=.03
    torch.set_num_threads(1)
    run(a.output.resolve(),a.lift)
