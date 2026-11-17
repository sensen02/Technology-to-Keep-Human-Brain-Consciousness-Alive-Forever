"""Measure actual force contributions of the terrain and original floor near final stance."""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np

from ramp_hfield import HERE, Capture, scene
from ramp_course import LEGS, W, stan, turn

BASE=HERE/'RAMP_OVERLAY_LONG'
TICKS=10


class SplitCapture(Capture):
    def __init__(self,body,role):
        super().__init__(body,TICKS,role)
        self.lower_native=np.zeros_like(self.forces)

    def solved(self,c,before):
        super().solved(c,before)
        self.lower_native[self.count]=self.lower


def run(output):
    origin=turn.read(BASE/'RESULT.json')
    for file in ('TRACE.npz','DECISIONS.npz'):
        assert stan.sha(BASE/file)==origin['files'][file]
    output.mkdir()
    body,role,_,_=scene(origin['plan'],True)
    with np.load(BASE/'TRACE.npz',allow_pickle=False) as data:
        body.d.qpos[:]=data['qpos'][-1]
        body.d.qvel[:]=data['qvel'][-1]
        body.d.ctrl[:]=data['ctrl'][-1]
        body.actuator.target_rad[:]=data['target'][-1]
        body.d.time=float(data['time'][-1])
    with np.load(BASE/'DECISIONS.npz',allow_pickle=False) as decisions:
        action=decisions['action'][-1].copy()
    mj.mj_forward(body.plant,body.d)
    begin=body.d.subtree_com[body.th].copy()
    a=SplitCapture(body,role)
    body.recorder=a
    for _ in range(TICKS):
        body.tick(action)
    assert a.count==TICKS*80 and not a.poses[:a.count,7].any()
    upper=a.forces-a.lower_native
    raw=output/'SOURCE_CONTACTS.npz'
    np.savez_compressed(raw,lower=a.lower_native,upper=upper,poses=a.poses)
    def loads(data):
        return dict(zip(LEGS,(data[:,:6].mean(0)/W).tolist()))

    result=dict(scope='10 ms actual physical integrations started at retained end qpos/qvel/targets and action',
                one_frozen_posture_not_whole_run_source=str(BASE),native_steps=a.count, original_shape_sha=origin['files']['RUN_IDENTITY.json'],
                contact_normalization='positive world vertical force divided by frozen total weight',
                original_approach_plane=loads(a.lower_native),upper_heightfield=loads(upper),
                upper_total_vertical_W=float(upper[:,:6].sum(1).mean()/W),
                lower_total_vertical_W=float(a.lower_native[:,:6].sum(1).mean()/W),
                measured_displacement=(body.d.subtree_com[body.th]-begin).tolist(),
                zero_extra_force=True,source_contact_sha256=stan.sha(raw))
    stan.put(output/'RESULT.json',result)
    print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    run(a.output.resolve())
