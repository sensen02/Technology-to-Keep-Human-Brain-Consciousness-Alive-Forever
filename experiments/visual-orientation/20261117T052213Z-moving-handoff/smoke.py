"""Verify checkpoint restoration by matched native tick replay on the same plant."""
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import torch

from handoff_replay import course, PAIRED
from handoff_snapshot import write_checkpoint, read_checkpoint, restore


torch.set_num_threads(1)
course.SCENES=PAIRED/"SCENES.json"
with TemporaryDirectory() as scratch:
    p = Path(scratch)
    brain = course.new_brain()
    env = course.Course("cal_right",p/"base",brain)
    try:
        for i in range(8):
            env.step(1,.6)
        path = p/"state.pkl"
        digest = write_checkpoint(env,path)
        state = read_checkpoint(path,digest)
        length = len(env.neck.link._journal)
        def steps():
            rows=[]
            for mode,pace in ((1,.4),(3,.5),(0,.5)):
                score,_,com,_,_,_,actual,gates = env.step(mode,pace)
                rows.append(np.r_[score["reward"],com,actual.flat,gates.flat,env.body.d.qpos,
                                  env.body.d.qvel,env.body.actuator.target_rad,env.past.flat])
            return rows
        baseline = steps()
        events = env.neck.link._journal[length:]
        restore(env,state,brain)
        np.testing.assert_array_equal(env.body.d.qvel,state["native_data"].qvel)
        replay = steps()
        np.testing.assert_array_equal(baseline,replay)
        assert events==env.neck.link._journal
        print("restore reproduces 4800 native steps and exact neck events",digest,flush=True)
    finally:
        env.close()
