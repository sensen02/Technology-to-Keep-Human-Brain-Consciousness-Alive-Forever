"""Repeat two declared conditions and inspect actual obstacle contact locations."""
from collections import defaultdict
import argparse
from context import *
import reach

parser=argparse.ArgumentParser();parser.add_argument('condition',choices=('flat','block'));args=parser.parse_args()
if args.condition=='flat':
    reach.run('flat','teacher',119,HERE/'REPLAY/flat_teacher_119')
else:
    original=reach.contacts
    peaks=defaultdict(float);integrals=defaultdict(float)
    last_time={};after_times=set()
    def observed(body):
        out=original(body)
        # contacts() is called before/after each step: only count a new time once.
        if body.d.time not in after_times:
            after_times.add(body.d.time);wrench=np.zeros(6)
            for i,c in enumerate(body.d.contact[:body.d.ncon]):
                if c.geom1 in body.block_ids:other=int(c.geom2)
                elif c.geom2 in body.block_ids:other=int(c.geom1)
                else:continue
                mujoco.mj_contactForce(body.m,body.d,i,wrench)
                name=mujoco.mj_id2name(body.m,mujoco.mjtObj.mjOBJ_BODY,int(body.m.geom_bodyid[other]))
                value=float(np.linalg.norm(wrench[:3]));peaks[name]=max(peaks[name],value)
                integrals[name]+=value*body_run.DT
        return out
    reach.contacts=observed
    reach.run('block015','oracle015',119,HERE/'REPLAY/block015_oracle015_119')
    write(HERE/'OBSTACLE_CONTACTS.json',dict(peak_force_native=dict(peaks),impulse_native_s=dict(integrals),
         sampled_times=len(after_times),teacher_stumble_detection_links=['tibia','tarsus1','tarsus2'],
         note='Additional read-only contact observer; exact physical replay checked separately.'))
