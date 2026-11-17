"""Paired bounded tactile teacher probe on a solid box with a later leading edge."""
import argparse
import json
from pathlib import Path
import time

import mujoco as mj
import numpy as np
import torch

from controller import Controller, CourseReference, previous
from rollout import (BlockRecord, HERE, STUMBLE_SOURCE, TURN_SOURCE, turn, stan, W, LEGS,
                     build_scene, event, native_metrics, snapshot, INITIAL, LIBRARY)
from physical import Local


def resistance(c, parts, leg):
    force = np.zeros(6)
    h = c.d.xmat[c.th].reshape(3,3)[:,0]
    amount = 0.
    for i in range(c.d.ncon):
        contact = c.d.contact[i]
        ga, gb = int(contact.geom1), int(contact.geom2)
        if contact.efc_address < 0 or c.planes[ga] == c.planes[gb]:
            continue
        sign, fly = (1.,gb) if c.planes[ga] else (-1.,ga)
        if c.geom_leg[fly]!=leg and parts[fly]!=leg:
            continue
        normal = sign*contact.frame[:3]
        if abs(normal[2]) > .65 or normal @ h > -.35:
            continue
        mj.mj_contactForce(c.plant,c.d,i,force)
        amount += max(0.,float(-normal @ h)*force[0]/W)
    return amount


class TactileReference(CourseReference):
    """One bounded corrective swing per phase uses current anterior tactile load."""
    def __init__(self,c,teacher,order,plan,parts,enabled):
        super().__init__(c,teacher,order,plan,False)
        self.c, self.parts, self.enabled = c, parts, enabled
        self.trigger_slot = -1
        self.trigger_t = None
        self.trigger_leg = None
        self.events = []
        self.sensed = 0.
        self.bump = np.zeros(2)

    def sample(self,t):
        r = super().sample(t)
        if t>=self.duration:
            return r
        slot,leg,u = previous.schedule(t,self.order,self.step_s)
        self.sensed = resistance(self.c,self.parts,leg)
        if self.enabled and self.trigger_slot!=slot and .15<u<.94 and self.sensed>=self.plan['reflex_detect_W']:
            self.trigger_slot,self.trigger_t,self.trigger_leg = slot,t,leg
            self.events.append(dict(relative_s=t, leg=LEGS[leg], resistance_W=self.sensed))
        self.bump[:] = 0.
        if self.trigger_t is None:
            return r
        elapsed=t-self.trigger_t
        rise=self.plan['rise_s']
        hold=self.plan['hold_s']
        release=self.plan['release_s']
        height=self.plan['raise_native']*previous.blend(elapsed/rise)*(1.-previous.blend((elapsed-rise-hold)/release))
        retract=self.plan['retract_native']*previous.blend(elapsed/rise)*(1.-previous.blend((elapsed-rise)/release))
        i=self.trigger_leg
        r['target_z'][i] += height
        h=np.r_[self.heading,0.] @ self.rot0
        r['target_xy'][i] -= retract*h[:2]
        self.last_target[i,:2] -= retract*self.heading
        self.last_target[i,2] += height
        self.bump[:] = (height,retract)
        return r


def run(output,plan_path,enabled,finish):
    torch.set_num_threads(1)
    output.mkdir()
    plan = turn.read(plan_path)
    c,roles,start_sha = build_scene(plan)
    record=BlockRecord(c,finish,roles)
    c.recorder=record
    sense=Local(c,plan['course_ms'])
    pinned=stan.identity.capture(c,dict(scope='closed loop local tactile collision teacher probe', enabled=enabled,
        snapshot_sha256=start_sha,finish_ms=finish),extra_paths=[Path(__file__),HERE/'rollout.py',
        HERE/'controller.py',HERE/'terrain_contacts.cpp',LIBRARY,STUMBLE_SOURCE/'physical.py',
        STUMBLE_SOURCE/'geometry.py',TURN_SOURCE/'turning.py',turn.TEACHER,INITIAL,plan_path,
        turn.PARENT/'PROTOTYPES.npz',turn.PARENT/'prototype.py'])
    stan.put(output/'RUN_IDENTITY.json',pinned)
    stock=turn.straight_policy()
    warm=plan['warm_ms']
    measure=c.measure()
    trace=[snapshot(c,measure,record)]
    control=None
    d=[]
    began=time.monotonic()
    for ms in range(finish):
        if ms%5==0:
            x=turn.features(c,measure)
            local=sense.sense(measure)
            if ms<warm:
                a,command=stock(x,0),np.zeros(2)
                touch=0.
                bump=np.zeros(2)
            else:
                if control is None:
                    control=Controller(c,plan,False)
                    control.reference=TactileReference(c,control.teacher,control.reference.order,plan,record.parts,enabled)
                    control.teacher.reference=control.reference
                a,command=control.action(c,(ms-warm)*.001)
                touch=control.reference.sensed
                bump=control.reference.bump.copy()
            assert a.shape==(42,) and np.isfinite(a).all() and abs(a).max() <= 1.
            d.append((ms,x.copy(),local.copy(),command.copy(),a.copy(),touch,bump))
        measure=c.tick(a)
        trace.append(snapshot(c,measure,record))
        if ms%1000==999:
            print(ms+1,trace[-1]['com'],round(time.monotonic()-began,1),flush=True)
    s={k:np.asarray([row[k] for row in trace]) for k in trace[0]}
    decision={k:np.asarray(v) for k,v in zip(
        ('ms','x','local','command','action','swing_resistance_per_W','tactile_plan_z_retract'),
        zip(*d,strict=True),strict=True)}
    assert record.count==finish*80 and not record.poses[:,7].any()
    np.testing.assert_array_equal(record.endpoints,s['qpos'])
    np.savez_compressed(output/'TRACE.npz',**s)
    np.savez_compressed(output/'DECISIONS.npz',**decision)
    np.savez(output/'NATIVE.npz',forces=record.forces,poses=record.poses,block=record.block)
    native=native_metrics(s,record,plan,finish)
    result=dict(plan=plan,actual_ms=finish,native_integrations=record.count,probe=True,complete=False,
                mode='tactile' if enabled else 'blind',native_engineering=native,
                block_passage=event(s,record.block,plan,True,warm+plan['course_ms']),
                triggered=control.reference.events,
                files={f:stan.sha(output/f) for f in ('TRACE.npz','DECISIONS.npz','NATIVE.npz')})
    stan.identity.verify_runtime(pinned,c)
    stan.put(output/'RESULT.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--on',action='store_true')
    p.add_argument('--ms',default=5500,type=int)
    a=p.parse_args()
    run(a.output.resolve(),HERE/'PROVOCATION.json',a.on,a.ms)
