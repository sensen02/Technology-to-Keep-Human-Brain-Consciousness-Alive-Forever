"""P1 embodied sensory-to-Eon-to-BANC foraging and actual intake."""
import argparse
import copy
import json
from pathlib import Path
import sys

import mujoco as mj
import numpy as np
import torch

from motor import DT, BODY_WEIGHT, MotorAdapter, Neck, NeuralInputs
from fastbrain import FastBrain, eon, load_connectome
from neural_runtime import read, sha, sources, verify, write
from sensors import PRIOR, Arena, estimate, vision_features
from preflight import (BODY_DIR, FRUIT, FRUIT_RADIUS, antennae, build_scene, place,
                       advance_mouth, mouth_xyz, FeedingBody, FeedingState, Food)
from transport_probe import advance, extra_live, feature, load_actor, observation, phase_state, setup


SCENES = {
    'dev': (137, (4.3, .55), 0.),
    'left': (1513, (4.8, .9), -.03),
    'right': (2579, (5.1, -1.1), .025),
}
MODES = ('intact','cut_dn','vision_only','odor_only','blank')
EYES = 200
ANTENNA = 50
TASTE = 50
BODY_CLOCK = 100
STN_DELAY = 15
MN9_WINDOW = 700


class MealReadout:
    """Reuse the passed StN engineering branch and the two MN9 heads."""
    def __init__(self, index, weights, inputs, ticks):
        frozen = read(PRIOR.parent/'20261118T203315Z-active-meal-stop/FIT_TAGGED.json')
        features = [index[int(root)] for root in frozen['feature_roots']]
        projection = abs(weights[features][:,inputs.stn]).toarray().astype(np.float64)
        self.synaptic = projection.sum(axis=0) / projection.sum()
        self.cfg = frozen['controller']
        self.range = frozen['stretch_recruitment']
        self.mn9 = np.zeros((ticks,), dtype=np.int64)
        self.sources = np.zeros((ticks,len(inputs.stn)),dtype=bool)
        self.terminals = np.zeros_like(self.sources)
        self.strength = 0.
        self.alpha = -np.expm1(-DT/self.cfg['filter_s'])
        self.halt = False

    def rate(self,stretch):
        low,high = self.range
        return 80.*np.clip((stretch-low)/(high-low),0.,1.)

    def record(self,tick,brain,inputs,raw_source):
        self.sources[tick] = raw_source
        self.terminals[tick] = brain.spikes[inputs.stn]
        time_offset = brain.queue.shape[0]+1
        if tick >= time_offset:
            impulse = self.terminals[tick-time_offset] @ self.synaptic
            self.strength += self.alpha*(impulse/(80.*DT)-self.strength)
        self.mn9[tick] = int(brain.spikes[inputs.mn9].sum())
        if (tick+1) % BODY_CLOCK == 0:
            self.halt = (self.strength > self.cfg['restart'] if self.halt
                         else self.strength >= self.cfg['stop'])

    def command(self,tick):
        return self.mn9[max(0,tick+1-MN9_WINDOW):tick+1].sum() >= 2 and not self.halt


@torch.inference_mode()
def run(scene, mode, out, duration):
    torch.set_num_threads(1)
    home=Path(__file__).resolve().parent
    assert read(home/'EON_PARITY.json')['status'] == 'BIT_EXACT_STOCK_EON_PARITY'
    sensor_file=home/'SENSOR_FIT/FIT.json'
    motor_file=home/'TRANSLATED_MOTOR.json'
    sensor_fit=read(sensor_file)
    adapter=MotorAdapter(read(motor_file))
    seed, xy, angle = SCENES[scene]
    fruit=np.array([*xy,FRUIT[2]])
    arena=Arena(fruit)
    body=build_scene(None)
    fruit_geom=body.m.geom('fruit').id
    body.m.geom_pos[fruit_geom]=fruit
    body.m.geom_group[fruit_geom] = 3 if mode in ('odor_only','blank') else 0
    with np.load(BODY_DIR/'CANONICAL_INIT.npz',allow_pickle=False) as src:
        snapshot={k:src[k] for k in ('qpos','qvel','ctrl_joint')}
    place(body,snapshot,(0.,0.),angle)
    setup(body,'terminal')
    body.motor.reset(seed=seed)
    act=load_actor(BODY_WEIGHT)
    cpg=copy.deepcopy(body.motor.cpg_network)
    freqs=cpg.intrinsic_freqs.copy()
    oral_geom, oral_point=advance_mouth(body.m,body.d)
    index,_=eon.get_hash_tables(eon.path_comp)
    weights=load_connectome(index)
    sourceset=NeuralInputs(index)
    brain=FastBrain(weights.T.tocsr(),sourceset.an_ix)
    neck=Neck(index,sourceset,tactile=True)
    steps=int(round(duration/DT))
    draws=sourceset.masks(steps,seed)
    stn_draws=np.random.default_rng(seed+12).random((steps,len(sourceset.stn)),dtype=np.float32)
    meal=MealReadout(index,weights,sourceset,steps)
    feeder=FeedingBody()
    body_state=FeedingState(Food(.85,.85))
    front=np.zeros(2)
    command=np.zeros(2)
    pfl=(0.,0.)
    scent=np.zeros(2)
    view=np.zeros(18)
    seen=False
    contact=False
    sugar=0.
    accumulated=0.
    contact_distance=np.inf
    trace=np.zeros((steps//BODY_CLOCK, body.m.nq+22))
    emitted_dn=np.zeros((steps,17),dtype=bool)
    arrived_dn=np.zeros_like(emitted_dn)
    arrived_an=np.zeros((steps,2),dtype=bool)
    python_inputs=[home/name for name in ('live.py','motor.py','fastbrain.py','sensors.py')]
    pins=sources([*python_inputs, home/'EON_PARITY.json',sensor_file,motor_file,BODY_WEIGHT,
                  eon.path_comp,eon.path_con,BODY_DIR/'CANONICAL_INIT.npz',
                  PRIOR/'preflight.py',PRIOR/'fruit_field.py',
                  PRIOR.parent/'20261118T203315Z-active-meal-stop/FIT_TAGGED.json'])
    input_events=np.zeros((steps,5),dtype=np.uint16)
    eye_record=[]
    mouth_flags=[]
    range_inverse=bearing=0.
    warning=False
    for tick in range(steps):
        if tick % ANTENNA == 0:
            mj.mj_kinematics(body.m,body.d)
            scent=arena.smell(antennae(body)) if mode not in ('blank','vision_only') else np.zeros(2)
            contact_distance=float(np.linalg.norm(mouth_xyz(body.d,oral_geom,oral_point)-fruit))
            contact=contact_distance <= FRUIT_RADIUS
            sugar=200. if contact else 0.
        if tick % EYES == 0:
            mj.mj_camlight(body.m,body.d)
            if mode in ('blank','odor_only'):
                view,seen=np.zeros(18),False
            else:
                rgb=body.sim.get_raw_vision(body.name)
                view,seen=vision_features(rgb)
                if tick%(EYES*10)==0:
                    eye_record.append(rgb[:,::3,::3])
            bearing,range_inverse,detect=estimate(sensor_fit,view,seen,scent)
            if contact or meal.halt:
                pfl=(0.,0.)
            elif detect:
                distance=1./max(range_inverse,.1)
                speed=np.clip(.36*(distance-.63),.20,.64)*max(.12,np.cos(bearing))**2
                turn=np.clip(1.12*bearing,-.52,.52)
                pfl=adapter.rates(speed,turn,0.)
            else:
                pfl=(0.,0.)
        if contact:
            pfl=(0.,0.)
        if tick % ANTENNA == 0 or tick % EYES == 0:
            # The translated odor estimate already participates in the head
            # policy. Native population drive is an independent failed arm.
            p=sourceset.rates(pfl,(0.,0.),sugar,0.)*DT
        source=stn_draws[tick] < meal.rate(body_state.crop.volume)*DT
        delivery=meal.sources[tick-STN_DELAY] if tick>=STN_DELAY else np.zeros(len(source),dtype=bool)
        mask=draws[tick]<p
        mask[-len(delivery):]=delivery
        dn=brain.spikes[neck.dn_ix].copy()
        command=neck.update(tick,dn,front,blocked=mode=='cut_dn')
        brain.step(sourceset.assembled[mask],neck.an_inputs)
        meal.record(tick,brain,sourceset,source)
        emitted_dn[tick]=dn
        arrived_dn[tick]=neck.last_dn
        arrived_an[tick]=neck.last_an
        sizes=[sum(map(len,sourceset.pfl)),sum(map(len,sourceset.odor)),len(sourceset.sugar)]
        a,b,c=np.cumsum(sizes)
        input_events[tick]=(mask[:a].sum(),mask[a:b].sum(),mask[b:c].sum(),delivery.sum(),meal.mn9[tick])
        advance(cpg,freqs,command)
        if tick%10==0:
            obs=observation(body)
            phase=phase_state(cpg)
            x=np.r_[feature(body.d.qpos.copy(), body.d.qvel.copy(),obs.stumbling_contact_forces,
                            command,phase,body.weight),extra_live(body)].astype(np.float32)
            control,pads,_=act.act(x,phase)
        body.apply(control,pads)
        if tick%5==0:
            force,_,_,_=body.contact()
            front=force[[0,3]]
        if (tick+1)%BODY_CLOCK==0:
            receipt=feeder.advance(body_state,DT*BODY_CLOCK,
                                   feed=bool(meal.command(tick)),contact=contact)
            body_state=receipt.state
            accumulated+=receipt.ingested.volume
            pos=body.d.xpos[body.thorax]
            orientation=body.d.xmat[body.thorax].reshape(3,3)
            row=(tick//BODY_CLOCK)
            trace[row]=np.r_[body.d.qpos, (tick+1)*DT, pos, np.arctan2(orientation[1,0],orientation[0,0]),
               orientation[2,2], contact_distance, receipt.ingested.volume, accumulated,
               body_state.crop.volume, body_state.reserve, meal.strength, bool(meal.command(tick)),
               contact, seen, scent, bearing, range_inverse, command,front.max()]
            mouth_flags.append(contact)
        if body.d.warning.number.any() or not np.isfinite(body.d.qpos).all():
            warning=True
            break
        if (tick+1)%10000==0:
            print(scene,mode,tick+1,np.round(body.d.xpos[body.thorax],3).tolist(),
                  'dist',round(contact_distance,3),'eat',round(accumulated,4),'route',list(pfl),flush=True)
    if body.sim.eye_renderer is not None:
        body.sim.eye_renderer.close()
    out.mkdir()
    verify(pins)
    trace=trace[:(tick+1)//BODY_CLOCK]
    if mode=='cut_dn':
        assert not arrived_dn.any()
    else:
        n_live=tick+1
        for j,col in enumerate(neck.columns):
            delay=neck.model.neck_delay[col]
            assert not arrived_dn[:delay,j].any()
            assert np.array_equal(arrived_dn[delay:n_live,j],emitted_dn[:n_live-delay,j])
    mj.mj_saveModel(body.m,str(out/'BODY.mjb'),None)
    np.savez_compressed(out/'TRACE.npz',trace=trace, emitted=emitted_dn[:tick+1],
                       arrived=arrived_dn[:tick+1], ascending=arrived_an[:tick+1],
                       input_events=input_events[:tick+1], crop_source=meal.sources[:tick+1],
                       eyes=np.asarray(eye_record))
    n=body.m.nq
    ate=np.flatnonzero(trace[:,n+7]>0.)
    first=float(trace[ate[0],n]) if len(ate) else None
    condition=dict(
        scene=scene, mode=mode, seed=seed, duration_s=steps*DT, fruit=fruit.tolist(),
        first_intake_s=first, ingested=accumulated, min_oral_distance=float(np.min(trace[:,n+6])),
        stop_driver_s=float(np.sum(trace[:,n+11]>=meal.cfg['stop'])*DT*BODY_CLOCK),
        last_xy=trace[-1,n+1:n+3].tolist(), min_upright=float(np.min(trace[:,n+5])),
        eon_source=int(emitted_dn.sum()), actual_neck_arrival=int(arrived_dn.sum()), native_body_steps=tick+1,
        ascending_arrival=int(arrived_an.sum()), warning=warning, qpos_columns=n,
        dn_roots=neck.root_names, original_orn_activation_hz=0.,
        head_olfaction_uses_translation=mode in ('intact','odor_only','cut_dn'),
        source_sha256=pins,
        navigation_accepted=bool(accumulated>=.04 and not warning and np.min(trace[:,n+5])>=.8),
        trace_sha256=sha(out/'TRACE.npz'))
    write(out/'RESULT.json',condition)
    print(json.dumps(condition,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('scene',choices=SCENES)
    p.add_argument('mode',choices=MODES)
    p.add_argument('output',type=Path)
    p.add_argument('--duration',type=float,default=4.2)
    cfg=p.parse_args()
    run(cfg.scene,cfg.mode,cfg.output,cfg.duration)
