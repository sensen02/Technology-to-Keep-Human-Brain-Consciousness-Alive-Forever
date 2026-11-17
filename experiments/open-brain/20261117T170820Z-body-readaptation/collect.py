"""Teacher labels on native states reached with the new fixed cervical dynamics."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np
import torch

from context import HERE,NECK,OLD_WEIGHT,first,ground,overlap,phase,study,hash_file
from tasks import Task,TASKS
from cases import Case as FirstCase


class BodyNeck(study.Neck):
    def __init__(self,task):
        super().__init__(task.body_case())
        self.cut=np.array([root in self.circuit.brain_roots for root in self.roots]) if task.cut_end else self.open_mask()
        self.stop=task.cut_end
        self.first=task.cut_start
        self.mask=np.zeros_like(self.cut)

    def open_mask(self):
        return np.zeros(len(self.roots),dtype=bool)

    def step(self,tick,period,forces):
        saved=self.cut
        self.cut=saved if self.first<=tick<self.stop else self.mask
        result=super().step(tick,period,forces)
        self.cut=saved
        return result


def prepare(task):
    if task.suite=='ground':
        parameters=json.loads((ground.HERE/'REFINE/CALIBRATION.json').read_text()) if task.family.startswith('stride') else None
        neck=ground.Ground(task.ground_case(),task.mode=='cut',parameters)
        steps=ground.TICKS
        transition=lambda tick,forces:neck.step(tick,task.period,forces)
    elif task.suite=='eon':
        meta,events=overlap.brain_record(overlap.HERE/'BRAIN'/f'181_{task.family}')
        parameters=json.loads((overlap.HERE/'CALIBRATION.json').read_text())
        neck=overlap.Overlap(meta['roots'],'open',task.family,parameters)
        steps=len(events)
        transition=lambda tick,forces:neck.step(tick,events[tick],forces)
    else:
        neck=BodyNeck(task)
        steps=study.TICKS
        transition=lambda tick,forces:neck.step(tick,task.period,forces)
    return neck,steps,transition


def evaluate(task,trace,feet,legs):
    if task.suite=='ground':
        return ground.score(task.ground_case(),task.mode,trace,feet,legs)
    if task.suite=='body' and task.cut_end:
        case=FirstCase(task.name,task.seed,task.start,task.stop,cut_start=task.cut_start,cut_end=task.cut_end)
        return first.score(case,trace)
    case=task.body_case()
    if task.suite=='eon':
        family='arc_'+task.family.rsplit('_',1)[1] if task.family.startswith(('a01','a02')) else task.family
        case=study.Case(task.name,family,task.seed,0,task.start,task.stop)
    return study.score(case,trace)


@torch.inference_mode()
def run(task,mode,checkpoint,out):
    torch.set_num_threads(1)
    neck,steps,step=prepare(task)
    body=study.Body()
    body.at_start(task.seed)
    oscillator=phase.Oscillator(body)
    actor=phase.read_policy(checkpoint) if checkpoint else None
    teacher=mode in ('teacher','correct')
    legs=body.fly.get_legs_order()
    from flygym.anatomy import BodySegment
    idx=[body.fly.get_bodysegs_order().index(BodySegment(f'{leg}_tarsus5')) for leg in legs]
    foot_ids=np.asarray(body.sim._internal_bodyids_by_fly[body.name])[idx]
    geom_legs=ground.foot_geometry(body,legs)
    wrench=np.zeros(6)
    roots=list(neck.circuit.port_roots)
    trace=np.empty((steps,15))
    feet=np.empty((steps,6))
    load=np.empty((steps,6))
    emitted=np.empty((steps,len(roots)),dtype=bool)
    arrived=np.empty_like(emitted)
    observations,phases,labels,controls,poses=[],[],[],[],[]
    for tick in range(steps):
        found,forces=body.contact()
        emitted[tick],arrived[tick]=step(tick,np.linalg.norm(forces[body.fronts],axis=1))
        command=neck.command
        if teacher:
            label=body.clip_action(body.teacher(command))
        else:
            oscillator.step(command)
        if tick%first.HOLD==0:
            local=body.observation(command,(found,forces))
            phase_state=oscillator.features()
            estimate=body.clip_action(body.neutral+actor.act(local,phase_state,True)) if actor else None
            control=label if mode=='teacher' else (.8*estimate+.2*label if mode=='correct' else estimate)
            observations.append(local)
            phases.append(phase_state)
            if teacher: labels.append((label-body.neutral).astype(np.float32))
            controls.append(control.copy())
            poses.append(body.d.qpos.copy())
        push_time=task.action+1900 if task.suite=='ground' and task.family.startswith('stride') else 6500
        body.d.xfrc_applied[body.thorax,1]=.12*body.weight*task.push if push_time<=tick<push_time+200 else 0.
        body.apply(control)
        fresh,forces=body.contact()
        rot=body.d.xmat[body.thorax]
        pos=body.d.xpos[body.thorax]
        trace[tick]=[(tick+1)*first.DT,*pos,rot[8],math.atan2(rot[3],rot[0]),*command,body.nonfoot(),*fresh]
        feet[tick]=(body.d.xpos[foot_ids]-pos)@rot.reshape(3,3)[:,0]
        load[tick]=ground.native_loads(body,geom_legs,wrench)
    assert not body.d.warning.number.any() and np.isfinite(trace).all()
    # The ground halt/stride protocol uses native supporting contacts. Other tasks
    # retain their recorded historical contact-cycle gate for direct comparison.
    if task.suite=='ground':
        for i in range(6):
            trace[:,9+i]=np.convolve(load[:,i],np.full(51,1./51),mode='same')>=.02
    result=evaluate(task,trace,feet,legs)
    out.mkdir()
    archive=out/'TRACE.npz'
    np.savez_compressed(archive,trace=trace,feet=feet,normal_load=load,x=np.asarray(observations),
           phase=np.asarray(phases),y=np.asarray(labels),ctrl=np.asarray(controls),qpos=np.asarray(poses),
           emitted=emitted,arrived=arrived)
    result.update(task=asdict(task),mode=mode,trace_sha256=hash_file(archive),ports=roots,
        parameter_sha256=hash_file(NECK/'PARAMETERS.json'),readout_sha256=hash_file(NECK/'READOUT.json'),
        source_sha256=hash_file(HERE/'collect.py'),weight_sha256=hash_file(checkpoint) if checkpoint else None,
        snapshot_sha256=hash_file(first.MODEL_FILE))
    first.write(out/'RESULT.json',result)
    print(task.name,mode,result['pass_'],[k for k,v in result['gates'].items() if not v],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('teacher','correct','evaluate'))
    parser.add_argument('name',choices=TASKS)
    parser.add_argument('out',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    args=parser.parse_args()
    run(TASKS[args.name],args.mode,args.checkpoint,args.out)
