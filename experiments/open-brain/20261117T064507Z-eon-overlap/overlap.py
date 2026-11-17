"""Run the shared DN interface through BANC, reading only actual VNC target voltages."""
from dataclasses import asdict
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

from brain_events import BIDIR, HERE, NAMES, OFF, ON, STEER, TICKS, digest, write

BODY_ARCHIVE = HERE.parent/'20261117T062447Z-body-commands'
sys.path.insert(0,str(BODY_ARCHIVE/'PHASE'))
from phase import Oscillator, read_policy, study
import body_run as first
from cases import Case as FirstCase
from physical import load_receptors

DT = first.DT
HOLD = first.HOLD
WEIGHT = BODY_ARCHIVE/'PHASE/FIT/epoch015.pt'
RECEIVING = ('forward_speed','walking_steer_context','walking_steer')
SIDE = ('left','right')
TAU = .060


def read(path):
    with path.open() as stream:
        return json.load(stream)


def brain_record(directory):
    input_path = directory/'INPUT.json'
    meta = read(input_path)
    result = read(directory/'RESULT.json')
    assert digest(input_path)==result['input_sha256']
    trace = directory/'BRAIN.npz'
    assert digest(trace)==result['trace_sha256']
    with np.load(trace,allow_pickle=False) as arrays:
        return meta, arrays['emitted']


class Overlap:
    """A decoder separated from the brain source and conditioned only on neck receiver state."""
    def __init__(self, order, cut_mode, name, calibration=None):
        catalog = read(BIDIR/'CATALOG.json')['ports']
        self.ports = {root:port for root,port in catalog.items()
                      if root in order or port['group'] in ('front_touch','front_load')}
        self.circuit = first.Circuit({'ports':self.ports})
        self.receptors = load_receptors(self.circuit)
        self.received = first.Receiver(self.circuit)
        anchors = read(BIDIR/'PHYSICAL_AN_PAIR/ground_forward/RESULT.json')['probes']
        for key, (_,_,sign) in self.received.probes.items():
            roots = anchors['/'.join(key)]
            self.received.probes[key] = roots,[self.circuit.index[r] for r in roots],sign
        source_group = {'a01':'walking_steer_context','a02':'walking_steer'}
        self.cut = np.zeros(len(self.ports),dtype=bool)
        if cut_mode in ('all','speed'):
            self.cut = np.array([p['direction']=='DN' and (cut_mode=='all' or p['group']=='forward_speed')
                                 for p in self.ports.values()])
        if cut_mode == 'steer':
            cell_type,side = name.split('_')
            self.cut = np.array([p['group']==source_group[cell_type] and p['side']==side for p in self.ports.values()])
            assert self.cut.sum()==1
        self.edrive = np.zeros((1,len(self.ports)),dtype=bool)
        self.dn_idx = [self.circuit.port_index[r] for r in order]
        self.local_drive = np.zeros((1,len(self.circuit.roots)),dtype=np.float32)
        self.level = np.zeros(6)
        self.command = np.zeros(2)
        self.calibration = calibration

    def step(self,tick,brain,forces=None):
        self.local_drive.fill(0.)
        if forces is not None and tick % 23 == 0:
            for force,side in zip(forces,SIDE,strict=True):
                for group,threshold in (('front_touch',.04),('front_load',.20)):
                    if force>threshold:
                        self.local_drive[0,self.receptors[group,side]]=68.75
        self.edrive[0,self.dn_idx] = brain
        sent,arrived,_,voltage = self.circuit.step(self.local_drive,self.cut,eon_output=self.edrive)
        if tick%5==0:
            self.received.update(voltage,DT*5)
            now = np.asarray([self.received.level(group,side) for group in RECEIVING for side in SIDE])
            self.level += (now-self.level)*min(1.,DT*5/TAU)
            if self.calibration:
                self.command = decode(self.level,self.calibration)
        return sent[0],arrived[0]


def decode(level,params):
    avg = max(0.,float(level[:2].mean()))
    speed = np.interp(avg, params['speed_breaks_mv'], [0.,0.,.47,1.])
    steering = []
    for offset,key in ((2,'a01'),(4,'a02')):
        differential = float(level[offset]-level[offset+1])
        dead = params[key]['dead_mv']
        gain = params[key]['usable_mv']['left' if differential>=0 else 'right']
        turn = math.copysign(.28*np.clip((abs(differential)-dead)/gain,0.,1.),differential)
        steering.append(turn)
    # Keep the studied two-dimensional gait range when the steering DNs coactivate.
    steer = float(np.clip(sum(steering),-.28,.28))
    turn_gain = min(1.,float(speed)/.28)
    left_right = np.clip(np.array([speed-turn_gain*steer,speed+turn_gain*steer]),0.,1.)
    return np.where(left_right>=.035,left_right,0.)


def probe(folder,out):
    meta, events = brain_record(folder)
    neck = Overlap(meta['roots'],'open',meta['name'])
    observations=np.empty((len(events),6))
    arrivals=np.empty((len(events),len(neck.ports)),dtype=bool)
    for tick, emitted in enumerate(events):
        _,arrivals[tick]=neck.step(tick,emitted)
        observations[tick]=neck.level
    out.mkdir()
    np.savez_compressed(out/'LEVELS.npz',level=observations,arrivals=arrivals)
    central = observations[6300:10700]
    print(meta['name'],np.round(np.quantile(central,[.1,.5,.9],axis=0),3).tolist(),flush=True)
    write(out/'RESULT.json',dict(neural_input=str(folder.resolve()),neural_sha=digest(folder/'BRAIN.npz'),
          levels_sha=digest(out/'LEVELS.npz'),tracked_ports=list(neck.ports),probes=neck.received.manifest()))


def calibrate(source,out):
    series={}
    files={}
    for name in ('off',*NAMES):
        trace=source/name/'LEVELS.npz'
        result=read(source/name/'RESULT.json')
        assert result['levels_sha']==digest(trace)
        with np.load(trace,allow_pickle=False) as data:
            series[name]=data['level'][6300:10700]
        files[str(trace.resolve())]=result['levels_sha']
    speed_lo=np.median(series['cruise'][:,:2].mean(axis=1))
    speed_hi=np.median(series['fast'][:,:2].mean(axis=1))
    assert speed_hi>1.15*speed_lo>.01
    thresholds={'speed_breaks_mv':[0.,float(.20*speed_lo),float(speed_lo),float(speed_hi)],
                'source':files,'window_ticks':[6300,10700],'smoothing_seconds':TAU}
    for key,j in (('a01',2),('a02',4)):
        diffs = {name:sample[:,j]-sample[:,j+1] for name,sample in series.items()}
        dead=max(float(np.quantile(np.abs(diffs['cruise']),.98)), .025)
        left=float(np.median(diffs[key+'_left']))
        right=float(-np.median(diffs[key+'_right']))
        assert min(left,right)>dead,(key,left,right,dead)
        thresholds[key]={'dead_mv':dead,'median_mv':{'left':left,'right':right},
                         'usable_mv':{'left':left-dead,'right':right-dead}}
    write(out,thresholds)
    print(thresholds,flush=True)


def physical_case(meta,mode):
    name=meta['name']
    direction=name.rsplit('_',1)[-1]
    family='arc_'+direction if name.startswith(('a01','a02','pfl')) else name
    return study.Case(name=meta['name'],family=family,seed=meta['seed'],pulse_every=0,start=ON,
                      stop=OFF,steer_cut=mode=='steer',push_side=-1 if meta['seed']==362 else 0)


@torch.inference_mode()
def rollout(folder,calibration,mode,out):
    torch.set_num_threads(1)
    params=read(calibration)
    meta,events=brain_record(folder)
    case=physical_case(meta,mode)
    body=study.Body()
    body.at_start(case.seed)
    oscillator=Oscillator(body)
    actor=read_policy(WEIGHT)
    neck=Overlap(meta['roots'],mode,case.name,params)
    raw=np.empty((TICKS,15))
    voltages=np.empty((TICKS,6))
    emitted=np.empty((TICKS,len(neck.ports)),dtype=bool)
    arrived=np.empty_like(emitted)
    positions,controls=[],[]
    for tick, incoming in enumerate(events):
        contacts=body.contact()
        found,forces=contacts
        emitted[tick],arrived[tick]=neck.step(tick,incoming,np.linalg.norm(forces[body.fronts],axis=1))
        oscillator.step(neck.command)
        if tick%HOLD==0:
            local=body.observation(neck.command,contacts)
            target=body.clip_action(body.neutral+actor.act(local,oscillator.features(),True))
            positions.append(body.d.qpos.copy())
            controls.append(target.copy())
        apply_push=6500<=tick<6700 and mode not in ('all','speed') and case.name!='off'
        body.d.xfrc_applied[body.thorax,1]=case.push_side*.12*body.weight if apply_push else 0.
        body.apply(target)
        rot=body.d.xmat[body.thorax]
        raw[tick]=[(tick+1)*DT,*body.d.xpos[body.thorax],rot[8],math.atan2(rot[3],rot[0]),
                   *neck.command,body.nonfoot(),*found]
        voltages[tick]=neck.level
    assert not body.d.warning.number.any() and np.isfinite(raw).all()
    if mode in ('all','speed') or case.name=='off':
        still=FirstCase(case.name,case.seed,ON,OFF,cut_end=TICKS)
        result=first.score(still,raw)
    else:
        result=study.score(case,raw)
    out.mkdir()
    trace=out/'TRACE.npz'
    np.savez_compressed(trace,trace=raw,emitted=emitted,arrived=arrived,levels=voltages,
                        ctrl=np.asarray(controls),qpos=np.asarray(positions))
    result.update(mode=mode,case=asdict(case),ports=list(neck.ports),dn_ports=meta['roots'],
                  cut=[r for r,c in zip(neck.ports,neck.cut,strict=True) if c],
                  trace_sha256=digest(trace),calibration_sha256=digest(calibration),body_weight_sha256=digest(WEIGHT),
                  initial_sha256=digest(first.MODEL_FILE),brain_sha256=digest(folder/'BRAIN.npz'),
                  neural_input=str(folder.resolve()))
    write(out/'RESULT.json',result)
    print(case.name,mode,result['pass_'],round(result['active_speed_mm_s'],3),
          round(result.get('yaw_degrees',0),3),[k for k,v in result['gates'].items() if not v],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('probe','calibrate','open','steer','all','speed'))
    parser.add_argument('source',type=Path)
    parser.add_argument('out',type=Path)
    parser.add_argument('--calibration',type=Path,default=HERE/'CALIBRATION.json')
    a=parser.parse_args()
    torch.set_num_threads(1)
    if a.action=='probe':
        probe(a.source,a.out)
    elif a.action=='calibrate':
        calibrate(a.source,a.out)
    else:
        rollout(a.source,a.calibration,a.action,a.out)
