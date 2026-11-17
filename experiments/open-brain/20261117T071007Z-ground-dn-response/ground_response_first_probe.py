"""Test DNg13 and two halts through the source-qualified BANC ground interface."""
import argparse
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parent/'20261117T062447Z-body-commands'
sys.path.insert(0,str(BASE/'PHASE'))
from phase import Oscillator, read_policy, study
import body_run as first
from physical import load_receptors, schedule_indices

DT = first.DT
TICKS = first.TICKS
WEIGHT = BASE/'PHASE/FIT/epoch015.pt'
CATALOG = first.PARENT/'CATALOG.json'
PROBES = first.PARENT/'PHYSICAL_AN_PAIR/ground_forward/RESULT.json'
GROUPS = {'forward','forward_speed','walking_steer','walking_steer_context',
          'walking_outer_stride','halt_bluebell','halt_web_a','halt_web_b','backward_moonwalker',
          'front_touch','front_load'}
TARGETS = {'stride_left':('walking_outer_stride','left'),
           'stride_right':('walking_outer_stride','right'),
           'bluebell':('halt_bluebell',None), 'web_b':('halt_web_b',None)}
SIDES=('left','right')
TELEMETRY=('forward','walking_outer_stride','halt_bluebell','halt_web_b')


@dataclass(frozen=True)
class Case:
    name: str
    seed: int
    period: int
    on: int
    action: int
    off: int
    push: int

    @property
    def stride(self):
        return self.name.startswith('stride')

    def schedule(self):
        group, side = TARGETS[self.name]
        start=self.action
        return [('forward',None,self.on,self.off),(group,side,start,self.off)]


def case_for(name, variant):
    # Stimulus times and seeds are fixed before either development or held runs.
    rows={'dev':(101,60,2200,7600,13800,0),
          'held0':(225,59,2260,7820,14100,0),
          'held1':(318,63,2410,8010,14340,-1)}
    seed, period, on, start, off, push=rows[variant]
    if name.startswith('stride'):
        start-=2700
        off-=1100
    return Case(name, seed+list(TARGETS).index(name)*19,period,on,start,off,push)


class Ground:
    def __init__(self,case,operative,calibration):
        catalog = first.json.loads(CATALOG.read_text())['ports']
        self.ports={root:port for root,port in catalog.items() if port['group'] in GROUPS}
        self.circuit=first.Circuit({'ports':self.ports})
        self.receptors=load_receptors(self.circuit)
        self.receiver=first.Receiver(self.circuit)
        anchors=first.json.loads(PROBES.read_text())['probes']
        for key,(_,_,sign) in self.receiver.probes.items():
            roots=anchors['/'.join(key)]
            self.receiver.probes[key]=roots,[self.circuit.index[root] for root in roots],sign
        group, side = TARGETS[case.name]
        self.cuts=np.asarray([operative and port['group']==group and (side is None or port['side']==side)
                              for port in self.ports.values()],dtype=bool)
        self.exc=schedule_indices(self.circuit,case.schedule())
        self.drive=np.zeros((1,len(self.circuit.roots)),dtype=np.float32)
        self.command=np.zeros(2)
        self.calibration=calibration

    def step(self,tick,period,front_forces):
        self.drive.fill(0.)
        if tick%period==0:
            for columns,first_tick,stop in self.exc:
                if first_tick<=tick<stop:
                    self.drive[0,columns]=68.75
        if tick%23==0:
            for force,side in zip(front_forces,SIDES,strict=True):
                for group,threshold in (('front_touch',.04),('front_load',.20)):
                    if force>threshold:
                        self.drive[0,self.receptors[group,side]]=68.75
        sent,arrived,_,voltage=self.circuit.step(self.drive,self.cuts)
        if tick%5==0:
            self.receiver.update(voltage,DT*5)
            self.command=self.decode()
        return sent[0],arrived[0]

    def decode(self):
        if self.calibration is None:
            return self.receiver.ground()
        # Use the established ground output range on the newly measured VNC groups.
        r=self.receiver
        stride=r.level('walking_outer_stride','left')-r.level('walking_outer_stride','right')
        # The approved .28 DNa02 ceiling is a bound; DNg13 remains a smaller adjustment.
        if stride==0.:
            turn=0.
        else:
            side='left' if stride>0 else 'right'
            turn=.25*np.clip(stride/self.calibration['stride_mv'][side],-1.,1.)
        brake=max(r.pair('halt_bluebell')/self.calibration['bluebell_mv'],
                  r.pair('halt_web_b')/self.calibration['web_b_mv'])
        # The legacy step readout still owns the previously certified groups.
        saved=r.filtered
        r.filtered={key:(0. if key[0] in ('walking_outer_stride','halt_bluebell','halt_web_b') else value)
                    for key,value in saved.items()}
        try:
            baseline=r.ground()
        finally:
            r.filtered=saved
        result=np.clip((baseline+np.array([-turn,turn]))*(1.-np.clip(brake,0.,1.)),-1.,1.)
        return np.where(np.abs(result)>=.035,result,0.)

    def levels(self):
        return [self.receiver.level(group,side) for group in TELEMETRY for side in SIDES]


def motion_window(trace,bounds):
    lower,upper=bounds
    block=trace[lower:upper]
    xy=block[:,1:3]
    step=np.diff(xy,axis=0)
    yaw=block[:,5]
    directions=np.stack((np.cos(yaw[:-1]),np.sin(yaw[:-1])),axis=1)
    contact=block[:,9:15]>0
    rises=(~contact[:-1]&contact[1:]).sum(0)
    falls=(contact[:-1]&~contact[1:]).sum(0)
    return dict(forward_speed_mm_s=float(np.sum(step*directions)/(DT*len(step))),
                rms_mm_s=float(np.sqrt(np.mean(np.sum(step**2,axis=1)))/DT),
                drift_mm=float(np.max(np.linalg.norm(xy-xy[0],axis=1))),
                support_fraction=float(np.mean(contact.sum(1)>=3)),
                cycles=np.minimum(rises,falls).tolist())


def swing_lengths(trace, feet, bounds):
    start, stop=bounds
    by_foot=[]
    for i in range(6):
        contact=trace[start:stop,9+i]>0
        leave=np.flatnonzero(contact[:-1]&~contact[1:])+start+1
        land=np.flatnonzero(~contact[:-1]&contact[1:])+start+1
        swings=[]
        for first_tick in leave:
            ends=land[land>first_tick]
            if ends.size and ends[0]-first_tick>=40:
                swings.append(float(np.ptp(feet[first_tick:ends[0]+1,i])))
        by_foot.append(swings)
    return by_foot


def first_still(trace,action,off):
    # First 200 ms qualifying uninterrupted window while the brake is still commanded.
    for tick in range(action,off-2000,25):
        q=motion_window(trace,(tick,tick+2000))
        if q['rms_mm_s']<=.30 and q['drift_mm']<=.10 and q['support_fraction']>=.95:
            return tick
    return None


def score(case,mode,trace,feet,legs):
    start=motion_window(trace,(0,1200))
    end=motion_window(trace,(len(trace)-2000,len(trace)))
    gates=dict(start=start['drift_mm']<=.08,
               end=end['drift_mm']<=.10 and end['rms_mm_s']<=.30,
               height=bool(trace[:,3].min()>=.53),upright=bool(trace[:,4].min()>=.80),
               axis=not trace[:,8].any())
    result=dict(start=start,end=end,min_height_mm=float(trace[:,3].min()),min_up=float(trace[:,4].min()),
                axial_contact_steps=int(trace[:,8].sum()))
    a=case.action
    yaw=np.unwrap(trace[:,5])
    if case.stride:
        bounds=(a+1500,case.off-700)
        swing=swing_lengths(trace,feet,bounds)
        mids=[float(np.median(s)) if s else 0. for s in swing]
        inner=case.name.rsplit('_',1)[-1]
        outer=next(s for s in SIDES if s!=inner)
        length=lambda side:float(np.mean([x for leg,x in zip(legs,mids,strict=True) if leg.startswith(side[0])]))
        change=float(np.rad2deg(yaw[case.off-1]-yaw[a]))
        signed=(1 if inner=='left' else -1)*change
        active=motion_window(trace,bounds)
        gates.update(walk=active['forward_speed_mm_s']>=2.5, complete_swings=min(map(len,swing))>=2)
        if mode=='open':
            gates.update(turn=signed>=12., stride=length(outer)-length(inner)>=.04)
        result.update(active=active,yaw_deg=change,signed_yaw_deg=signed,leg_stride_mm=dict(zip(legs,mids,strict=True)),
                      swing_counts=dict(zip(legs,map(len,swing),strict=True)),
                      outer_stride_mm=length(outer),inner_stride_mm=length(inner),outer_minus_inner_mm=length(outer)-length(inner))
    else:
        before=motion_window(trace,(a-3000,a-500))
        braking=motion_window(trace,(a+2600,a+5400))
        gates.update(walking_before=before['forward_speed_mm_s']>=2.5 and min(before['cycles'])>=2)
        if mode=='open':
            gates['brake_stand']=(braking['rms_mm_s']<=.30 and braking['drift_mm']<=.10 and
                                 braking['support_fraction']>=.95)
        else:
            gates['keeps_walking']=braking['forward_speed_mm_s']>=2.5
        stopped=first_still(trace,a,case.off) if mode=='open' else None
        result.update(prior=before,braking=braking,stable_start_tick=stopped)
    return dict(result,pass_=all(gates.values()),gates=gates)


@torch.inference_mode()
def run(case,mode,out,parameter_file):
    torch.set_num_threads(1)
    parameters=first.json.loads(parameter_file.read_text()) if parameter_file else None
    neck=Ground(case,mode=='cut',parameters)
    body=study.Body()
    body.at_start(case.seed)
    controller=read_policy(WEIGHT)
    oscillator=Oscillator(body)
    legs=body.fly.get_legs_order()
    from flygym.anatomy import BodySegment
    indices=[body.fly.get_bodysegs_order().index(BodySegment(f'{leg}_tarsus5')) for leg in legs]
    tips=np.asarray(body.sim._internal_bodyids_by_fly[body.name])[indices]
    raw=np.empty((TICKS,15))
    levels=np.empty((TICKS,8))
    feet=np.empty((TICKS,6))
    sent=np.empty((TICKS,len(neck.ports)),dtype=bool)
    arrived=np.empty_like(sent)
    poses,controls=[],[]
    for tick in range(TICKS):
        contacts=body.contact()
        found,forces=contacts
        sent[tick],arrived[tick]=neck.step(tick,case.period,np.linalg.norm(forces[body.fronts],axis=1))
        oscillator.step(neck.command)
        if tick%first.HOLD==0:
            local=body.observation(neck.command,contacts)
            target=body.clip_action(body.neutral+controller.act(local,oscillator.features(),True))
            poses.append(body.d.qpos.copy())
            controls.append(target.copy())
        push_time=case.action+(1900 if case.stride else -1000)
        body.d.xfrc_applied[body.thorax,1]=.12*body.weight*case.push if push_time<=tick<push_time+200 else 0.
        body.apply(target)
        rot=body.d.xmat[body.thorax]
        pos=body.d.xpos[body.thorax]
        raw[tick]=[(tick+1)*DT,*pos,rot[8],math.atan2(rot[3],rot[0]),*neck.command,body.nonfoot(),*found]
        feet[tick]=(body.d.xpos[tips]-pos)@rot.reshape(3,3)[:,0]
        levels[tick]=neck.levels()
    assert not body.d.warning.number.any() and np.isfinite(raw).all()
    out.mkdir()
    trace=out/'TRACE.npz'
    np.savez_compressed(trace,trace=raw,feet=feet,levels=levels,emitted=sent,arrived=arrived,
                        ctrl=np.asarray(controls),qpos=np.asarray(poses))
    info=score(case,mode,raw,feet,legs)
    info.update(case=asdict(case),mode=mode,trace_sha256=first.sha(trace),calibration_sha256=(
                first.sha(parameter_file) if parameter_file else None),body_weight_sha=first.sha(WEIGHT),
                frozen_init_sha=first.sha(first.MODEL_FILE),ports=list(neck.ports), probe=neck.receiver.manifest(),
                cut_roots=[r for r,cut in zip(neck.ports,neck.cuts,strict=True) if cut],file_sha=first.sha(HERE/'ground_response.py'))
    first.write(out/'RESULT.json',info)
    preview={k:info[k] for k in ('yaw_deg','inner_stride_mm','outer_stride_mm','prior','braking') if k in info}
    print(case.name,mode,info['pass_'],preview,info['gates'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name',choices=TARGETS)
    parser.add_argument('variant',choices=('dev','held0','held1'))
    parser.add_argument('mode',choices=('open','cut'))
    parser.add_argument('out',type=Path)
    parser.add_argument('--calibration',type=Path)
    args=parser.parse_args()
    run(case_for(args.name,args.variant),args.mode,args.out,args.calibration)
