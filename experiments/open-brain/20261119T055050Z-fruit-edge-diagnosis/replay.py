"""Replay actual archived commands and isolate intervention on the fruit-edge turn."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import mujoco
import numpy as np

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'20261119T052620Z-hunger-foraging'
sys.path.insert(0,str(SOURCE))
import orchard_trial as trial

START=4.42
VARIANTS={
    'FADE':('FADE',5.26),
    'HUNGRY':('HUNGRY',3.24),
    'FADE_YAW':('FADE',5.26),
    'FADE_BRAKE':('FADE',5.26),
    'FADE_LATE_YAW':('FADE',5.26),
    'FADE_LATE_BRAKE':('FADE',5.26),
}


def run(variant,out):
    case,duration=VARIANTS[variant]
    trigger={'FADE_LATE_YAW':4.66,'FADE_LATE_BRAKE':4.76}.get(variant,START)
    archive=SOURCE/'VERIFIED'/case
    trace=np.load(archive/'TRACE.npy',allow_pickle=False)
    positions=np.load(archive/'QPOS.npy',allow_pickle=False)
    rows=[json.loads(line) for line in (archive/'EYES.jsonl').read_text().splitlines()]
    with np.load(archive/'NEURAL.npz',allow_pickle=False) as stored:
        commands={name:stored[name].copy() for name in stored.files}
    gate=np.load(archive/'NUTRIENT_GATE.npy',allow_pickle=False)
    original_walk=trial.ReversedYawWalk
    original_contacts=trial.Contacts
    activity=[]
    contact_data=[]
    first_nonfoot=None
    headings=[]
    earliest_cam_divergence=None

    class ArchivedTarget:
        def __init__(self,eye):
            pass

        def select(self,eye,items,now):
            row=rows[round(now/.02)]
            return row['selection'],row['candidates']

    class ArchivedEyes:
        def __init__(self,phy):
            pass

        def sample(self):
            return None,None,(),None

    class RecordedStream:
        def __init__(self,*args,**kwargs):
            self.config=None
            self.queue=[]
            self.index=0
            self.stdin=self
            self.stdout=self

        def __enter__(self):
            return self

        def __exit__(self,*args):
            return False

        def write(self,line):
            nonlocal earliest_cam_divergence
            row=json.loads(line)
            if self.config is None:
                self.config=row
                self.queue.append(dict(ready=True))
                return
            j=self.index
            refer=rows[j]
            effective={k:refer[k] for k in row}
            if row!=effective and earliest_cam_divergence is None:
                earliest_cam_divergence=[j,round(j*.02,6),[k for k in row if row[k]!=effective[k]]]
            t=j*200
            self.queue.append(dict(walk=commands['ground'][t:t+200:2].tolist(),
                  feed=commands['feed'][t:t+200:2].tolist(),hunger=gate[j].tolist()))
            self.index+=1

        def flush(self):
            pass

        def readline(self):
            return json.dumps(self.queue.pop(0))+'\n'

        def wait(self):
            assert self.index==round(duration/.02)
            n=self.config['ticks']
            np.savez_compressed(self.config['stream'],**{
                name:a[:self.index] if name=='head' else a[:n]
                for name,a in commands.items()
            })
            return 0

    class AuditedWalk(original_walk):
        def action(self,phy,speed,yaw_rate,live_observation):
            t=phy.time()
            before=(speed,yaw_rate)
            if t>=trigger-1e-10:
                if variant=='FADE_LATE_YAW':
                    yaw_rate=0.
                if variant=='FADE_YAW':
                    yaw_rate=np.clip(yaw_rate,-.44,.44)
                if variant in ('FADE_BRAKE','FADE_LATE_BRAKE'):
                    speed=min(.492,speed)
            activity.append((t,*before,speed,yaw_rate,self.heading))
            return super().action(phy,speed,yaw_rate,live_observation)

    class AuditedContacts(original_contacts):
        def __init__(self,phy):
            super().__init__(phy)
            m=phy.model
            self.names=[(m.id2name(i,'geom') or m.id2name(m.geom_bodyid[i],'body') or '')
                        for i in range(m.ngeom)]
            self.force_buffer=np.empty(6)

        def sample(self):
            nonlocal first_nonfoot
            floor,fruit,feet,nonfoot,claws=super().sample()
            t=self.phy.time()
            start=4.54 if case=='FADE' else 2.40
            end=5.08 if case=='FADE' else 3.1
            if start<=t<=end:
                self.capture(t)
            if nonfoot and first_nonfoot is None:
                first_nonfoot=[t,nonfoot]
            if round(t/.00005)%20==0:
                torso=self.phy.named.data.xquat['walker/thorax']
                headings.append((t,*trial.land.angles(torso)))
            return floor,fruit,feet,nonfoot,claws

        def capture(self,t):
            for i,c in enumerate(self.phy.data.contact[:self.phy.data.ncon]):
                if c.efc_address<0:
                    continue
                a,b=int(c.geom1),int(c.geom2)
                if self.bodies[b][0]:
                    a,b=b,a
                if not self.bodies[a][0] or self.bodies[b][0]:
                    continue
                walker,foot,body=self.bodies[a]
                target=self.names[b]
                if foot is not None and target!=trial.RESOURCES[1].geom:
                    continue
                mujoco.mj_contactForce(self.phy.model.ptr,self.phy.data.ptr,i,self.force_buffer)
                force=float(self.force_buffer[0])
                if force>1e-12:
                    contact_data.append((t,body,target,*map(float,c.pos),
                                         abs(float(c.frame[2])),force))

    trial.FruitSearch=ArchivedTarget
    trial.OrchardEyes=ArchivedEyes
    trial.subprocess=SimpleNamespace(Popen=RecordedStream,PIPE=None)
    trial.ReversedYawWalk=AuditedWalk
    trial.Contacts=AuditedContacts
    trial.TRIALS[case]=(*trial.TRIALS[case][:2],duration)
    trial.run(case,out)
    np.save(out/'APPLIED_WALK.npy',np.array(activity),allow_pickle=False)
    np.save(out/'TORSO_ANGLES.npy',np.array(headings),allow_pickle=False)
    np.savez_compressed(out/'CONTACTS.npz',
        time=np.array([x[0] for x in contact_data]),pair=np.array([x[1:3] for x in contact_data]),
        physical=np.array([x[3:] for x in contact_data]))
    replay=np.load(out/'TRACE.npy',allow_pickle=False)
    actual=np.load(out/'QPOS.npy',allow_pickle=False)
    t_first=next((float(replay[i,0]) for i in range(len(replay))
                  if not np.array_equal(replay[i,:8],trace[i,:8])),None)
    facts=dict(case=variant,neural_steps_new=0,relative_to_original_drive=case,
               intervention_starts_s=trigger if variant!=case else None,
               reference_trace_exact=bool(np.array_equal(replay,trace[:len(replay)])),
               reference_qpos_exact=bool(np.array_equal(actual,positions[:len(actual)])),
               first_body_difference_s=t_first,
               first_aligned_feedback_difference=earliest_cam_divergence,
               first_nonfoot=first_nonfoot,
               past_start_nonfoot_ms=float(np.count_nonzero((replay[:,0]>=START if case=='FADE'
                   else replay[:,0]>=2.30)&(replay[:,6]>0))*.05),
               post_start_min_support=int((replay[:,4]+replay[:,5])[replay[:,0]>=START if case=='FADE'
                   else replay[:,0]>=2.30].min()))
    (out/'DIAGNOSTIC.json').write_text(json.dumps(facts,indent=2)+'\n')
    print(facts,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',choices=VARIANTS)
    parser.add_argument('out',type=Path)
    args=parser.parse_args()
    run(args.case,args.out.resolve())
