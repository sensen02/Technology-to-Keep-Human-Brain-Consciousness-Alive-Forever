"""Predeclared complete-trajectory splits with different actual spike schedules."""
from dataclasses import asdict, dataclass
import json
from context import HERE,study,ground


@dataclass(frozen=True)
class Task:
    name: str
    suite: str
    family: str
    seed: int
    period: int
    start: int
    action: int
    stop: int
    push: int = 0
    cut_end: int = 0
    cut_start: int = 0
    mode: str = 'open'

    def body_case(self):
        brake=self.action if self.family=='brake' else 0
        # Ground family-specific activation is represented directly below.
        return study.Case(self.name,self.family,self.seed,self.period,self.start,self.stop,
                          push_side=self.push)

    def ground_case(self):
        return ground.Case(self.family,self.seed,self.period,self.start,self.action,self.stop,self.push)


FAMILIES=('cruise','fast','arc_left','arc_right','back','brake')
TRAIN=[]
for index,family in enumerate(FAMILIES):
    for variant,period in enumerate((54,60,67)):
        start=1900+170*variant
        TRAIN.append(Task(f'tr_{family}_{variant}','body',family,1101+23*index+variant,period,start,0,
                          12010+280*variant,push=(1 if variant==2 else 0)))
for index,family in enumerate(('stride_left','stride_right','bluebell','web_b')):
    for variant,period in enumerate((55,61,68)):
        start=2050+130*variant
        activation=4950+120*variant if family.startswith('stride') else 7750+120*variant
        stop=13040+110*variant if family.startswith('stride') else 14100+110*variant
        TRAIN.append(Task(f'tr_{family}_{variant}','ground',family,1253+19*index+variant,period,start,activation,stop,
                          push=-1 if variant==2 else 0))
for index,family in enumerate(('cruise','fast','a01_left','a01_right','a02_left','a02_right')):
    TRAIN.append(Task(f'tr_eon_{family}','eon',family,1431+17*index,0,2300,3800,11800))
TRAIN.extend((Task('tr_restore','body','cruise',1551,62,2230,0,12700,cut_end=4800),
              Task('tr_allcut','body','cruise',1553,62,2230,0,12700,cut_end=17000),
              Task('tr_cut_mid','body','cruise',1557,62,2230,0,12700,cut_end=17000,cut_start=8500)))
DEVELOP=[Task(f'dev_{family}','body',family,1601+29*i,57,2350,0,12580) for i,family in enumerate(FAMILIES)]
DEVELOP += [Task(f'dev_{family}','ground',family,1823+19*i,58,2290,
                 5170 if family.startswith('stride') else 7890,
                 13240 if family.startswith('stride') else 14220)
            for i,family in enumerate(('stride_left','stride_right','bluebell','web_b'))]
CORRECT=[Task(f'correct_{case.name}','body' if case.suite=='body' else 'ground',case.family,
              case.seed+601,64,case.start+90,case.action+100 if case.action else 0,case.stop+100,push=-1)
         for case in DEVELOP]
EXTRA=[]
for i,family in enumerate(('stride_left','stride_right','bluebell','web_b')):
    for mode in ('open','cut'):
        EXTRA.append(Task(f'extra_{family}_{mode}','ground',family,2611+31*i,66,2370,
                    5390 if family.startswith('stride') else 8130,
                    13450 if family.startswith('stride') else 14440,push=-1,mode=mode))
for i,family in enumerate(('cruise','fast','back','brake')):
    EXTRA.append(Task(f'extra_{family}','body',family,2861+31*i,66,2370,0,12830,push=1))

TASKS={task.name:task for task in TRAIN+DEVELOP+CORRECT+EXTRA}
if __name__=='__main__':
    with (HERE/'TASKS.json').open('x') as stream:
        json.dump({name:[asdict(t) for t in rows] for name,rows in
                   (('train',TRAIN),('develop',DEVELOP),('correct',CORRECT),('extra',EXTRA))},
                   stream,indent=2)
        stream.write('\n')
