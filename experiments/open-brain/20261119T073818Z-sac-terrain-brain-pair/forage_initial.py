"""Fast navigation curriculum using measured Eon/BANC command response tables."""
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
from engine.embodied.feeding import Food, FeedingBody, FeedingState

DT = .16
GAMMA = .985
HORIZON = 72
MEAL = .18
OBS = 14


class MotorTable:
    def __init__(self, rows):
        self.paces = np.array(sorted({r['intent'][0] for r in rows}))
        self.turns = np.array(sorted({r['intent'][1] for r in rows}))
        self.table = np.array([r['drive'] for r in sorted(rows,key=lambda r:r['intent'])]).reshape(
            len(self.paces),len(self.turns),2)

    def drive(self, action):
        pace = .76*(float(action[0])+1.)/2.
        turn = .22*float(action[1])
        # Scale the first measured pair to the verified zero-input rest.
        rows = np.stack([np.interp(turn,self.turns,self.table[i,:,j])
            for j in range(2) for i in range(len(self.paces))]).reshape(2,-1)
        left,right = [np.interp(pace,np.r_[0.,self.paces],np.r_[0.,line]) for line in rows]
        return .98*np.clip((left+right)/1.4,0.,1.), np.clip(3.25*(left-right),-2.6,2.6)


@dataclass(frozen=True)
class Terrain:
    near: float
    far: float
    edge: float
    side: int
    goal_x: float
    goal_y: float
    roughness: float

    @classmethod
    def from_seed(cls, seed, level, side=None):
        rng = np.random.default_rng(seed)
        near = rng.uniform(.71,.99)
        return cls(near,near+rng.uniform(.23,.35),rng.uniform(.26,.46),
                   0 if level==0 else (int(rng.choice((-1,1))) if side is None else side),
                   rng.uniform(2.05,2.45),rng.uniform(-.095,.095),
                   0. if level<2 else rng.uniform(.10,.26))

    @property
    def goal(self):
        return np.array([self.goal_x,self.goal_y])

    def blocked(self, points):
        points=np.asarray(points)
        x,y=points[...,0],points[...,1]
        return ((x>self.near)&(x<self.far)&(self.side*y<self.edge)) if self.side else np.zeros_like(x,dtype=bool)

    def distance(self, point):
        direct=np.linalg.norm(self.goal-point)
        if not self.side or point[0]>=self.far:
            return direct
        edge_y=self.side*(self.edge+.045)
        entry=np.array([self.near,edge_y])
        leave=np.array([self.far,edge_y])
        d=self.goal-point
        if point[0]<self.near and all(self.side*(point[1]+(x-point[0])*d[1]/d[0])>=self.edge
                                       for x in (self.near,self.far)):
            return direct
        if point[0]>=self.near:
            return float(np.linalg.norm(leave-point)+np.linalg.norm(self.goal-leave))
        return float(np.linalg.norm(entry-point)+(self.far-self.near)+np.linalg.norm(self.goal-leave))


class Course:
    def __init__(self, motor):
        self.motor = motor
        self.feeding = FeedingBody()

    def reset(self,seed,level,side=None,food=True):
        self.terrain = Terrain.from_seed(seed,level,side)
        rng=np.random.default_rng(seed+4627)
        self.position=np.array([0.,rng.uniform(-.055,.055)])
        self.heading=float(rng.uniform(-.10,.10))
        self.speed=self.turn=0.
        self.contacts=np.zeros(2,dtype=np.float32)
        self.state=FeedingState(Food(.9,.81) if food else Food(0.,0.),reserve=.26,circulating=.23)
        self.food=food
        self.steps=self.hits=self.stalled=0
        self.ingested=0.
        self.cost=self.terrain.distance(self.position)
        self.crossed_side=None
        return self.observe()

    def observe(self):
        d=self.terrain.goal-self.position
        angle = np.arctan2(d[1],d[0])-self.heading
        seen = self.food and np.cos(angle)>.24
        angles = self.heading + np.linspace(-1.05,1.05,6)
        directions = np.column_stack((np.cos(angles),np.sin(angles)))
        radii = np.linspace(.055,.83,20)
        rays=self.position+radii[None,:,None]*directions[:,None,:]
        intersections=self.terrain.blocked(rays)
        hazard=(intersections*(1-radii[None,:]/.90)).max(axis=1)
        distance=min(1.,np.linalg.norm(d)/2.6) if seen else 1.
        return np.array([float(seen),np.sin(angle) if seen else 0.,
                         np.cos(angle) if seen else 0.,distance,*hazard,*self.contacts,
                         self.speed,.30-self.state.circulating],dtype=np.float32)

    def step(self, action):
        self.steps += 1
        self.contacts[:]=0.
        v,w=self.motor.drive(action)
        distance=np.linalg.norm(self.terrain.goal-self.position)
        contact = bool(self.food and distance<=.14)
        self.speed=.40*self.speed+.60*(0. if contact else v)
        self.turn=.36*self.turn+.64*(0. if contact else w)
        self.heading += DT*self.turn
        vector=np.array([np.cos(self.heading),np.sin(self.heading)])
        rough=1.-self.terrain.roughness*(.5+.5*np.sin(9.*self.position[0]+2.*self.position[1]))
        displacement=DT*self.speed*rough*vector
        trial=self.position+displacement
        hit=bool(self.terrain.blocked(trial) or abs(trial[1])>1.36 or trial[0]<-.30)
        if hit:
            self.contacts[:]=1.
            self.hits+=1
            self.stalled+=1
            self.speed*=.22
        else:
            if self.position[0]<self.terrain.far<=trial[0] and self.crossed_side is None:
                self.crossed_side=int(np.sign(trial[1]))
            self.position=trial
            self.stalled=0
        effort = .018*abs(v)*DT + .013*abs(w)*DT
        at_food = bool(self.food and np.linalg.norm(self.terrain.goal-self.position)<=.14)
        eating = at_food and not hit and self.speed<=.30
        fed=self.feeding.advance(self.state,DT,feed=eating,contact=at_food,
                                  activity_use=.012*abs(v))
        self.state=fed.state
        self.ingested+=fed.ingested.volume
        success = self.ingested>=MEAL
        fall = bool(hit and (self.stalled>=3 or v>.84))
        term=success or fall
        timeout=self.steps==HORIZON and not term
        remaining=self.terrain.distance(self.position)
        # Goal distance shapes training only. Terminal potential is zero by
        # definition, including falls, to preserve the discounted objective.
        old_potential=-.7*self.cost
        next_potential=0. if term else -.7*remaining
        route=GAMMA*next_potential-old_potential if self.food else 0.
        reward=(5.0*fed.ingested.volume/MEAL + 7.0*success - 6.0*fall -
                .22*hit - .014 - effort + route)
        self.cost=remaining
        return self.observe(),float(reward),term,timeout,dict(success=bool(success),fall=fall,
            contact=at_food,ingested=float(self.ingested),remaining=float(remaining),
            crossed_side=self.crossed_side,collision_steps=self.hits)
