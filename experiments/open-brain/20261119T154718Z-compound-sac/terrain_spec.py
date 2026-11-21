"""Deterministic compound courses shared by proxy training and MuJoCo."""
from dataclasses import dataclass
import sys
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
CLOSED=HERE.parent/'20261119T084744Z-closed-loop-forage'
REWARD=HERE.parent/'20261119T152434Z-reward-repair'
sys.path.insert(0,str(CLOSED))
from controller import terrain as old_terrain

STAGES=('flat','wide_bridge','slope_bridge','staggered','staggered_slope')


@dataclass(frozen=True)
class Gate:
    near: float
    far: float
    edge: float
    side: int


@dataclass(frozen=True)
class Terrain:
    gates: tuple
    height_profile: tuple
    goal_x: float
    goal_y: float
    roughness: float

    @property
    def side(self):
        return self.gates[0].side if self.gates else 0

    @property
    def near(self):
        return self.gates[0].near if self.gates else .9

    @property
    def far(self):
        return self.gates[0].far if self.gates else 1.2

    @property
    def edge(self):
        return self.gates[0].edge if self.gates else 0.

    @property
    def goal(self):
        return np.array((self.goal_x,self.goal_y))

    def height(self,x):
        a=np.array(self.height_profile)
        return np.interp(x,a[:,0],a[:,1])

    def slope(self,x):
        return (self.height(x+.007)-self.height(x-.007))/.014

    def blocked(self,points):
        p=np.asarray(points)
        x,y=p[...,0],p[...,1]
        bad=np.zeros_like(x,dtype=bool)
        for g in self.gates:
            bad|=(x>g.near)&(x<g.far)&(g.side*y<g.edge)
        return bad

    def distance(self,p):
        route=np.asarray(p,dtype=float)
        total=0.
        for g in self.gates:
            if route[0]>=g.far:
                continue
            target=g.side*(g.edge+.115)
            if route[0]<g.near:
                q=np.array((g.near-.14,target))
                total+=float(np.linalg.norm(q-route))
                route=q
            q=np.array((g.far,target))
            total+=float(np.linalg.norm(q-route))
            route=q
        return total+float(np.linalg.norm(self.goal-route))


def make_terrain(seed,stage,side):
    if stage not in STAGES or side not in (-1,1):
        raise ValueError((stage,side))
    plain=( (-2.,0.),(5.,0.) )
    if stage in ('flat','wide_bridge'):
        old=old_terrain(seed,stage,side)
        gates=(Gate(old.near,old.far,old.edge,side),) if old.side else ()
        return Terrain(gates,plain,old.goal_x,old.goal_y*side if stage=="flat" else old.goal_y,old.roughness)
    rng=np.random.default_rng(seed+513709)
    x=float(rng.uniform(.86,.99))
    g=Gate(x,x+float(rng.uniform(.225,.275)),float(rng.uniform(.115,.155)),side)
    rise=float(rng.uniform(.037,.067))
    if stage=='slope_bridge':
        profile=((-2.,0.),(.20,0.),(g.near-.12,rise),(g.far+.12,rise),
                 (g.far+.63,0.),(5.,0.))
        return Terrain((g,),profile,float(g.far+rng.uniform(.74,.84)),
                       float(side*rng.uniform(.16,.23)),.045)
    x2=g.far+float(rng.uniform(.91,1.13))
    other=Gate(x2,x2+float(rng.uniform(.215,.267)),float(rng.uniform(.095,.145)),-side)
    if stage=='staggered_slope':
        profile=((-2.,0.),(.20,0.),(g.near-.10,rise),(g.far+.10,rise),
                 (other.near-.15,0.),(other.far+.14,0.),(other.far+.68,-.040),(5.,-.040))
    else:
        profile=plain
    return Terrain((g,other),profile,float(other.far+rng.uniform(.69,.85)),
                   float(-side*rng.uniform(.145,.23)),.065)


def dry_check():
    for name in STAGES:
        a=make_terrain(27261,name,1)
        b=make_terrain(27261,name,-1)
        assert len(a.gates)==len(b.gates) and a.goal_x==b.goal_x and a.goal_y==-b.goal_y
        assert a.height_profile==b.height_profile
        for g,h in zip(a.gates,b.gates,strict=True):
            assert g.side==-h.side
            assert g.near<g.far and h.far==g.far
            point=((g.near+g.far)/2.,0.)
            side=((g.near+g.far)/2.,g.side*(g.edge+.25))
            assert a.blocked(point) and not a.blocked(side)
        x=np.linspace(0.,a.goal_x,800)
        assert abs((a.height(x[1:])-a.height(x[:-1]))/np.diff(x)).max()<=.175
