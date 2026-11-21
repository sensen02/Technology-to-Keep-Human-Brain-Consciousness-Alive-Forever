"""Shared nose observations and inexpensive P1 walking proxy. Lengths are cm."""
from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'20261118T223412Z-fruit-cues'))
from fruit_field import lattice

DT = .08
HORIZON = 150
MEAL = .18
GRID = .035
ORIGIN = (-1.40, -1.75)
X, Y = [start + GRID*np.arange(n) for start, n in zip(ORIGIN, (128, 101))]
XX, YY = np.meshgrid(X, Y, indexing='ij')
NOSE = ((.103, .030), (.103, -.030))
ACTION_PACE = .60
ACTION_TURN = .20


@dataclass(frozen=True)
class Course:
    seed: int
    stage: str
    side: int
    goal_x: float
    goal_y: float
    near: float
    far: float
    low: float
    high: float
    strength: float

    @property
    def goal(self):
        return self.goal_x, self.goal_y


def course(seed, stage, side):
    rng = np.random.default_rng(seed)
    x = float(rng.uniform(1.46, 1.66))
    y = float(side*rng.uniform(.055, .15))
    near = float(rng.uniform(.55, .69))
    short = float(rng.uniform(.10, .20))
    long = float(rng.uniform(.43, .66))
    if stage == 'detour':
        near += .30
        short *= .62
        long *= .81
        x += .23
        y += side*.09
    low, high = (-long, short) if side == 1 else (-short, long)
    strength = float(rng.uniform(.76, 1.24))
    return Course(seed,stage,side,x,y,near,near+.072,low,high,strength)


@lru_cache(maxsize=96)
def odor_field(config):
    """Screened 2-D diffusion. Omit solid face flux and use a co-located source."""
    wall = (config.near <= XX) & (XX <= config.far) & (config.low <= YY) & (YY <= config.high)
    if config.stage == 'open':
        wall[:] = False
    ids, graph = lattice(wall)
    degree = np.asarray(graph.sum(axis=1)).ravel()
    decay = .16
    diffusion = .036/GRID**2
    rhs = np.zeros(len(degree))
    i, j = np.rint((np.asarray(config.goal)-ORIGIN)/GRID).astype(int)
    source = ids[i,j]
    assert source >= 0
    rhs[source] = config.strength/GRID**2
    operator = diags(degree*diffusion+decay)-graph*diffusion
    solution = spsolve(operator,rhs)
    scale = float(max(solution)/config.strength)
    values = np.full(wall.shape, np.nan)
    values[~wall] = solution/scale
    assert solution.min() >= 0. and abs(decay*solution.sum()*GRID**2-config.strength) < 1e-8
    return values


def sample(field, points):
    readings = []
    for x, y in points:
        gx, gy = (x-ORIGIN[0])/GRID, (y-ORIGIN[1])/GRID
        i, j = math.floor(gx), math.floor(gy)
        if not (0 <= i < field.shape[0]-1 and 0 <= j < field.shape[1]-1):
            readings.append(0.)
            continue
        fx, fy = gx-i, gy-j
        corners = ((i,j,(1-fx)*(1-fy)), (i+1,j,fx*(1-fy)),
                   (i,j+1,(1-fx)*fy), (i+1,j+1,fx*fy))
        c = w = 0.
        for a, b, weight in corners:
            value = field[a,b]
            if math.isfinite(value):
                c += value*weight
                w += weight
        readings.append(c/w if w else 0.)
    return readings


class Nose:
    """Head-side features from current and recent bilateral samples."""
    def __init__(self):
        self.last = self.avg = None

    def observe(self, readings, pace, yaw_rate, front=(0.,0.)):
        left, right = readings
        level = .5*(left+right)
        power = math.log1p(36.*level)/3.
        if self.last is None:
            self.last = self.avg = power
        change = power-self.last
        trend = power-self.avg
        contrast = (left-right)/(left+right+1e-5)
        self.last = power
        self.avg = .82*self.avg+.18*power
        # Slots 0 and 3 reserve the legacy RGB target and distance interface.
        # This head never supplies a visual target or inferred target range.
        return np.array([0.,math.log1p(36*left)/3.,math.log1p(36*right)/3.,1.,
                         power,np.clip(14*contrast,-1.,1.),np.clip(9*change,-1.,1.),
                         np.clip(5*trend,-1.,1.),self.avg, float(level>1e-4),
                         *front,pace,yaw_rate],dtype=np.float32)


def intent(obs, action):
    # Apply the same no-signal gate in both proxy and physical closed loop.
    if obs[9] == 0.:
        return 0., 0.
    pace = .20 + (ACTION_PACE-.20)*.5*(float(action[0])+1.)
    turn = ACTION_TURN*float(action[1])
    return pace, turn


def odor_steering(obs):
    """Bilateral attraction reference, available without a learned actor."""
    return np.array([.35,-np.clip(2.2*obs[5],-1.,1.)],dtype=np.float32)


class Proxy:
    def __init__(self, motor, nose=NOSE):
        self.motor = motor
        self.nose_offsets = nose

    def reset(self, cfg, seed, active=True, *, fixed_pose=False):
        self.cfg, self.active = cfg, active
        self.field = odor_field(cfg)
        self.rng = np.random.default_rng(seed)
        self.x = 0.
        self.y = 0. if fixed_pose else float(self.rng.uniform(-.045,.045))
        self.yaw = 0. if fixed_pose else float(self.rng.uniform(-.14,.14))
        self.speed = self.turn = self.food = 0.
        self.nose = Nose()
        self.steps = self.hits = self.bump_run = 0
        self.contact = False
        self.path = [(self.x,self.y)]
        self.vgain = float(self.rng.uniform(.76,1.03))
        self.ugain = float(self.rng.uniform(.80,1.12))
        self.bias = float(self.rng.uniform(-.13,.13))
        self.cost = self.distance()
        return self.observe()

    def distance(self):
        g = self.cfg
        direct = math.hypot(g.goal_x-self.x,g.goal_y-self.y)
        start, end = g.near-.19, g.far+.19
        if g.stage == 'open' or self.x > end:
            return direct
        projected = self.y+(g.goal_y-self.y)*(min(end,max(start,self.x))-self.x)/(g.goal_x-self.x)
        if self.x < start and (projected > g.high+.19 or projected < g.low-.19):
            at_end = self.y+(g.goal_y-self.y)*(end-self.x)/(g.goal_x-self.x)
            if at_end > g.high+.19 or at_end < g.low-.19:
                return direct
        route = [math.hypot(max(0.,start-self.x),j-self.y)+max(0.,end-max(self.x,start))+
                 math.hypot(g.goal_x-end,g.goal_y-j) for j in (g.low-.21,g.high+.21)]
        return min(route)

    def observe(self):
        cos, sin = math.cos(self.yaw), math.sin(self.yaw)
        xy = [(self.x+f*cos-l*sin,self.y+f*sin+l*cos) for f,l in self.nose_offsets]
        odor = sample(self.field,xy) if self.active else (0.,0.)
        if self.active:
            common = 1.+.035*math.sin(self.steps*.45)
            odor = [max(0.,c*(common+self.rng.normal(0.,.010))) for c in odor]
        self.obs = self.nose.observe(odor,self.speed,self.turn)
        return self.obs

    def step(self, action):
        self.steps += 1
        pace, steer = intent(self.obs,action)
        v, w = self.motor(pace,steer)
        if self.contact:
            v = w = 0.
        self.speed = .34*self.speed+.66*v*self.vgain
        self.turn = .42*self.turn+.58*(w*self.ugain+self.bias*(pace>0.))
        self.yaw += DT*self.turn
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        x, y = self.x+DT*self.speed*c, self.y+DT*self.speed*s
        g = self.cfg
        obstacle = g.stage != 'open' and g.near-.14<=x<=g.far+.14 and g.low-.13<=y<=g.high+.13
        hit = obstacle or abs(y)>1.36 or x<-.96 or x>2.64
        if hit:
            self.speed *= .18
            self.hits += 1
            self.bump_run += 1
        else:
            self.x,self.y=x,y
            self.bump_run=0
        self.path.append((self.x,self.y))
        oral = self.x+.15*c, self.y+.15*s
        self.contact = math.hypot(oral[0]-g.goal_x,oral[1]-g.goal_y) <= .34
        intake = DT*.13 if self.contact and not hit and self.speed<=.32 else 0.
        self.food += intake
        success = self.food>=MEAL
        fall = self.bump_run>=7
        end = success or fall
        new_cost = self.distance()
        progress = np.clip(self.cost-new_cost,-.14,.14)
        reward = (5.*progress + .26*intake/MEAL + 4.*success - 4.*fall -
                  .38*hit - .015 - .018*abs(w) - .01*abs(v))
        self.cost = new_cost
        info = dict(meal=bool(success),fall=fall,intake=self.food,hits=self.hits,
                    end_xy=[self.x,self.y],steps=self.steps)
        return self.observe(),float(reward),end,self.steps>=HORIZON,info
