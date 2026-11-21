"""A short-range obstacle cue. It cannot report a food location or identity."""
import math

import numpy as np

from odor_env import Nose, Proxy, sample

RANGE = .72
DETOUR_RANGE = 1.12
RAYS = ((.08,.38,.69,1.00),(-.08,-.38,-.69,-1.00))
ANTENNA_OFFSETS = ((.09599,.01056),(.09728,-.00595))


def ray_hazard(yaw, raycast, limit=RANGE):
    sectors = []
    for offsets in RAYS:
        risk = []
        for off in offsets:
            h = yaw+off
            d = raycast(math.cos(h),math.sin(h))
            risk.append(1.-d/limit if 0. <= d < limit else 0.)
        sectors.append(sum(risk)/len(risk))
    return sectors


def wall_ray(config, x, y, dx, dy, limit=RANGE):
    if config.stage == 'open':
        return -1.
    lo, hi = 0., limit
    for p, v, a, b in ((x,dx,config.near,config.far),(y,dy,config.low,config.high)):
        if abs(v)<1e-10:
            if not a<=p<=b:
                return -1.
        else:
            first, second = sorted(((a-p)/v,(b-p)/v))
            lo, hi = max(lo,first),min(hi,second)
            if lo>hi:
                return -1.
    return lo


class FittedNose(Nose):
    def observe(self, readings, pace, yaw_rate, front=(0.,0.)):
        obs = super().observe(readings,pace,yaw_rate,front)
        obs[5] = np.clip(3.5*obs[5],-1.,1.)
        return obs


class AssistedProxy(Proxy):
    def __init__(self,motor):
        super().__init__(motor,nose=ANTENNA_OFFSETS)

    def reset(self,*args,**kwargs):
        super().reset(*args,**kwargs)
        self.nose = FittedNose()
        return self.observe()

    def distance(self):
        g = self.cfg
        direct = math.hypot(g.goal_x-self.x,g.goal_y-self.y)
        near,far,low,high = g.near-.15,g.far+.15,g.low-.15,g.high+.15
        if g.stage == 'open' or self.x >= far:
            return direct
        dx = g.goal_x-self.x
        a = self.y+(g.goal_y-self.y)*(max(near,self.x)-self.x)/dx
        b = self.y+(g.goal_y-self.y)*(far-self.x)/dx
        if min(a,b)>high or max(a,b)<low:
            return direct
        if self.x>=near:
            end = min((low,high),key=lambda side:abs(self.y-side))
            return math.hypot(far-self.x,end-self.y)+math.hypot(g.goal_x-far,g.goal_y-end)
        return min(math.hypot(near-self.x,end-self.y)+(far-near)+
                   math.hypot(g.goal_x-far,g.goal_y-end) for end in (low,high))

    def observe(self):
        cos,sin = math.cos(self.yaw),math.sin(self.yaw)
        xy = [(self.x+f*cos-l*sin,self.y+f*sin+l*cos) for f,l in self.nose_offsets]
        odor = sample(self.field,xy) if self.active else (0.,0.)
        if self.active:
            common = 1.+.035*math.sin(self.steps*.45)
            odor = [max(0.,c*(common+self.rng.normal(0.,.003))) for c in odor]
        x,y = np.mean(xy,axis=0)
        limit = DETOUR_RANGE if self.cfg.stage=='detour' else RANGE
        danger = ray_hazard(self.yaw,lambda dx,dy:wall_ray(self.cfg,x,y,dx,dy,limit),limit)
        self.obs = self.nose.observe(odor,self.speed,self.turn,danger)
        return self.obs
