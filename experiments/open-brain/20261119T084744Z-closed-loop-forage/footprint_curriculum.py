"""Train with a physically measured transverse support margin at the lip."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

import refine_sac

BASE=refine_sac.PhysicalDomain
MIN=0.


class GroundedDomain(BASE):
    def reset(self,seed,stage,side,food=True,noise=True):
        self.optics=None
        result=super().reset(seed,stage,side,food,noise)
        self.optics=self.terrain
        gate=self.traversal(self.optics)
        self.cost=gate.distance(self.position)
        return result

    def traversal(self,cfg):
        if not cfg.side:
            return cfg
        margin=MIN*(1. if self.stage!='wide_bridge' else .85)
        return replace(cfg,near=cfg.near-.02,far=cfg.far+.02,edge=cfg.edge+margin)

    def observe(self):
        actual=self.terrain
        if getattr(self,'optics',None) is not None:
            self.terrain=self.optics
        output=super().observe()
        self.terrain=actual
        return output

    def advance_actor(self,obs,action):
        actual=self.terrain
        self.terrain=self.traversal(actual)
        sample=super().advance_actor(obs,action)
        self.terrain=actual
        return sample


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('vision',type=Path)
    p.add_argument('start',type=Path)
    p.add_argument('out',type=Path)
    p.add_argument('footprint',type=Path)
    p.add_argument('--steps',type=int,default=15000)
    args=p.parse_args()
    evidence=json.loads(args.footprint.read_text())
    MIN=.65*evidence['median_maximum_transverse_reach_mm']/10.
    refine_sac.PhysicalDomain=GroundedDomain
    args.profile='strong'
    print('Learned body margin cm',MIN,flush=True)
    refine_sac.train(args)
