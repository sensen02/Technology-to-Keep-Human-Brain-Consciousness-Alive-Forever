"""Predeclared terrain, target, and start configurations for flat food approaches."""
import numpy as np
from collections import namedtuple

SLICE=namedtuple('Scenario','band seed goal_x goal_y yaw_deg strength')
BANDS=('near','middle','far')
SEEDS={'near':(88300,88360), 'middle':(88360,88420), 'far':(88420,88480)}


def scenario(seed):
    band=next(k for k,(lo,hi) in SEEDS.items() if lo<=seed<hi)
    index=seed-SEEDS[band][0]
    side=1 if index%2==0 else -1
    rng=np.random.default_rng(seed)
    x={'near':(.74,.94),'middle':(1.00,1.25),'far':(1.40,1.67)}[band]
    y={'near':(.065,.17),'middle':(.055,.165),'far':(.052,.14)}[band]
    goal_x,goal_y=float(rng.uniform(*x)),float(side*rng.uniform(*y))
    initial_bearing=np.rad2deg(np.arctan2(goal_y,goal_x))
    yaw_range={'near':9.,'middle':8.,'far':5.}[band]
    yaw=float(np.clip(initial_bearing+rng.uniform(-yaw_range,yaw_range),-16.,16.))
    strength=float(rng.uniform(1.03,1.24)) if band=='far' else float(rng.uniform(.85,1.23))
    return SLICE(band,seed,goal_x,goal_y,yaw,strength)
