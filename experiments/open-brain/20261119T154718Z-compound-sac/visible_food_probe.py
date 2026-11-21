"""Diagnostic paired live trial with a clearly visible food silhouette."""
import argparse
import json
from pathlib import Path

from terrain_spec import HERE
from common import detail
from live_compound import make_board, run_course_docking
import live_compound
from physical_suite import inspect


def board_with_cue(seed, challenge, side, *, food=True):
    """Keep the registered ground contact radius and inventory."""
    env,cfg,bank = make_board(seed,challenge,side,food=food)
    if food:
        fruit, = bank.resources
        # A red oval attached to the food patch provides a visible profile.
        # It neither supports the body nor triggers ingestion by itself.
        detail(env.task._arena.mjcf_model.worldbody,'test_fruit_visual_profile','ellipsoid',
               pos=(fruit.x,fruit.y,fruit.top+.115),size=(.30,.36,.125),
               rgba=(.84,.065,.024,1.))
    return env,cfg,bank


def run(args):
    plan = json.loads((HERE/'physical_plan.json').read_text())
    case, = (row for row in plan['cases'] if row['name'] == args.case)
    for field in ('challenge', 'seed', 'side', 'duration'):
        setattr(args,field,case[field])
    args.profile, args.no_food, args.early, args.neural_seed = 'strong', False, True, plan['neural_seed']
    args.vision = live_compound.CLOSED/'VISION/VISION.npz'
    run_course_docking.init_course = board_with_cue
    run_course_docking.run(args)
    output = inspect(args.out,plan,case)
    output.update(food_visual_profile=dict(shape='ellipsoid',center_height_above_patch_mm=1.15,
                                           collision=False,body_feeding_unchanged=True))
    (args.out/'AUDIT.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--case',choices=('SLOPE_L','COMPOUND_L'),required=True)
    run(p.parse_args())
