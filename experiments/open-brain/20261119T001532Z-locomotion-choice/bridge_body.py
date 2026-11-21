"""Replay brain-selected takeoff through neutral FlyBody, with native fruit contact."""
import argparse
from functools import partial
import json
from pathlib import Path
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'20261119T000140Z-perch-wing-audit'))
from perch_wing import perching, land, ClearAbdomenLanding, STOW_ROUTES


def run(args):
    receipt, levels=land.commands_from(args.neck)
    policy, stock, observation=land.setup_policy()
    motor=partial(ClearAbdomenLanding,waypoints=STOW_ROUTES['pitch'],settle_s=.070)
    with patch.object(land,'LandingMotor',motor):
        run=perching.simulate(policy,stock,observation,levels[args.lane],receipt['dt_ms'],
                              args.seed,args.x,0.,args.duration,.073,False,True)
    perching.save(args.output,run)
    (args.output/'SOURCE.json').write_text(json.dumps(
        dict(neck=receipt,case=args.lane,teacher='neutral FlyBody, no walked-pose transfer',
             pre_scheduled_takeoff=False),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('neck',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--lane',choices=land.CASES,default='intact')
    p.add_argument('--x',type=float,default=8.03)
    p.add_argument('--seed',type=int,default=29)
    p.add_argument('--duration',type=float,default=1.62)
    run(p.parse_args())
