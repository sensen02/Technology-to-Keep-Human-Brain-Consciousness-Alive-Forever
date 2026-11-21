"""Keep wing membranes clear of the abdomen during terminal stow."""
import argparse
import json
from functools import partial
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / '20261118T234622Z-flight-fruit'
sys.path.insert(0, str(ORIGINAL))
import perching
from perching import land

FOLD_ERROR_RAD = np.deg2rad(2.)
# Yaw, roll, pitch in the order used by the stock wing controller.
STOW_ROUTES = {
    'dorsal': ((1.25, -.30, -.85), (1.50, .25, -1.)),
    'pitch': ((1.10, -.10, -1.20), (1.50, .10, -1.20)),
}


class ClearAbdomenLanding(land.LandingMotor):
    """Bring both wings above the abdomen, allow it to relax, then stow."""

    def __init__(self, *args, waypoints, settle_s, **kwargs):
        super().__init__(*args, **kwargs)
        self.waypoints = np.tile(np.asarray(waypoints), (1, 2))
        self.settle_s = settle_s
        self.stage_age = None
        self.stow_ready = None

    def update_episode(self, command, now, feet, nonfeet, physics):
        super().update_episode(command, now, feet, nonfeet, physics)
        if self.fold_started is None:
            return
        self.stage_age = now - self.fold_started - self.approach.rear_hold_s
        actual = physics.bind(self.task._wing_joints).qpos
        if self.stage_age < .045 + self.settle_s + .025 or np.max(np.abs(actual - self.wing_stow)) > FOLD_ERROR_RAD:
            self.landing_phase = 3
        elif self.landing_phase == 4 and self.stow_ready is None:
            self.stow_ready = now

    def wing_target(self, ctrl_freq):
        if self.fold_from is None:
            return super().wing_target(ctrl_freq)
        self.base_pattern(ctrl_freq)
        age = self.stage_age
        lifted, dorsal = self.waypoints
        if age < .020:
            return self.fold_from + land.smooth(age / .020) * (lifted - self.fold_from)
        if age < .045:
            return lifted + land.smooth((age - .020) / .025) * (dorsal - lifted)
        return dorsal + land.smooth((age - .045 - self.settle_s) / .025) * (self.wing_stow - dorsal)


def run(args):
    base = json.loads((ORIGINAL / args.case / 'RESULT.json').read_text())
    neck = ORIGINAL / ('NECK_' + args.case)
    received, commands = land.commands_from(neck)
    policy, stock, observation = land.setup_policy()
    geo = base['geometry']
    motor = partial(ClearAbdomenLanding, waypoints=STOW_ROUTES[args.route], settle_s=args.settle)
    with patch.object(land, 'LandingMotor', motor):
        body = perching.simulate(policy, stock, observation, commands['intact'], received['dt_ms'],
                                 base['seed'], geo['x_cm'], geo['y_cm'], args.duration,
                                 base['approach']['descent_delay_s'], False, not args.no_eye)
    perching.save(args.output, body)
    perching.write(args.output / 'STOW.json',
                   dict(route=args.route, settle_s=args.settle, stock_stow_rad=body['motor'].wing_stow.tolist(),
                        stow_ready_s=body['motor'].stow_ready, post_neck_sha256=received['sha256'],
                        original_source_sha256=land.digest(ORIGINAL / 'perching.py'),
                        stow_source_sha256=land.digest(Path(__file__))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--case', choices=('NEAR', 'FAR'), default='NEAR')
    parser.add_argument('--route', choices=STOW_ROUTES, default='pitch')
    parser.add_argument('--settle', type=float, default=.070)
    parser.add_argument('--duration', type=float, default=perching.DURATION_S)
    parser.add_argument('--no-eye', action='store_true')
    run(parser.parse_args())
