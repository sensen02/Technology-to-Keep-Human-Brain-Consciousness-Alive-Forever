"""Run independent 2-second physical baselines; no canonical files modified."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.embodied.bridge_body import BridgeBody, BridgeBodyConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=float, default=2.)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/neural_motor_bridge')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'protocol': 'independent fresh simulation per command; stand/forward/left/right each 2s; stop after 1s forward preconditioning then 2s stop',
              'acceptance': {'upright': 'height >0.5mm and tilt <45deg after first 0.2s',
                             'stand_stop': 'last 0.5s displacement <0.5mm',
                             'forward': 'net world x displacement >5mm',
                             'turn': 'signed unwrapped yaw change >0.3rad'},
              'cases': {}}
    frames = []
    for command in ('stand', 'forward', 'left', 'right', 'stop'):
        with BridgeBody(BridgeBodyConfig(render=args.render)) as body:
            if command == 'stop':
                body.step(command='forward', duration_s=1.)
            initial = body.observe()
            observations = [initial]
            for tick in range(round(args.duration / .01)):
                observations.append(body.step(command=command, duration_s=.01))
                if args.render and tick % 5 == 0:
                    frames.append(body.render())
            t = np.array([o['time_s'] - initial['time_s'] for o in observations])
            p = np.array([o['position_mm'] for o in observations])
            yaw = np.unwrap([o['yaw_rad'] for o in observations])
            tilt = np.rad2deg([o['tilt_rad'] for o in observations])
            tail = t >= max(.2, args.duration - .5)
            settled = t >= .2
            delta = p[-1] - p[0]
            delta_yaw = yaw[-1] - yaw[0]
            stable = bool(np.all(p[settled, 2] > .5) and np.all(tilt[settled] < 45))
            tail_displacement = float(np.linalg.norm(p[-1, :2] - p[tail][0, :2]))
            task = {'stand': tail_displacement < .5, 'stop': tail_displacement < .5,
                    'forward': delta[0] > 5., 'left': delta_yaw > .3,
                    'right': delta_yaw < -.3}[command]
            case = {'duration_s': float(t[-1]), 'displacement_mm': delta.tolist(),
                    'yaw_change_rad': float(delta_yaw), 'height_final_mm': float(p[-1, 2]),
                    'height_min_after_0_2s_mm': float(p[settled, 2].min()),
                    'tilt_max_after_0_2s_deg': float(tilt[settled].max()),
                    'tilt_final_deg': float(tilt[-1]), 'last_0_5s_displacement_mm': tail_displacement,
                    'upright_pass': stable, 'command_pass': bool(task), 'pass': stable and bool(task),
                    'trajectory': {'time_s': t.tolist(), 'position_mm': p.tolist(), 'yaw_unwrapped_rad': yaw.tolist(), 'tilt_deg': tilt.tolist()}}
            report['cases'][command] = case
            report.setdefault('provenance', body.provenance())
            print(command, {k: v for k, v in case.items() if k != 'trajectory'}, flush=True)
    report['all_pass'] = all(c['pass'] for c in report['cases'].values())
    (args.output / 'body_baseline.json').write_text(json.dumps(report, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for command, case in report['cases'].items():
        tr = case['trajectory']; p = np.array(tr['position_mm']); t = tr['time_s']
        axes[0, 0].plot(p[:, 0], p[:, 1], label=command)
        axes[0, 1].plot(t, np.rad2deg(tr['yaw_unwrapped_rad']), label=command)
        axes[1, 0].plot(t, p[:, 2], label=command)
        axes[1, 1].plot(t, tr['tilt_deg'], label=command)
    for ax, title, xlabel, ylabel in zip(axes.flat, ['Measured ground trajectory', 'Measured yaw', 'Measured thorax height', 'Measured body tilt'], ['x (mm)', 'time (s)', 'time (s)', 'time (s)'], ['y (mm)', 'degrees', 'mm', 'degrees']):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel); ax.legend(); ax.grid(alpha=.2)
    fig.suptitle('Engineered FlyGym hybrid bridge — position + adhesion, NOT canonical muscle control')
    fig.tight_layout(); fig.savefig(args.output / 'body_baseline.png', dpi=140); plt.close(fig)
    if frames:
        import imageio.v2 as imageio
        imageio.imwrite(args.output / 'body_baseline_render.png', frames[20])
        imageio.mimsave(args.output / 'body_baseline.mp4', frames, fps=20)
    return 0 if report['all_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
