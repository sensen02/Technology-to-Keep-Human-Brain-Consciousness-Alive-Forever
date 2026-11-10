"""Does holding a leg clear leave the body supportable at all?

The solver can place five pads on the floor with one leg clear and the COM where the criteria want
it, but a kinematic pose is not a stance. This holds each solved pose under the real actuator and
compares it with the nominal six leg pose held the same way, so the support loss is attributable to
the raised leg rather than to the settling of the test itself.
"""
from pathlib import Path
import json, os, sys, subprocess
import numpy as np
import mujoco as mj

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repair_core as rc
import train_single_leg as T

PY = sys.executable


def solve(raised, clear, com_target):
    env = dict(os.environ, RAISED=raised, CLEAR=str(clear), COM_TARGET=str(com_target),
               HOLD_S='0.30', PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='2')
    subprocess.run([PY, str(HERE / 'solve_five_leg_pose.py')], cwd=str(HERE), env=env,
                   check=True, capture_output=True)
    tag = '%s_%s' % (raised, str(com_target).replace('.', ''))
    return json.loads((HERE / ('STAND_FIVE_LEG_%s.json' % tag)).read_text())


def hold(q, rz, seconds=.30):
    c = rc.Core(); c.reset()
    if q is not None:
        c.d.qpos[c.a_qpos] = q; c.d.qpos[2] = rz
        mj.mj_forward(c.plant, c.d)
    c.actuator.reset(c.d)
    c._padz_prev = np.asarray([float(c.measure()['padz'][l]) for l in range(rc.NL)])
    c.buf[:] = c.actuator.target_rad[None]
    z0 = float(c.d.qpos[2])
    sup, com, nf = [], [], []
    for k in range(int(seconds / rc.PDT)):
        m = c.tick_rate(np.zeros(rc.NJ))
        sup.append(float(m['loads'].sum() / rc.WEIGHT)); com.append(float(m['com']))
        nf.append(float(m['nonfoot']))
    return dict(root_dz=float(c.d.qpos[2] - z0), support_min=min(sup), support_mean=float(np.mean(sup)),
                com_min=min(com), nonfoot_max=max(nf))


def main():
    out = dict(label='can five legs support the body with one leg held clear?',
               reference={}, rows=[])
    out['reference']['nominal_six_leg'] = hold(None, None)
    out['reference']['nominal_six_leg']['note'] = ('the same hold applied to the untouched stand pose; '
                                                   'the sink here is the incremental actuator settling, '
                                                   'not a pose defect')
    print('nominal six legs:', json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                           for k, v in out['reference']['nominal_six_leg'].items()}), flush=True)
    for raised in ('RH', 'LH', 'RM'):
        for clear in (0.01, 0.04):
            r = solve(raised, clear, 1.45)
            h = hold(np.asarray(r['q_rad']), r['root_z'])
            row = dict(raised=raised, clearance=clear, solved_pad=r['pad_lowest'][raised],
                       com_kinematic=r['com'], nonfoot_low=r['nonfoot_low'],
                       supportable=bool(h['support_min'] >= .90 and h['com_min'] >= 1.00),
                       **h)
            out['rows'].append(row)
            print('%-2s clear %.2f -> kinematic pad %.4f com %.3f | hold: support_min %.3f com_min %.3f '
                  'nonfoot %.2f root_dz %+.3f  %s' % (
                      raised, clear, r['pad_lowest'][raised], r['com'], h['support_min'],
                      h['com_min'], h['nonfoot_max'], h['root_dz'],
                      'SUPPORTABLE' if row['supportable'] else 'NOT SUPPORTABLE'), flush=True)
    (HERE / 'FIVE_LEG_HOLD.json').write_text(json.dumps(out, indent=2) + '\n')
    n = sum(r['supportable'] for r in out['rows'])
    print('supportable configurations:', n, '/', len(out['rows']))


if __name__ == '__main__':
    main()
