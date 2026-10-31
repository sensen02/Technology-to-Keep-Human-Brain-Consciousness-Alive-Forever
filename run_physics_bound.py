"""P3 round 46: physics bound — can ANY leg configuration support body weight with the muscle cone?

Rounds 35-45 established that training a control layer cannot fix a force deficit, so before spending
hours of training the question has to be settled physically: over leg configurations AND body height,
what is the best achievable non-negative-cone support?

For every configuration:
  * body height offset dz, body pitch/roll, and a shared leg-joint variation
    (coxa pitch, trochanter pitch, tibia pitch) is applied to all six legs;
  * the corrected cone LP is solved at that configuration with the real contacts;
  * the minimax slack (deficit, in body-weight units) and the achievable vertical support are recorded.

If some configuration reaches slack ~0 the muscle set CAN hold the body and control training has a
reachable target; if none does, the deficit is physical for this model and no controller can sustain
support.
"""
from pathlib import Path
import os, json, sys, time
os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np
import mujoco
from scipy.optimize import linprog
sys.path.insert(0, '/run/media/sensen/Data2/cell_wound_prototype')
from flygym.compose.fly.musculoskeletal import _load_mjcf

R46 = Path('/home/sensen/Desktop/engine修复验收/s5-physics-bound/20261029T170000Z')
R25 = Path('/home/sensen/Desktop/engine修复验收/s5-synaptic-output-training/20261028T150000Z')
BASE = R25 / 'MODELBUILD' / 'MODEL.xml'
HANDOVER = R25 / 'TRIAL' / 'HANDOVER_BEFORE.npz'
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
DZ = [-0.30, -0.15, 0.0, 0.15]
TIB = [-0.5, -0.25, 0.0, 0.25, 0.5]
TRO = [-0.25, 0.0, 0.25]
COX = [-0.2, 0.0, 0.2]


def jd(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o).__name__)


def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=1, default=jd, ensure_ascii=False) + '\n')


def solve(m, d, y0, B, contacts, mu):
    nv, nu, nc = m.nv, m.nu, len(contacts)
    if nc == 0:
        return None
    nvar = nu + 3 * nc + 1
    A = np.zeros((nv, 3 * nc))
    for j, c in enumerate(contacts):
        jacp = np.zeros((3, nv)); jacr = np.zeros((3, nv))
        mujoco.mj_jac(m, d, jacp, jacr, np.array(c['pos']), c['body'])
        A[:, 3 * j:3 * j + 3] = (np.array(c['frame']) @ jacp).T
    rows, rhs = [], []
    for k in range(nv):
        r = np.zeros(nvar); r[:nu] = B[k, :]; r[nu:nu + 3 * nc] = A[k, :]; r[-1] = -1.0
        rows.append(r); rhs.append(y0[k])
        r2 = -r.copy(); r2[-1] = -1.0; rows.append(r2); rhs.append(-y0[k])
    for j in range(nc):
        for t in (1, 2):
            for sgn in (1.0, -1.0):
                r = np.zeros(nvar); r[nu + 3 * j + t] = sgn; r[nu + 3 * j + 0] = -mu
                rows.append(r); rhs.append(0.0)
        r = np.zeros(nvar); r[nu + 3 * j + 0] = -1.0; rows.append(r); rhs.append(0.0)
    obj = np.zeros(nvar); obj[-1] = 1.0
    bounds = [(0.0, 1.0)] * nu + [(None, None)] * (3 * nc) + [(0.0, None)]
    s = linprog(obj, A_ub=np.array(rows), b_ub=np.array(rhs), bounds=bounds, method='highs')
    if not s.success:
        return None
    x = s.x
    return dict(slack=float(s.fun), support=float(sum(x[nu + 3 * j + 0] for j in range(nc))))


def main():
    started = time.time()
    R46.mkdir(parents=True, exist_ok=True)
    m = _load_mjcf(str(BASE)).compile()
    z = dict(np.load(HANDOVER))
    q0 = z['qpos'].copy()
    weight = float(json.load(open(R25 / 'MODELBUILD' / 'HARNESS.json'))['weight'])
    mu = float(np.max(m.geom_friction[:, 0]))
    jidx = {}
    for leg in LEGS:
        for nm in ('Coxa_pitch', 'Trochanter_pitch', 'Tibia_pitch'):
            j = m.joint('joint_%s%s' % (leg, nm.replace('_pitch', '_pitch')))
            jidx[(leg, nm)] = int(m.jnt_qposadr[j.id])
    rows_out = []
    # cache the per-actuator columns is impossible across poses (they change), so B is rebuilt per pose
    for dz in DZ:
        for tib in TIB:
            for tro in TRO:
                for cox in COX:
                    q = q0.copy()
                    q[2] += dz
                    for leg in LEGS:
                        q[jidx[(leg, 'Coxa_pitch')]] += cox
                        q[jidx[(leg, 'Trochanter_pitch')]] += tro
                        q[jidx[(leg, 'Tibia_pitch')]] += tib
                    d = mujoco.MjData(m)
                    d.qpos[:] = q
                    d.qvel[:] = 0
                    d.ctrl[:] = 0
                    d.act[:] = 0.0
                    mujoco.mj_forward(m, d)
                    contacts = []
                    for i in range(d.ncon):
                        c = d.contact[i]
                        b1 = m.body(m.geom_bodyid[c.geom1]).name
                        b2 = m.body(m.geom_bodyid[c.geom2]).name
                        if b1 == 'world' or b2 == 'world':
                            leg = next((L for L in LEGS if (b1 if b1 != 'world' else b2).startswith(L)), None)
                            if leg is None:
                                continue
                            gid = c.geom2 if b1 == 'world' else c.geom1
                            contacts.append(dict(pos=c.pos.tolist(), frame=c.frame.reshape(3, 3).tolist(),
                                                 body=int(m.geom_bodyid[gid])))
                    n_legs = len({c['body'] for c in contacts})
                    if len(contacts) == 0:
                        rows_out.append(dict(dz=dz, tibia=tib, trochanter=tro, coxa=cox, contacts=0,
                                             feasible=False))
                        continue
                    a0 = d.qfrc_actuator.copy()
                    y0 = (d.qfrc_bias - d.qfrc_passive - a0).copy()
                    B = np.zeros((m.nv, m.nu))
                    for i in range(m.nu):
                        d.act[:] = 0.0
                        mujoco.mj_forward(m, d)
                        d.act[i] = 1.0
                        mujoco.mj_forward(m, d)
                        B[:, i] = d.qfrc_actuator - a0
                    d.act[:] = 0.0
                    mujoco.mj_forward(m, d)
                    res = solve(m, d, y0, B, contacts, mu)
                    row = dict(dz=dz, tibia=tib, trochanter=tro, coxa=cox, contacts=len(contacts),
                               bodies=len({c['body'] for c in contacts}))
                    if res:
                        row.update(slack=res['slack'], slack_over_weight=res['slack'] / weight,
                                   support=res['support'], support_over_weight=res['support'] / weight,
                                   feasible=bool(res['slack'] <= 1e-6))
                    rows_out.append(row)
                    if len(rows_out) % 20 == 0:
                        print('poses done', len(rows_out), flush=True)
    rows_out.sort(key=lambda r: r.get('slack', 1e9) if r.get('slack') is not None else 1e9)
    best = rows_out[0] if rows_out else None
    out = dict(weight=weight, mu=mu, n_poses=len(rows_out),
               n_feasible=int(sum(1 for r in rows_out if r.get('feasible'))),
               best=best, domain=dict(dz=DZ, tibia=TIB, trochanter=TRO, coxa=COX),
               verdict=('SOME_CONFIGURATION_SUPPORTS_BODY_WEIGHT' if best and best.get('feasible')
                        else 'NO_CONFIGURATION_SUPPORTS_BODY_WEIGHT_IN_THE_DOMAIN'),
               rows=rows_out, runtime_s=time.time() - started,
               claim_class='capability diagnostic; OUR OWN DERIVATION')
    dump(R46 / 'PHYSICS_BOUND.json', out)
    print('ROUND46_DONE', json.dumps(dict(n_poses=len(rows_out), n_feasible=out['n_feasible'],
                                          verdict=out['verdict'],
                                          best_slack=round(best['slack_over_weight'], 4) if best and 'slack_over_weight' in best else None,
                                          best_domain={k: best[k] for k in ('dz', 'tibia', 'trochanter', 'coxa')} if best else None,
                                          runtime_s=round(time.time() - started, 1))), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
