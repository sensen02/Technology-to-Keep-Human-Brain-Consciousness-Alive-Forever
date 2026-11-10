"""Why does the right hind leg never advance, even though the teacher can advance it?

Records, tick by tick for the right hind leg: pad position, every joint angle with its distance
to the joint limits, the commanded action, whether the pad is in contact, and how much forward
foot displacement the current pose could still deliver within the remaining joint travel.
Run twice: once for the declared three phase teacher, once for the trained policy that fails.
"""
from pathlib import Path
import json, sys
import numpy as np
import mujoco as mj

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repair_core as rc
import train_single_leg as T

LEG = 'RH'


def joint_rows(core, leg):
    idx = T.leg_joint_action_index(core, leg)
    rows = []
    for a in idx:
        jid = int(core.actuator.joint_ids[a])
        j = core.plant.joint(jid)
        lo, hi = (float(j.range[0]), float(j.range[1])) if j.limited.any() else (-np.inf, np.inf)
        q = float(core.d.qpos[int(core.actuator.qpos_addresses[a])])
        rows.append(dict(name=j.name, q=q, lo=lo, hi=hi,
                         margin_lo=q - lo, margin_hi=hi - q,
                         at_limit=bool(j.limited.any() and (q - lo < 1e-3 or hi - q < 1e-3))))
    return rows


def pad_contact(core, leg):
    """Normal force on this leg's pad spheres from site contacts, in newtons (native units)."""
    total = 0.
    pads = set(int(g) for g in core.pads[LEG])
    for c in range(int(core.d.ncon)):
        con = core.d.contact[c]
        if int(con.geom1) in pads or int(con.geom2) in pads:
            f = np.zeros(6)
            mj.mj_contactForce(core.plant, core.d, c, f)
            total += abs(float(f[0]))
    return total


def jog_lever(core, leg):
    """Forward foot displacement still available if every joint used its remaining travel."""
    J = T.foot_horizontal_jacobian(core, leg)
    reach = 0.
    for row in joint_rows(core, leg):
        a = [i for i in T.leg_joint_action_index(core, leg)
             if core.plant.joint(int(core.actuator.joint_ids[i])).name == row['name']][0]
        dof = int(core.actuator.dof_addresses[a])
        j = float(J[dof])
        travel = min(row['margin_hi'], row['margin_lo'])
        reach += abs(j) * max(0., travel)
    return reach


def rollout(kind, ticks=None):
    ticks = ticks or T.T_EP
    net = None
    if kind == 'policy':
        Z = np.load(HERE / 'MODEL_singleleg_demo_soloRH.npz')
        net = T.Net(np.random.default_rng(11))
        for k in ('w0', 'b0', 'w1', 'b1', 'w2', 'b2'):
            net.p[k] = Z['actor_' + k].astype(np.float64)
        net.p['logstd'] = Z['logstd'].astype(np.float64)
    core = rc.Core()
    core.reset()
    leg = rc.LEG_ORDER.index(LEG)
    prompt = T.onehot(leg)
    x0 = float(core.measure()['foot'][leg][0])
    z0 = float(core.measure()['padz'][leg])
    rows = []
    for k in range(ticks):
        a = (T.teacher_action(core, leg, k) if kind == 'teacher'
             else np.tanh(net.forward(core.obs(prompt)[None])[2][0]))
        m = core.tick(a)
        jr = joint_rows(core, leg)
        rows.append(dict(
            k=k, padx=float(m['foot'][leg][0]) - x0, padz=float(m['padz'][leg]) - z0,
            load=float(m['loads'][leg] / rc.WEIGHT),
            contact=pad_contact(core, leg), reachable_forward=jog_lever(core, leg),
            action=[round(float(v), 3) for v in a],
            joints={r['name'].replace('joint_' + LEG, ''): round(r['q'], 3) for r in jr},
            at_limits=[r['name'].replace('joint_' + LEG, '') for r in jr if r['at_limit']]))
    return rows


def summary(kind, rows):
    span = [min(r['padx'] for r in rows), max(r['padx'] for r in rows)]
    zs = [r['padz'] for r in rows]
    lim_counts = {}
    for r in rows:
        for n in r['at_limits']:
            lim_counts[n] = lim_counts.get(n, 0) + 1
    in_air = [r for r in rows if r['padz'] > 0.01]
    return dict(kind=kind, padx_min=span[0], padx_max=span[1],
                padz_min=min(zs), padz_max=max(zs),
                ticks_in_air=len(in_air),
                ticks_at_joint_limit=lim_counts,
                reachable_forward_max=max(r['reachable_forward'] for r in rows),
                reachable_forward_when_air_max=(max((r['reachable_forward'] for r in in_air),
                                                    default=0.)),
                contact_when_air_max=(max((r['contact'] for r in in_air), default=0.)),
                load_max=float(max(r['load'] for r in rows)))


def main():
    out = dict(label='right hind leg forward-advance diagnosis; no learning, no acceptance change',
               leg=LEG, teacher={}, policy={})
    for kind in ('teacher', 'policy'):
        rows = rollout(kind)
        out[kind]['summary'] = summary(kind, rows)
        out[kind]['trace'] = rows[::15] + [rows[-1]]
        print(kind, json.dumps(out[kind]['summary'], default=str), flush=True)
        air = [r for r in rows if r['padz'] > 0.01]
        for r in (air[::20] + air[-1:]) if air else []:
            print('   k%3d padx %+.4f padz %+.4f load %.3f contact %.1f reach %.4f acts %s limits %s'
                  % (r['k'], r['padx'], r['padz'], r['load'], r['contact'],
                     r['reachable_forward'], r['action'], r['at_limits']), flush=True)
    (HERE / 'RH_ADVANCE_DIAG.json').write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
