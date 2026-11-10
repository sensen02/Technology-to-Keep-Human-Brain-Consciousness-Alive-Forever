"""Open-loop capability probe: can the repaired interface raise one leg's pads at all?

This is a diagnostic, not a controller and not a learning result. It asks whether a constant
joint-rate command on a single leg, at the actuator's own rate limit, can produce the pad
height that the single-leg task requires (0.015 native) under the declared adhesion law.
If it cannot, the task is not a learning problem.
"""
from pathlib import Path
import json, sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repair_core as rc

W = rc.WEIGHT
NAMES = ('Coxa_yaw', 'Coxa_pitch', 'Coxa_roll', 'Trochanter_yaw', 'Trochanter_pitch',
         'Trochanter_roll', 'Tibia_pitch')
# candidate rate directions, in units of the actuator rate limit, per leg (7 joints)
CANDIDATES = {
    'tibia_up': {'Tibia_pitch': .5},
    'tibia_down': {'Tibia_pitch': -.5},
    'troch_pitch_up': {'Trochanter_pitch': .5},
    'troch_pitch_down': {'Trochanter_pitch': -.5},
    'coxa_pitch_up': {'Coxa_pitch': .5},
    'coxa_pitch_down': {'Coxa_pitch': -.5},
    'troch_roll_up': {'Trochanter_roll': .5},
    'troch_roll_down': {'Trochanter_roll': -.5},
    'coxa_yaw_up': {'Coxa_yaw': .5},
    'flex_combo': {'Tibia_pitch': -.4, 'Trochanter_pitch': -.4, 'Coxa_pitch': .4},
    'extend_combo': {'Tibia_pitch': .4, 'Trochanter_pitch': .4, 'Coxa_pitch': -.4},
}


def leg_joint_index(leg, joint):
    return NAMES.index(joint), leg


def action_for(leg, spec):
    """Replicate one leg's rate pattern onto the 42-dim model order, by joint NAME."""
    a = np.zeros(rc.NJ)
    L = rc.LEG_ORDER[leg]
    for joint, val in spec.items():
        a[JIDX['joint_' + L + joint]] = val
    return a


def run(leg, spec, ticks=300):
    core = rc.Core()
    a = action_for(leg, spec)
    rest = float(core.measure()['padz'][leg])
    max_lift = rest
    min_load = 1e9
    sup_min = 1e9
    com_min = 1e9
    nonfoot = 0.
    for t in range(ticks):
        m = core.tick(a)
        max_lift = max(max_lift, float(m['padz'][leg]))
        min_load = min(min_load, float(m['loads'][leg]))
        sup_min = min(sup_min, float(m['loads'].sum() / W))
        com_min = min(com_min, m['com'])
        nonfoot = max(nonfoot, float(m['nonfoot']))
    return dict(rest_padz=rest, max_padz=max_lift, lift=max_lift - rest,
                min_load_over_body_weight=float(min_load / W), support_min=float(sup_min),
                com_min=float(com_min), nonfoot_max=float(nonfoot))


def main():
    global JIDX
    core = rc.Core()
    jn = [core.plant.joint(int(j)).name for j in core.actuator.joint_ids]
    JIDX = {n: i for i, n in enumerate(jn)}
    order_check = {}
    for leg, L in enumerate(rc.LEG_ORDER):
        block = jn[leg * 7:(leg + 1) * 7]
        assert all(n.startswith('joint_' + L) for n in block), (L, block)
        order_check[L] = [n.split(L, 1)[1] for n in block]
    # the right legs do not use the same intra-leg joint order as the left legs
    intra_leg_orders = sorted({tuple(v) for v in order_check.values()})
    out = {'label': 'open-loop constant joint-rate probe, diagnostic only, no learning',
           'actuator_rate_limit_rad_s': rc.MAX_RATE, 'candidates': list(CANDIDATES),
           'joint_block_order_verified': True,
           'intra_leg_joint_orders': [list(t) for t in intra_leg_orders],
           'note_intra_leg_order': 'right legs order the coxa joints differently from the left '
                                   'legs, so actions must be indexed by joint name, not by offset',
           'per_leg': {}}
    best = {}
    for leg in range(rc.NL):
        rows = {}
        for name, spec in CANDIDATES.items():
            r = run(leg, spec)
            r['passes_lift'] = bool(r['max_padz'] >= 0.015)
            rows[name] = r
        out['per_leg'][rc.LEG_ORDER[leg]] = rows
        top = max(rows.items(), key=lambda kv: kv[1]['max_padz'])
        best[rc.LEG_ORDER[leg]] = dict(candidate=top[0], max_padz=top[1]['max_padz'],
                                       passes_lift=top[1]['passes_lift'])
        print(json.dumps({rc.LEG_ORDER[leg]: best[rc.LEG_ORDER[leg]]}), flush=True)
    out['best_per_leg'] = best
    out['any_leg_can_reach_0015'] = bool(any(v['passes_lift'] for v in best.values()))
    out['n_legs_reaching_0015'] = int(sum(v['passes_lift'] for v in best.values()))
    (HERE / 'LIFT_PROBE.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: out[k] for k in ('any_leg_can_reach_0015', 'n_legs_reaching_0015')}))


if __name__ == '__main__':
    main()
