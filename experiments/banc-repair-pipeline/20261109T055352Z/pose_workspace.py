"""Why three legs cannot raise a foot from the solved stand pose.

Offline geometry only: compare the solved triad stand pose against the model's own joint
ranges, and print which joints sit on their safety margin. The lift probe
(LIFT_PROBE.json) shows the same legs are the ones whose pad height does not respond to any
commanded joint motion.
"""
from pathlib import Path
import json, sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repair_core as rc


def main():
    core = rc.Core()
    p = core.plant
    jids = core.actuator.joint_ids
    names = [p.joint(int(j)).name for j in jids]
    rng = np.asarray(p.jnt_range[jids], float)
    q = core.pose_q
    margin = np.minimum(q - rng[:, 0], rng[:, 1] - q)
    probe = json.loads((HERE / 'LIFT_PROBE.json').read_text())
    per_leg = {}
    for leg, L in enumerate(rc.LEG_ORDER):
        idx = [i for i, n in enumerate(names) if n.startswith('joint_' + L)]
        m = {names[i].replace('joint_' + L, ''): float(margin[i]) for i in idx}
        roll = m['Coxa_roll']
        can = probe['per_leg'][L]
        best = max(can.items(), key=lambda kv: kv[1]['max_padz'])
        per_leg[L] = dict(coxa_roll_margin_rad=roll,
                          coxa_roll_at_margin=bool(roll <= 0.0105),
                          best_probe_candidate=best[0],
                          best_probe_max_padz=float(best[1]['max_padz']),
                          probe_passes_lift=bool(best[1]['passes_lift']),
                          joint_margins_rad=m)
    at_limit = [L for L, v in per_leg.items() if v['coxa_roll_at_margin']]
    can_lift = [L for L, v in per_leg.items() if v['probe_passes_lift']]
    out = dict(
        label='offline pose-versus-workspace comparison; no learning, no new physics',
        pose_file=str(rc.POSE_FILE),
        margin_threshold_rad=0.0105,
        legs_with_coxa_roll_on_margin=at_limit,
        legs_that_can_raise_a_pad=can_lift,
        legs_that_cannot=[L for L in per_leg if L not in can_lift],
        cooccurrence='every leg that can raise a pad carries its probe foot load to zero and '
                     'shows a pad height that changes with the command; the three legs that '
                     'cannot show a pad height identical under all eleven candidates, i.e. the '
                     'height is set by ground penetration rather than by the commanded joints',
        not_established='the coxa-roll margin does not by itself explain the pattern: RF sits on '
                        'the margin and still raises a pad by 0.217, while LH has 0.093 rad of '
                        'margin and raises nothing. The cause of the hind-leg failure is NOT '
                        'established by this probe and needs its own test, for example re-solving '
                        'the pose with a roll-margin constraint and re-running the same probe',
        conclusion='three of six legs cannot raise a foot from the current solved stand pose, so '
                   'a single-leg stepping task cannot pass for them regardless of training; that '
                   'must be fixed or declared before more stepping training is meaningful',
        per_leg=per_leg)
    (HERE / 'POSE_WORKSPACE.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: out[k] for k in ('legs_with_coxa_roll_on_margin', 'legs_that_can_raise_a_pad',
                                          'legs_that_cannot')}, indent=2))


if __name__ == '__main__':
    main()
