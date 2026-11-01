"""Read-only contact-wrench decomposition of saved equilibrium attempts; no model, forward or step."""
from pathlib import Path
import json
import numpy as np
R6 = Path('/home/sensen/Desktop/engine修复验收/s5-native-equilibrium-search/20261101T193630Z')
R7 = Path('/home/sensen/Desktop/engine修复验收/s5-equilibrium-derivative-continuation/20261101T200110Z')
OUT = Path('/home/sensen/Desktop/engine修复验收/s5-equilibrium-wrench-diagnosis/20261101T200110Z')
BODY = Path('/home/sensen/Desktop/engine修复验收/s5-muscle-native-recording/20261101T191552Z/RUN/BODY.npz')


def analyse(state, weight, com_seed, label):
    foot = state['foot_force']
    net_foot = foot.sum(0)
    item = dict(
        label=label,
        weight_native=float(weight),
        net_foot_force=net_foot.tolist(),
        per_leg_vertical_force=foot[:, 2].tolist(),
        zero_or_negative_leg_count=int(np.sum(foot[:, 2] <= 0)),
        vertical_over_weight=float(net_foot[2] / weight),
        horizontal_over_vertical=float(np.linalg.norm(net_foot[:2]) / net_foot[2]),
        root_qacc=state['qacc'][:6].tolist(),
        max_abs_qacc=float(abs(state['qacc']).max()),
        root_position=state['qpos'][:3].tolist(),
        act_dot_max_abs=float(abs(state['act_dot']).max()),
        moment_source='contact_rows' if 'contact_rows' in state.files else 'unavailable in this package',
    )
    if 'contact_rows' in state.files:
        rows = state['contact_rows']
        pos, force = rows[:, 3:6], rows[:, 6:9]
        item['n_contact_rows'] = int(len(rows))
        item['net_contact_force_from_rows'] = force.sum(0).tolist()
        item['net_contact_moment_about_seed_com'] = np.cross(pos - com_seed, force).sum(0).tolist()
        item['net_contact_moment_about_root'] = np.cross(pos - state['qpos'][:3], force).sum(0).tolist()
        item['moment_norm_over_weight_seed_com'] = float(np.linalg.norm(np.cross(pos - com_seed, force).sum(0)) / weight)
        item['moment_norm_over_weight_root'] = float(np.linalg.norm(np.cross(pos - state['qpos'][:3], force).sum(0)) / weight)
        item['per_contact_vertical_force'] = force[:, 2].tolist()
        item['contact_distances'] = rows[:, 2].tolist()
        item['zero_vertical_contact_count'] = int(np.sum(force[:, 2] <= 0))
    else:
        item['n_contact_rows'] = None
        item['net_contact_moment_about_seed_com'] = None
        item['net_contact_moment_about_root'] = None
        item['moment_norm_over_weight_seed_com'] = None
        item['moment_norm_over_weight_root'] = None
        item['per_contact_vertical_force'] = None
        item['contact_distances'] = None
        item['zero_vertical_contact_count'] = None
    return item


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    variables = json.loads((R6 / 'VARIABLES.json').read_text())
    weight = float(variables['weight_native'])
    body = np.load(BODY)
    com_seed = body['com'][300]
    results = []
    for directory, tag in [(R6, 'round6_search60'), (R7, 'round7_continuation150')]:
        with np.load(directory / 'FINAL_STATE.npz') as state:
            results.append(analyse(state, weight, com_seed, tag))
    summary = dict(
        scope='Pure arithmetic on saved arrays; no model load, forward, optimization or integration',
        source_directories=[str(R6), str(R7)],
        com_caveat='Contact points are absolute world positions and the seed COM comes from the unmodified t=0.15 body frame; a candidate pose changes the true COM, so the moment columns about the seed COM are approximations pending a fresh forward. The round7 package stores no contact positions, so its moment terms are reported as unavailable rather than reconstructed.',
        weight_unit='native model force unit; absolute physiology NOT_ESTABLISHED',
        results=results,
        interpretation=[
            'Both best candidates carry a horizontal resultant near 20 percent of weight while total vertical support is close to weight.',
            'A static equilibrium requires the net contact force to equal weight with zero horizontal resultant and zero moment; neither candidate satisfies this.',
            'This expresses the residual in force units rather than a normalized SSE and shows the failure is not a small numerical overshoot.',
            'Nothing here identifies which parameter is wrong, and no candidate was integrated.',
        ],
    )
    (OUT / 'WRENCH.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({r['label']: {k: r[k] for k in ['vertical_over_weight', 'horizontal_over_vertical',
                                                     'moment_norm_over_weight_seed_com', 'moment_norm_over_weight_root',
                                                     'zero_or_negative_leg_count', 'max_abs_qacc']} for r in results}, indent=2))


if __name__ == '__main__':
    main()
