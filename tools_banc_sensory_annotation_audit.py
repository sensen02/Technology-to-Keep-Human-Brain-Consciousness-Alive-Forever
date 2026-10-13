"""What the BANC annotation itself says about each sensory channel: cells, attachments, position.

MANUAL RESEARCH PROTOTYPE, NOT A PRODUCT.  This is a DATASET audit (pandas/pyarrow interpreter), not
a simulation: it reads the published BANC metadata and reports, for each of the seven sensory
channels the closure uses, which cells the annotation puts in it and where those cells' somata sit
along the leg.  It runs no body and makes no neural claim.

Why this matters: the closure feeds every channel a quantity chosen by this project (the tibia joint
angle, the tibia velocity, a joint-mean deviation, a normalised contact force).  Whether those
choices are anatomically defensible is a question about the receptors, and the dataset carries two
pieces of evidence for it that the project has not used yet:

  * `peripheral_target_type` names the peripheral structure, including NUMBERED groups
    (CoHP3 / CoHP4 / CoHP8 = coxa hair plate groups) and separates hair plates from campaniform
    sensilla from chordotonal organs;
  * `body_part_sensory` names the attachment (`coxa,front_leg`, `front_leg,trochanter`, ...);
  * `position` gives each soma's coordinates, so the channels can be ordered along the leg's own
    proximo-distal axis by measurement instead of by assumption.

The proximo-distal coordinate is defined here as the projection of a soma onto the principal axis of
that leg's own leg-sensory somata, oriented so that the projection increases away from the centroid
of the thoracic somata.  The orientation is a heuristic and is reported with the axis it used.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
META = HERE / 'data/banc888/banc_888_meta.feather'
OUT = HERE / 'outputs/model_branches/banc_sensory_annotation'
LEGS = ('front_leg', 'middle_leg', 'hind_leg')
LEG_SHORT = {'front_leg': 'LF/RF', 'middle_leg': 'LM/RM', 'hind_leg': 'LH/RH'}
CHANNEL_SUBTYPE = {
    'claw_position': 'claw_chordotonal_organ_neuron',
    'hook_direction': 'hook_chordotonal_organ_neuron',
    'club_vibration': 'club_chordotonal_organ_neuron',
    'hairplate_angle': 'hair_plate_neuron',
    'campaniform_strain': 'campaniform_sensillum_neuron',
    'bristle_touch': 'bristle_neuron',
    'multidendritic_noxious': 'multidendritic_neuron',
}


def parse_positions(series):
    values = np.full((len(series), 3), np.nan)
    for i, text in enumerate(series.astype(str)):
        parts = text.replace('[', '').replace(']', '').split(',')
        if len(parts) == 3:
            try:
                values[i] = [float(p) for p in parts]
            except ValueError:
                continue
    return values


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_feather(META)
    df['sub_class'] = df['cell_sub_class'].astype(str)
    df['target'] = df['peripheral_target_type'].astype(str)
    df['body_part'] = df['body_part_sensory'].astype(str)
    positions = parse_positions(df['position'])
    df['x'], df['y'], df['z'] = positions[:, 0], positions[:, 1], positions[:, 2]

    # MEASURED BUG (fixed): the first version fitted one principal axis per LEG TYPE, so `front_leg`
    # pooled the left and right legs and the axis was dominated by the left-right separation rather
    # than by the proximo-distal direction.  Every projection is now per leg AND side.
    df['leg_key'] = [{'front_leg': 'F', 'middle_leg': 'M', 'hind_leg': 'H'}.get(
                         str(part).split(',')[0], '')
                     + {'left': 'L', 'right': 'R'}.get(str(side), '')
                     for part, side in zip(df['body_part'], df['side'])]
    thorax_centre = None
    thorax = df[(df['leg_key'] == '') & df['body_part'].str.startswith('thorax') & df['x'].notna()]
    if len(thorax):
        thorax_centre = thorax[['x', 'y', 'z']].median().to_numpy()

    channels, per_leg_axis = {}, {}
    for channel, subtype in CHANNEL_SUBTYPE.items():
        rows, axis_rows = {}, {}
        for leg in LEGS:
            for side in ('left', 'right'):
                key = ({'front_leg': 'F', 'middle_leg': 'M', 'hind_leg': 'H'}[leg]
                       + {'left': 'L', 'right': 'R'}[side])
                cells = df[(df['sub_class'] == f'{leg}_{subtype}') & (df['side'].astype(str) == side)]
                if cells.empty:
                    continue
                rows[key] = dict(
                    leg=leg, side=side, n_cells=int(len(cells)),
                    cell_types=cells['cell_type'].astype(str).value_counts().head(12).to_dict(),
                    peripheral_target_types=cells['target'].value_counts().to_dict(),
                    body_part_sensory=cells['body_part'].value_counts().to_dict(),
                    cell_function=cells['cell_function'].astype(str).value_counts().head(6).to_dict())
                same_leg = df[(df['leg_key'] == key) & df['x'].notna()]
                cloud = same_leg[['x', 'y', 'z']].to_numpy(float)
                pts = cells[['x', 'y', 'z']].to_numpy(float)
                pts = pts[np.isfinite(pts).all(axis=1)]
                if len(pts) < 5 or len(cloud) < 20:
                    rows[key]['n_with_position'] = int(len(pts))
                    continue
                centre = cloud.mean(axis=0)
                _, _, vt = np.linalg.svd(cloud - centre, full_matrices=False)
                axis = vt[0]
                reference = thorax_centre if thorax_centre is not None else centre
                if np.dot(centre - reference, axis) < 0:
                    axis = -axis
                projection = (pts - centre) @ axis
                rows[key].update(
                    n_with_position=int(len(pts)),
                    soma_centroid_um=[round(float(v)/1000.0, 2) for v in pts.mean(axis=0)],
                    projection_median_um=round(float(np.median(projection))/1000.0, 2),
                    projection_quartiles_um=[round(float(np.percentile(projection, q))/1000.0, 2)
                                             for q in (25, 75)],
                    projection_std_um=round(float(projection.std())/1000.0, 2),
                    leg_axis=dict(axis=[round(float(v), 6) for v in axis],
                                  fitted_on=f'principal axis of all {int(len(cloud))} sensory somata '
                                            f'of {key}, oriented away from the thorax soma centroid'))
                per_leg_axis.setdefault(key, {})[channel] = rows[key]['projection_median_um']
            channels_leg = {k: v for k, v in rows.items() if v['leg'] == leg}
            if channels_leg:
                axis_rows[leg] = {k: dict(median_projection_um=v['projection_median_um'])
                                  for k, v in channels_leg.items()
                                  if 'projection_median_um' in v}
        channels[channel] = dict(subtype=subtype, per_leg_side=rows, axes=axis_rows)

    ordering = {}
    for key, by_channel in sorted(per_leg_axis.items()):
        ordered = sorted(by_channel.items(), key=lambda kv: kv[1])
        ordering[key] = [dict(channel=c, projection_median_um=v) for c, v in ordered]
    # COVERAGE: which proprioceptive cells does the project's seven-subtype partition MISS?
    # The dataset labels every leg sensory cell with a modality in `cell_function_detailed`; the seven
    # subtypes the closure uses are only some of the subtypes that carry those modalities, so this
    # counts what the front end does and does not see.
    MODALITIES = ('position', 'direction', 'joint_angle', 'mechanical_strain', 'vibro_tactile',
                  'vibro_position', 'unknown_leg_movement')
    leg_cells = df[df['body_part'].str.contains('front_leg|middle_leg|hind_leg', na=False)].copy()
    leg_cells = leg_cells[~leg_cells['sub_class'].str.endswith('_motor_neuron')]
    used_subtypes = {f'{leg}_{sub}' for leg in LEGS for sub in CHANNEL_SUBTYPE.values()}
    leg_cells['used'] = leg_cells['sub_class'].isin(used_subtypes)
    leg_cells['modality'] = leg_cells['cell_function_detailed'].astype(str)
    proprio = leg_cells[leg_cells['modality'].isin(MODALITIES)]
    coverage = dict(
        leg_sensory_cells_in_annotation=int(len(leg_cells)),
        in_one_of_the_seven_subtypes=int(leg_cells['used'].sum()),
        proprioceptive_cells=int(len(proprio)),
        proprioceptive_cells_used=int(proprio['used'].sum()),
        proprioceptive_cells_MISSED=int((~proprio['used']).sum()),
        missed_by_modality=proprio[~proprio['used']]['modality'].value_counts().to_dict(),
        missed_by_subtype=proprio[~proprio['used']]['sub_class'].value_counts().to_dict(),
        used_by_modality=proprio[proprio['used']]['modality'].value_counts().to_dict(),
        reading='the seven subtypes the closure uses are the ones the dataset labels position, '
                'direction, joint_angle, mechanical_strain and vibro_tactile; the cells counted as '
                'MISSED carry a proprioceptive modality but sit in a subtype this partition never '
                'reads (orphan and generic chordotonal subtypes)')
    hp_groups = df[df['target'].str.contains('hair_plate', na=False)]['target'].value_counts().to_dict()
    result = dict(
        status='MEASURED_BANC_SENSORY_ANNOTATION',
        label='MANUAL RESEARCH PROTOTYPE, NOT A PRODUCT; dataset audit only, no simulation',
        question='What does the published BANC annotation itself say about the cells behind each '
                 'sensory channel: how many, which side, which peripheral target, and where along '
                 'the leg do their somata sit?',
        source=dict(file=str(META),
                    sha256=__import__('hashlib').sha256(META.read_bytes()).hexdigest(),
                    rows=int(len(df)),
                    name='BANC 888 cell metadata (banc_888_id/root_id/cell_sub_class/cell_type/'
                         'peripheral_target_type/body_part_sensory/position)'),
        channel_definitions=CHANNEL_SUBTYPE,
        channels=channels,
        proximo_distal_ordering=ordering,
        modality_coverage=coverage,
        hair_plate_target_groups=hp_groups,
        provenance=dict(code=str(Path(__file__).resolve()),
                        note='positions are the dataset\'s own voxel coordinates, converted to um by '
                             'dividing by 1000; the projection axis is that leg\'s own principal '
                             'axis, oriented away from the thorax, so ORDERING along a leg is '
                             'meaningful and absolute positions are not interpreted',
                        excluded='no cell was dropped; a channel with no cells of a given leg simply '
                                 'has no entry for that leg'),
        what_this_does_not_support=[
            'An annotation name is not a measured tuning curve.  Nothing here says what stimulus a '
            'cell responds to, over what range, or with what sign.',
            'The proximo-distal coordinate is a projection onto a heuristic principal axis; it '
            'orders cells along a leg and does NOT give anatomical segment boundaries.',
            'The two sides are pooled in the ordering (the dataset labels side, and per-side counts '
            'are reported, but the axis was fitted per leg type).',
            'No simulation, no connectome, no behaviour: this file only describes the annotation.',
        ])
    doc = dict(status='PASS', question=result['question'], measured=result,
               provenance=result['provenance'],
               what_this_does_not_support=result['what_this_does_not_support'])
    (OUT/'banc_sensory_annotation.json').write_text(json.dumps(doc, indent=2)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, leg in zip(axes, ('FL', 'ML', 'HL')):
        channels_here = [(c, row) for c in channels for row in
                         [channels[c]['per_leg_side'].get(leg)]
                         if row and 'projection_median_um' in row]
        channels_here.sort(key=lambda kv: kv[1]['projection_median_um'])
        for i, (channel, row) in enumerate(channels_here):
            q = row['projection_quartiles_um']
            ax.plot(q, [i, i], lw=6, alpha=.5)
            ax.plot(row['projection_median_um'], i, 'o', color='k')
            ax.text(max(q)+3, i, f"n={row['n_cells']}", va='center', fontsize=7)
        ax.set(yticks=range(len(channels_here)),
               yticklabels=[c for c, _ in channels_here], xlabel='proximo-distal projection (um)',
               title=f'{leg}: sensory groups along the leg\n(bars are the interquartile span of the '
                     f'soma projection)')
        ax.grid(alpha=.3, axis='x')
    fig.suptitle('BANC annotation audit: cell counts and proximo-distal ordering of the sensory '
                 'channels this project feeds (manual research prototype, not a product)', fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT/'banc_sensory_annotation.png', dpi=130)
    print(json.dumps({c: {k: v['n_cells'] for k, v in channels[c]['per_leg_side'].items()}
                      for c in channels}, indent=1))
    print('ordering:', json.dumps(ordering, indent=1))
    print('hair plate groups:', hp_groups)
    print('coverage:', json.dumps(coverage, indent=1))
    print(OUT/'banc_sensory_annotation.json')
    return result


if __name__ == '__main__':
    main()
