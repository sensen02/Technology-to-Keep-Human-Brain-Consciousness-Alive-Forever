"""Cheap static checks, run before any simulation of this task.

No MuJoCo step, no neural integration: it reads the sealed arrays and the dataset only, and it
proves that (1) the encoder gain is the single number being changed, (2) the encoder output is
linear in that gain, (3) the information estimator reproduces the parent numbers 0.43 / 0.86,
(4) the rate diagnostics reproduce the parent numbers 163.8 / 94.5 / 120.5 / 0.72 / 47% / 55%,
and (5) the cell-class join is sound.  Failure of any of these aborts before the warms.
"""
import sys
from pathlib import Path

import numpy as np

import common as C

sys.path[:0] = [str(C.BASE), str(C.R), str(C.R / 'tools')]
js, sha, dump = C.js, C.sha, C.dump

RESULT = {}


def check(name, ok, detail):
    RESULT[name] = dict(passed=bool(ok), detail=detail)
    print(('  OK   ' if ok else '  FAIL ') + name + ' ' + str(detail)[:300], flush=True)
    return bool(ok)


def encoder_group_checks():
    """The gain is the ONLY difference between the four encoders; the encoder is linear in it."""
    from engine.embodied.sensory_channels import (ChannelConfig, ChannelSensoryEncoder,
                                                  ChannelQuantities, channel_index_map, LEGS)
    lookup = (Path('/home/sensen/Desktop/engine修复验收/s3-banc888-runtime-admission/'
                   '20261101T220018Z') / 'SENSORY_LOOKUP_SNAPSHOT888.json')
    assert lookup.exists(), lookup
    roots = np.load(C.IDENTITIES_NPZ)['root_ids']
    groups, stats = channel_index_map(12000, roots, lookup_path=lookup)
    cfg0 = ChannelConfig()
    check('encoder_default_gain_is_0p3', cfg0.current_nA_per_unit == C.REF_GAIN,
          dict(default=cfg0.current_nA_per_unit, reference_scales=dict(
              force=cfg0.reference_force, angle=cfg0.reference_angle_rad,
              velocity=cfg0.reference_velocity_rad_s),
              unmodelled_emit_zero=cfg0.unmodelled_channels_emit_zero,
              cells_placed=stats['cells_placed']))

    # same synthetic physical quantities, four gains, no simulation involved
    q = {leg: ChannelQuantities(tibia_joint_dev_rad=0.11 * (i + 1) - 0.3,
                               tibia_joint_vel_rad_s=3.5 * (i + 1) - 6.0,
                               multi_joint_dev_rad=0.07 * (i + 1) - 0.2,
                               contact_force_norm=0.4 * (i % 3),
                               foot_contact=float((i % 2) == 0)) for i, leg in enumerate(LEGS)}
    cur = {}
    for g in C.GAINS:
        enc = ChannelSensoryEncoder(groups, 12000, ChannelConfig(current_nA_per_unit=g))
        cur[g], rec = enc.encode(q)
        assert rec['scale_nA_per_unit'] == g
    base = cur[C.REF_GAIN]
    scale = {}
    for g in C.GAINS:
        scaled = base * (g / C.REF_GAIN)
        rel = float(np.abs(cur[g] - scaled).max() / max(np.abs(scaled).max(), 1e-300))
        scale['%g' % g] = dict(max_abs_diff=float(np.abs(cur[g] - scaled).max()),
                               max_relative_diff=rel,
                               support_identical=bool(np.array_equal(cur[g] != 0, base != 0)),
                               sign_identical=bool(np.array_equal(np.sign(cur[g]),
                                                                  np.sign(base))),
                               max_abs_current=float(np.abs(cur[g]).max()))
    check('encoder_output_linear_in_gain',
          all(v['support_identical'] and v['sign_identical'] and v['max_relative_diff'] < 1e-13
              for v in scale.values()), scale)
    check('only_the_gain_differs_between_configs',
          all(ChannelConfig(current_nA_per_unit=g).reference_force == cfg0.reference_force
              and ChannelConfig(current_nA_per_unit=g).reference_angle_rad == cfg0.reference_angle_rad
              and ChannelConfig(current_nA_per_unit=g).reference_velocity_rad_s
              == cfg0.reference_velocity_rad_s for g in C.GAINS),
          'reference_force / reference_angle_rad / reference_velocity_rad_s identical at every '
          'gain, so the normalized channel quantities are identical and only the nA-per-unit '
          'factor moves')


def current_stream_checks():
    rec = C.recorded_currents()
    t = C.arrays(C.CURRENTS)['time']
    identical = all(np.array_equal(C.currents_for_gain(g), rec * (g / C.REF_GAIN))
                    for g in C.GAINS)
    at_ref_bitwise = np.array_equal(C.currents_for_gain(C.REF_GAIN), rec)
    check('current_stream_time_grid',
          bool(np.max(np.abs(t - (0.2 + np.arange(1200) * C.CDT))) < 1e-11),
          dict(n=len(t), first=float(t[0]), last=float(t[-1]), dt=float(t[1] - t[0])))
    check('recorded_stream_at_reference_gain_is_used_bitwise', at_ref_bitwise,
          dict(gain=C.REF_GAIN, factor=C.REF_GAIN / C.REF_GAIN,
               shapes=dict(currents=list(rec.shape)), dtype=str(rec.dtype),
               max_abs_nA=float(np.abs(rec).max()),
               nonzero_fraction=float((rec != 0).mean()),
               cells_receiving_current=int((rec != 0).any(0).sum())))
    check('rescaled_stream_definition_is_self_consistent', identical,
          'currents(g) == TRUE_CURRENTS * (g/0.3) for every gain, and this is a linear '
          'rescaling of the recorded encoder output (not a re-recording): the encoder check '
          'above shows the map gain -> current is exactly this product up to float rounding')
    per_gain = {('%g' % g): dict(max_abs_nA=float(np.abs(rec * (g / C.REF_GAIN)).max()),
                                 rms_nA=float(np.sqrt(((rec * (g / C.REF_GAIN)) ** 2).mean())))
                for g in C.GAINS}
    check('rescaled_stream_amplitudes', True, per_gain)


def teacher_checks():
    Y = C.teacher_torque()
    rms = dict(full=float(np.sqrt((Y ** 2).mean())),
               fit=float(np.sqrt((Y[C.FIT_LO:C.FIT_HI] ** 2).mean())),
               held=float(np.sqrt((Y[C.HELD_LO:C.HELD_HI] ** 2).mean())))
    ok = (Y.shape == (1200, 42) and abs(rms['full'] - 3.0221744268988404) < 1e-9
          and abs(rms['held'] - 3.0025179959344728) < 1e-9)
    check('teacher_torque_sampling', ok, dict(shape=list(Y.shape), rms=rms,
                                              note='this reproduces the published teacher RMS '
                                                   '3.022 (full) and 3.003 (held out)'))


def annotation_checks():
    ann = C.anatomy()
    cells = C.motor_cells_204()
    import collections
    sup204 = collections.Counter(ann['super_class'][cells].tolist())
    fun204 = collections.Counter(ann['cell_function'][cells].tolist())
    # flywire arrays carry their own annotation; cross-check by root_id
    z = np.load(C.FLYWIRE, allow_pickle=True)
    rid = z['root_ids']
    asc = z['ann_super_class']
    idx = {int(v): k for k, v in enumerate(rid)}
    fw = np.array([str(asc[idx[int(i)]]) if int(i) in idx else 'NA' for i in ann['ids']],
                  dtype=object)
    fw = np.where(fw == '', 'NA', fw)
    agree = int(np.sum(fw == ann['super_class']))
    check('annotation_join_by_banc_888_id',
          ann['matched'] == 12000 and ann['unmatched'] == 0,
          dict(matched=ann['matched'], unmatched=ann['unmatched'], meta_rows=ann['meta_rows'],
               key=ann['key']))
    check('readout_204_are_all_motor_class',
          dict(sup204).get('motor', 0) == 204,
          dict(super_class=dict(sup204), cell_function=dict(fun204)))
    check('flywire_annotation_agrees_with_the_feather_join', agree == 12000,
          dict(agreeing_cells=agree, cells=12000,
               note='ann_super_class by root_id equals the banc_888_id join for every tier cell, '
                    'so the class used by the rate diagnostics is not an artifact of one table'))
    RESULT['_ann_used_by_task'] = dict(super_class=ann['super_class'].tolist(),
                                       note='kept out of the JSON dump; only the counters are '
                                            'recorded')


def physical_cache_checks():
    with C.PHYSICAL.open('rb') as f:
        import pickle
        phys = pickle.load(f)
    dsm = __import__('dynamic_support_metrics')
    phases = sorted(phys)
    e = phys[1.3]
    check('sealed_physical_handoff_states',
          len(phys) == 17 and abs(0.8 * dsm.H0 - 0.9658) < 5e-5 and phases == sorted(phys),
          dict(count=len(phys), phases=phases,
               integration_size=int(e['integration'].shape[0]),
               assist=e['assist'],
               D8_floor=float(0.8 * dsm.H0),
               all_required_phases_present=all(p in phys for p in C.PHASES),
               encoder_gain_in_cache=e['enc'].cfg.current_nA_per_unit))
    check('sealed_cache_encoder_gain_matches_recorded_stream',
          e['enc'].cfg.current_nA_per_unit == C.REF_GAIN,
          dict(cache_gain=e['enc'].cfg.current_nA_per_unit, recorded_at=C.REF_GAIN))


def metric_reproduction_checks():
    """The information estimator and the diagnostics must reproduce the parent numbers."""
    z05 = C.arrays(C.SEALED_CONFIGS / 'dt_0p5ms' / 'WARM.npz')
    cell = z05['cell_filtered_rates']
    source = np.vstack([np.zeros((1, 12000)), cell[:-1]])
    teacher = C.teacher_torque()
    cells = C.motor_cells_204()

    log = []
    f204 = C.fit_information(source, teacher, cells, 'static_204', log=log.append)
    f12k = C.fit_information(source, teacher, None, 'static_12000', log=log.append)
    check('information_estimator_reproduces_parent_204_cells',
          abs(f204['r2_held'] - 0.4285357152731656) < 5e-4,
          dict(r2_held=f204['r2_held'], r2_fit=f204['r2_fit'], gamma=f204['gamma'],
               parent_published=0.43, previous_stage_exact=0.4285357152731656))
    check('information_estimator_reproduces_parent_12000_cells',
          abs(f12k['r2_held'] - 0.8685005342623501) < 5e-4,
          dict(r2_held=f12k['r2_held'], r2_fit=f12k['r2_fit'], gamma=f12k['gamma'],
               parent_published=0.86, previous_stage_exact=0.8685005342623501))

    ann = C.anatomy()
    diag = C.rate_diagnostics(cell, z05['spikes12000_packed'], ann, cells)
    sc = diag['per_super_class']
    okc = (abs(sc['sensory']['mean_hz'] - 163.8) < 0.5
           and abs(sc['sensory']['median_hz'] - 217) < 1.0
           and abs(sc['motor']['mean_hz'] - 94.5) < 0.5
           and abs(sc['motor']['median_hz'] - 51.6) < 0.5
           and abs(diag['readout_204']['mean_hz'] - 120.5) < 0.5
           and abs(diag['readout_204']['temporal_cv_median'] - 0.72) < 0.01
           and abs(sc['descending']['silent_fraction'] - 0.47) < 0.01
           and abs(sc['central_brain_intrinsic']['silent_fraction'] - 0.55) < 0.01)
    check('rate_diagnostics_reproduce_parent_dense_facts', okc,
          dict(sensory=sc['sensory'], motor=sc['motor'], readout_204=diag['readout_204'],
               descending_silent=sc['descending']['silent_fraction'],
               central_brain_intrinsic_silent=sc['central_brain_intrinsic']['silent_fraction'],
               parent=dict(sensory_mean=163.8, sensory_median=217, motor_mean=94.5,
                           motor_median=51.6, motor_max=337, readout_mean=120.5,
                           readout_cv=0.72, descending_silent=0.47, cbi_silent=0.55)))
    dump(C.O / 'STATIC_GAIN0P3_REFERENCE.json',
         dict(gain=C.REF_GAIN, source='sealed numerics/dt_0p5ms WARM.npz (read-only)',
              diagnostics=diag, fit_204=dict(r2_fit=f204['r2_fit'], r2_held=f204['r2_held'],
                                             gamma=f204['gamma'], alpha=f204['alpha']),
              fit_12000=dict(r2_fit=f12k['r2_fit'], r2_held=f12k['r2_held'],
                             gamma=f12k['gamma'], alpha=f12k['alpha']),
              dense_relief=C.dense_relief(diag),
              note='this is the reference point the sweep is compared against; it is computed '
                   'from the sealed trace, not from a new simulation of this task'))


def main():
    C.O.mkdir(parents=True, exist_ok=True)
    assert (C.O / 'PREREGISTRATION.json').exists(), 'run prereg.py first'
    before = js(C.O / 'INPUT_HASHES_BEFORE.json')
    assert {p: sha(p) for p in before} == before, 'a sealed input changed before the static part'
    encoder_group_checks()
    current_stream_checks()
    teacher_checks()
    annotation_checks()
    physical_cache_checks()
    metric_reproduction_checks()
    RESULT.pop('_ann_used_by_task', None)
    ok = all(v['passed'] for v in RESULT.values())
    dump(C.O / 'STATIC_CHECKS.json', dict(checks=RESULT, all_passed=bool(ok),
                                          importance='核心必需'))
    assert {p: sha(p) for p in before} == before, 'a sealed input changed during the static part'
    print('STATIC', 'ALL PASSED' if ok else 'FAILURES PRESENT', flush=True)
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
