"""Cheap static part: fit the linear readout and describe it. No simulation at all.

Reads only the sealed warm-cache pool rates and the sealed teacher torque.
Writes the fitted readout as arrays plus a JSON description, and every basic
property the task asks for (range vs limit, saturation, correlation with the
teacher torque, fit-versus-fit stability).
"""
import numpy as np

import common as C


def safe_corr(a, b):
    """Pearson correlation; a constant signal (fully saturated joint) is reported as 0.0."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return 0.0 if d == 0.0 else float((a * b).sum() / d)


def describe(tag, raw, clipped, limits, target, extra=None):
    sat = np.abs(np.abs(clipped) - limits[None, :]) < 1e-12
    per_joint_r = [safe_corr(clipped[:, j], target[:, j]) for j in range(42)]
    overall = safe_corr(clipped.ravel(), target.ravel())
    out = dict(
        tag=tag,
        command_rms=float(np.sqrt((clipped ** 2).mean())),
        teacher_rms=float(np.sqrt((target ** 2).mean())),
        command_rms_over_teacher_rms=float(np.sqrt((clipped ** 2).mean()) / np.sqrt((target ** 2).mean())),
        overall_correlation_with_teacher=overall,
        per_joint_correlation_median=float(np.median(per_joint_r)),
        per_joint_correlation_min=float(np.min(per_joint_r)),
        per_joint_correlation_max=float(np.max(per_joint_r)),
        per_joint_above_0p5=int(np.sum(np.array(per_joint_r) > 0.5)),
        sign_agreement=float(np.mean(np.sign(clipped) == np.sign(target))),
        saturation_fraction_overall=float(sat.mean()),
        saturation_fraction_per_joint_median=float(np.median(sat.mean(0))),
        saturation_fraction_per_joint_max=float(sat.max(0).max()),
        joints_with_saturation=int(np.sum(sat.any(0))),
        saturation_fraction_per_window=dict(
            fit=float(sat[C.FIT_LO:C.FIT_HI].mean()),
            held=float(sat[C.HELD_LO:C.HELD_HI].mean())),
        per_joint_command_range_vs_limit=[
            dict(joint=int(j), limit=float(limits[j]),
                 command_min=float(clipped[:, j].min()), command_max=float(clipped[:, j].max()),
                 command_absmax_over_limit=float(np.abs(clipped[:, j]).max() / limits[j]),
                 saturation_fraction=float(sat[:, j].mean()),
                 teacher_absmax=float(np.abs(target[:, j]).max()))
            for j in range(42)],
        per_joint_command_absmax_over_limit_median=float(
            np.median([np.abs(clipped[:, j]).max() / limits[j] for j in range(42)])),
        per_joint_command_absmax_over_limit_min=float(
            np.min([np.abs(clipped[:, j]).max() / limits[j] for j in range(42)])),
        per_joint_command_absmax_over_limit_max=float(
            np.max([np.abs(clipped[:, j]).max() / limits[j] for j in range(42)])),
    )
    if extra:
        out.update(extra)
    return out


def main():
    if not (C.O / 'PREREGISTRATION.json').exists():
        raise SystemExit('run prereg.py first')
    teacher = C.teacher_torque()
    matrix, limits = C.decoder_identity()

    warm = {}
    for dt, tag in C.DTS:
        z, source = C.warm_rates(tag)
        warm[tag] = dict(dt=dt, source=source, z=z)

    # ---- fit on the 0.5 ms warm cache, first 600 intervals (0.2-0.8 s) --------
    src_fit = warm['dt_0p5ms']['source']
    W, b = C.fit_readout(src_fit, teacher, C.FIT_LO, C.FIT_HI)
    raw_fit = src_fit @ W.T + b                       # unclipped, full window
    scale = float(np.sqrt((teacher[C.FIT_LO:C.FIT_HI] ** 2).mean())
                  / np.sqrt((raw_fit[C.FIT_LO:C.FIT_HI] ** 2).mean()))

    # ---- stability: independent fit on the held window ------------------------
    W_held, b_held = C.fit_readout(src_fit, teacher, C.HELD_LO, C.HELD_HI)

    per_tag = {}
    for dt, tag in C.DTS:
        source = warm[tag]['source']
        per_tag[tag] = {}
        per_tag[tag]['R0'] = describe('R0', limits / 100.0 * (source @ matrix.T),
                                      np.clip(limits / 100.0 * (source @ matrix.T), -limits, limits),
                                      limits, teacher)
        rr = source @ W.T + b
        per_tag[tag]['R1'] = describe('R1', rr, np.clip(rr, -limits, limits), limits, teacher,
                                      dict(scale=1.0))
        r2r = scale * rr
        per_tag[tag]['R2'] = describe('R2', r2r, np.clip(r2r, -limits, limits), limits, teacher,
                                      dict(scale=scale))
        # held-out explanatory power of the single fitted readout on this cache
        for tag2, raw in [('R1', rr), ('R2', r2r)]:
            cl = np.clip(raw, -limits, limits)
            per_tag[tag][tag2]['r2_fit_window_unclipped'] = float(
                C.r2_score(teacher[C.FIT_LO:C.FIT_HI], raw[C.FIT_LO:C.FIT_HI]))
            per_tag[tag][tag2]['r2_held_window_unclipped'] = float(
                C.r2_score(teacher[C.HELD_LO:C.HELD_HI], raw[C.HELD_LO:C.HELD_HI]))
            per_tag[tag][tag2]['r2_held_window_after_clipping'] = float(
                C.r2_score(teacher[C.HELD_LO:C.HELD_HI], cl[C.HELD_LO:C.HELD_HI]))
        # R0 must reproduce the stored sealed command trace of this cache
        cl0 = np.clip(limits / 100.0 * (source @ matrix.T), -limits, limits)
        per_tag[tag]['R0']['stored_clipped_command_max_abs_diff'] = float(
            np.abs(cl0 - warm[tag]['z']['clipped_command']).max())
        per_tag[tag]['R0']['stored_raw_command_max_abs_diff'] = float(
            np.abs(limits / 100.0 * (source @ matrix.T) - warm[tag]['z']['raw_command']).max())

    stability = dict(
        fit_window_s=[0.2, 0.8], second_fit_window_s=[0.8, 1.4],
        matrix_vector_correlation=safe_corr(W.ravel(), W_held.ravel()),
        mean_abs_matrix=float(np.abs(W).mean()), mean_abs_matrix_second_fit=float(np.abs(W_held).mean()),
        matrix_frobenius_norm=float(np.linalg.norm(W)),
        relative_frobenius_difference=float(np.linalg.norm(W - W_held) / np.linalg.norm(W)),
        max_abs_element_difference=float(np.abs(W - W_held).max()),
        max_abs_element=float(np.abs(W).max()),
        bias_vector_correlation=safe_corr(b, b_held),
        bias_rms=float(np.sqrt((b ** 2).mean())), bias_rms_second_fit=float(np.sqrt((b_held ** 2).mean())),
        r2_on_own_window=dict(first=float(C.r2_score(teacher[C.FIT_LO:C.FIT_HI],
                                                     (src_fit @ W.T + b)[C.FIT_LO:C.FIT_HI])),
                              second=float(C.r2_score(teacher[C.HELD_LO:C.HELD_HI],
                                                      (src_fit @ W_held.T + b_held)[C.HELD_LO:C.HELD_HI]))),
        r2_crossed=dict(first_matrix_on_held=float(C.r2_score(teacher[C.HELD_LO:C.HELD_HI],
                                                              (src_fit @ W.T + b)[C.HELD_LO:C.HELD_HI])),
                        second_matrix_on_fit=float(C.r2_score(teacher[C.FIT_LO:C.FIT_HI],
                                                              (src_fit @ W_held.T + b_held)[C.FIT_LO:C.FIT_HI]))),
        interpretation='Two independent least-squares fits on disjoint half windows of the same '
                       'teacher disagreement in matrix elements is the direct measure of readout '
                       'stability; a large crossed-R^2 drop means the fitted interface is only '
                       'weakly identified by this data.')
    summary = dict(
        verdict_stability=('UNSTABLE' if stability['matrix_vector_correlation'] < 0.5
                           else ('WEAKLY_IDENTIFIED' if stability['matrix_vector_correlation'] < 0.8
                                 else 'STABLE_BY_CORRELATION')),
        note='This label is about the fitted interface, not about the neurons.')

    np.savez_compressed(C.O / 'READOUT_R1.npz', W_fit=W, b_fit=b, scale_r2=np.array(scale),
                        W_stability_fit_0p8_1p4=W_held, b_stability_fit_0p8_1p4=b_held,
                        limits=limits, old_matrix=matrix,
                        fit_window=np.array([C.FIT_LO, C.FIT_HI]),
                        held_window=np.array([C.HELD_LO, C.HELD_HI]))
    C.dump(C.O / 'READOUT_DESCRIPTION.json', dict(
        file=str(C.O / 'READOUT_R1.npz'),
        arrays=dict(W_fit=[42, 66], b_fit=[42], scale_r2=[], W_stability_fit_0p8_1p4=[42, 66],
                    b_stability_fit_0p8_1p4=[42], limits=[42], old_matrix=[42, 66]),
        R1_rule='command = clip(W_fit @ rates + b_fit, -limits, +limits)',
        R2_rule=f'command = clip({scale!r} * (W_fit @ rates + b_fit), -limits, +limits)',
        scale_r2=scale,
        scale_definition='s = RMS(teacher torque over 0.2-0.8 s) / RMS(unclipped R1 command over '
                         '0.2-0.8 s), elementwise over 42 joints x 600 intervals.',
        rates='the runtime 66-pool rates vector (the same vector the current readout consumes)',
        fitted_on='sealed WARM.npz command_source_rates of the 0.5 ms cache, 0.2-0.8 s, '
                  'targeted at the teacher applied joint torque',
        label='OFFLINE ENGINEERING FITTING of the readout interface, NOT biological calibration. '
              'The neurons still generate the command.',
        importance='核心必需'))
    C.dump(C.O / 'READOUT_STATIC.json', dict(
        per_timestep=per_tag, stability=stability, stability_summary=summary,
        teacher_rms=dict(full=float(np.sqrt((teacher ** 2).mean())),
                         fit=float(np.sqrt((teacher[C.FIT_LO:C.FIT_HI] ** 2).mean())),
                         held=float(np.sqrt((teacher[C.HELD_LO:C.HELD_HI] ** 2).mean())),
                         static_full=float(np.sqrt((teacher.mean(0) ** 2).mean())),
                         fluctuation_full=float(np.sqrt(((teacher - teacher.mean(0)) ** 2).mean()))),
        R1_r2_train=float(C.r2_score(teacher[C.FIT_LO:C.FIT_HI], raw_fit[C.FIT_LO:C.FIT_HI])),
        R1_r2_held=float(C.r2_score(teacher[C.HELD_LO:C.HELD_HI], raw_fit[C.HELD_LO:C.HELD_HI])),
        R1_r2_train_second_fit=float(C.r2_score(teacher[C.FIT_LO:C.FIT_HI],
                                                (src_fit @ W_held.T + b_held)[C.FIT_LO:C.FIT_HI])),
        R1_r2_held_second_fit=float(C.r2_score(teacher[C.HELD_LO:C.HELD_HI],
                                              (src_fit @ W_held.T + b_held)[C.HELD_LO:C.HELD_HI])),
        drift_note='Teacher torque is sampled at 1 ms command boundaries from the 0.5 ms '
                   'teacher BODY.npz (index 400+2k). RMS full=3.022 and held=3.003 reproduce '
                   'the parent published values exactly; the parent R^2 (0.542/0.326) is '
                   'reproduced only in magnitude (this file reports its own exact convention).',
        importance='核心必需'))
    print('FITTED scale', scale, 'held R2', C.r2_score(teacher[C.HELD_LO:C.HELD_HI], raw_fit[C.HELD_LO:C.HELD_HI]), flush=True)


if __name__ == '__main__':
    main()
