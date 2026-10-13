"""Sensory front-end balance audit on a saved episode.

Read-only. Re-runs the actual encoder path (observe_quantities -> ChannelSensoryEncoder)
on saved body states to measure what each channel actually feeds the network, and to
check the declared reference pose. No integration, no tuning, no current rewrite.
"""
from __future__ import annotations
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import json, signal, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_neural_muscle_closure as closure
from engine.embodied.sensory_channels import ChannelConfig, ChannelSensoryEncoder
TRACE = HERE / 'outputs/model_branches/clock_refinement/clock_refinement_traces.npz'
XML = HERE / 'outputs/model_branches/clock_refinement/fruitfly_zero_floor_6.25us.xml'
OUT = HERE / 'outputs/model_branches/sensory_frontend_balance'
START = time.time()


def main(report):
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(XML))
    _, pose, _, joints, geom_leg = closure.prepare_body()
    with np.load(TRACE) as z:
        qpos = z['6.25us_normal_body_qpos']
        qvel = z['6.25us_normal_body_qvel']
        t = z['6.25us_normal_body_time_s']
    assert np.array_equal(qpos[0], pose), 'trace initial pose is not the closure seated pose'
    built = closure.build_neural_tier(closure.TierConfig(background_mean_nA=0, background_sd_nA=0))
    groups, _ = closure.channel_index_map(int(built['n']), built['tier']['root_ids'])
    tier_roots = [str(r) for r in built['tier']['root_ids']]
    cfg = ChannelConfig()
    enc = ChannelSensoryEncoder(groups, int(built['n']), cfg)
    d = mujoco.MjData(model)
    channels = sorted({ch for leg in groups for ch in groups[leg]})
    n = qpos.shape[0]
    values = {ch: np.zeros((n, len(closure.LEGS))) for ch in channels}
    current_abs = {ch: np.zeros(n) for ch in channels}
    cells = {ch: int(sum(np.asarray(groups[leg][ch]).size for leg in closure.LEGS)) for ch in channels}
    # Per-cell identity so a later step can separate injected from propagable current.
    cell_index, cell_channel, cell_leg = [], [], []
    for ch in channels:
        for leg in closure.LEGS:
            for idx in np.asarray(groups[leg][ch]).ravel():
                cell_index.append(int(idx))
                cell_channel.append(ch)
                cell_leg.append(leg)
    cell_index = np.asarray(cell_index, dtype=np.int64)
    cell_abs_sum = np.zeros(cell_index.size)
    for i in range(n):
        d.qpos[:] = qpos[i]
        d.qvel[:] = qvel[i]
        d.act[:] = 0
        d.ctrl[:] = 0
        mujoco.mj_forward(model, d)
        quantities = closure.observe_quantities(model, d, pose, joints, geom_leg)
        current, record = enc.encode(quantities)
        for li, leg in enumerate(closure.LEGS):
            for ch in channels:
                values[ch][i, li] = record['per_leg'][leg][ch]['value']
        for ch in channels:
            idx = np.concatenate([groups[leg][ch] for leg in closure.LEGS]) if cells[ch] else np.zeros(0, int)
            current_abs[ch][i] = float(np.abs(current[idx]).sum()) if idx.size else 0.0
        cell_abs_sum += np.abs(current[cell_index])
    totals = {ch: float(current_abs[ch].sum()) for ch in channels}
    grand = sum(totals.values())
    summary = {}
    for ch in channels:
        v = values[ch]
        summary[ch] = dict(
            cells=cells[ch], config_normalized_unit={
                'claw_position': 'tibia deviation / %.2f rad' % cfg.reference_angle_rad,
                'hook_direction': 'tibia velocity / %.2f rad/s' % cfg.reference_velocity_rad_s,
                'hairplate_angle': 'mean multi-joint deviation / %.2f rad' % cfg.reference_angle_rad,
                'campaniform_strain': 'sum of foot contact force norms / %.2f model force unit' % cfg.reference_force,
                'bristle_touch': 'foot contact indicator (0/1)',
                'club_vibration': 'unmodelled, emitted zero',
                'multidendritic_noxious': 'unmodelled, emitted zero'}[ch],
            value_min=float(v.min()), value_max=float(v.max()), value_at_t0_max=float(np.abs(v[0]).max()),
            samples_all_zero=int(np.count_nonzero(np.all(v == 0, axis=1))),
            delivered_current_nA_total=float(totals[ch]),
            delivered_current_nA_max_step=float(current_abs[ch].max()),
            share_of_total_input=float(totals[ch] / grand) if grand else None,
            per_leg_value_at_t0={leg: float(v[0, li]) for li, leg in enumerate(closure.LEGS)},
        )
    nonzero_at_t0 = [ch for ch in channels if np.abs(values[ch][0]).max() > 0]
    result = dict(
        status='MEASURED_SENSORY_FRONTEND_BALANCE',
        label='manual read-only front-end audit on saved states, not a tuning result',
        question='What does each declared sensory channel actually feed the network, and what is the reference pose?',
        reference_pose=dict(
            used='the closure seated pose itself (observe_quantities subtracts `seated`)',
            consequence='claw/hook/hairplate report exactly zero deviation at t=0, i.e. the network senses deviation from where it started, not from a traceable anatomical zero',
            external_note='earlier literature review requires an anatomical zero, not a posture target'),
        config=dict(current_nA_per_unit=cfg.current_nA_per_unit, reference_angle_rad=cfg.reference_angle_rad,
                    reference_velocity_rad_s=cfg.reference_velocity_rad_s, reference_force=cfg.reference_force,
                    config_provenance=cfg.provenance),
        channels_nonzero_at_t0=nonzero_at_t0,
        per_channel=summary,
        total_input_current_nA_steps=float(grand),
        limitations=['One saved normal episode, one pose, one tier; no new dynamics and no parameter change.',
                     'A channel with few verified-transmitter cells cannot drive the network even when it carries the largest current.',
                     'Normalizers are the project declared values, explicitly not measurements.'],
        memory_guard=report('completed'), wall_s=time.time() - START)
    if not np.isfinite(result['memory_guard'].get('previous_soft_limit_mb', 0.0)):
        result['memory_guard']['previous_soft_limit_mb'] = 'unlimited'
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'sensory_frontend_balance.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(OUT / 'sensory_frontend_balance_traces.npz', time_s=t,
                        **{f'value_{ch}': values[ch] for ch in channels},
                        **{f'current_{ch}': current_abs[ch] for ch in channels},
                        legs=np.asarray(closure.LEGS),
                        cell_local_index=cell_index,
                        cell_channel=np.asarray(cell_channel),
                        cell_leg=np.asarray(cell_leg),
                        cell_current_mean_abs_nA=cell_abs_sum / n,
                        cell_root_id=np.asarray([tier_roots[i] for i in cell_index]))
    print(json.dumps({ch: {k: summary[ch][k] for k in ('cells', 'value_max', 'delivered_current_nA_total',
                                                       'share_of_total_input', 'samples_all_zero')}
                      for ch in channels}, indent=2))
    return result


if __name__ == '__main__':
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('60s front-end-audit deadline')))
    signal.alarm(60)
    with closure.mg.guarded_run('sensory_frontend_balance', **closure.mg.BUDGETS['neural_muscle_closure']) as report:
        main(report)
    signal.alarm(0)
