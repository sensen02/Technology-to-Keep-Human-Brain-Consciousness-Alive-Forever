"""Load-support audit: does the modelled ground contact actually carry body weight?

Read-only w.r.t. frozen inputs. Two normal 200 ms episodes (100 us and 25 us body
step) with an observer that records total vertical ground reaction force and the
per-leg share, compared with the model's own weight in model force units.
"""
from __future__ import annotations
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import json, signal, sys, time
from pathlib import Path
from unittest.mock import patch
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_neural_duration_probe as duration
import run_zero_floor_branch as zero
from engine.embodied import branch_episode as branch
closure = duration.closure
OUT = HERE / 'outputs/model_branches/support_audit'
CASES = ((100, zero.XML), (25, HERE / 'outputs/model_branches/fine_clock/fruitfly_zero_floor_25us.xml'))
START = time.time()


def vertical_grf(model, data, geom_leg, buf):
    """Return (total world wrench, foot world force, other world force, nfoot, ncon) for both frame conventions."""
    total = {'A': np.zeros(3), 'B': np.zeros(3)}
    foot = {'A': np.zeros(3), 'B': np.zeros(3)}
    other = {'A': np.zeros(3), 'B': np.zeros(3)}
    nfoot = 0
    for i in range(data.ncon):
        c = data.contact[i]
        mujoco.mj_contactForce(model, data, i, buf)
        frame = np.asarray(c.frame).reshape(3, 3)
        for tag, world in (('A', frame @ buf[:3]), ('B', frame.T @ buf[:3])):
            total[tag] += world
            if geom_leg.get(int(c.geom1)) or geom_leg.get(int(c.geom2)):
                foot[tag] += world
            else:
                other[tag] += world
        if geom_leg.get(int(c.geom1)) or geom_leg.get(int(c.geom2)):
            nfoot += 1
    return total, foot, other, nfoot


def run_case(model, pose, joints, geom_leg, built, manifest, groups, dt):
    import mujoco
    rec = {k: [] for k in ('time_s', 'root_z', 'r_qpos', 'ncon', 'nfoot',
                           'grf_z_A', 'grf_z_B', 'foot_z_A', 'foot_z_B', 'other_z_A', 'other_z_B',
                           'grf_mag_A', 'grf_mag_B', 'leg_z_A')}
    real_forward = mujoco.mj_forward
    buf = np.zeros(6)

    def observer(m, d):
        real_forward(m, d)
        total, foot, other, nfoot = vertical_grf(m, d, geom_leg, buf)
        rec['time_s'].append(float(d.time))
        rec['root_z'].append(float(d.qpos[2]))
        rec['r_qpos'].append(d.qpos.copy())
        rec['ncon'].append(int(d.ncon))
        rec['nfoot'].append(nfoot)
        rec['grf_z_A'].append(float(total['A'][2]))
        rec['grf_z_B'].append(float(total['B'][2]))
        rec['foot_z_A'].append(float(foot['A'][2]))
        rec['foot_z_B'].append(float(foot['B'][2]))
        rec['other_z_A'].append(float(other['A'][2]))
        rec['other_z_B'].append(float(other['B'][2]))
        rec['grf_mag_A'].append(float(np.linalg.norm(total['A'])))
        rec['grf_mag_B'].append(float(np.linalg.norm(total['B'])))
        per_leg = {leg: 0.0 for leg in closure.LEGS}
        for i in range(d.ncon):
            c = d.contact[i]
            mujoco.mj_contactForce(m, d, i, buf)
            frame = np.asarray(c.frame).reshape(3, 3)
            world = frame.T @ buf[:3]
            for g in (int(c.geom1), int(c.geom2)):
                leg = geom_leg.get(g)
                if leg:
                    per_leg[leg] += float(world[2])
        rec['leg_z_A'].append([per_leg[leg] for leg in closure.LEGS])
    with patch.object(mujoco, 'mj_forward', observer):
        summary, trajectories, commands, spike_counts = branch.episode(
            model, pose, joints, geom_leg, built, manifest, groups, 'normal', .2, body_dt=dt)
    summary.pop('records', None)
    rec['trajectory'] = np.asarray(trajectories)
    rec['controls'] = np.asarray(commands)
    rec['spike_counts'] = np.asarray(spike_counts)
    return summary, {k: np.asarray(v) for k, v in rec.items()}, rec['r_qpos']


def leg_detail(model, pose, geom_leg, buf):
    """Static per-leg contact and load detail at the declared initial pose; no integration."""
    import mujoco
    d = mujoco.MjData(model)
    d.qpos[:] = pose
    d.qvel[:] = 0
    d.act[:] = 0
    d.ctrl[:] = 0
    mujoco.mj_forward(model, d)
    out = {}
    for leg in closure.LEGS:
        geoms = [g for g, l in geom_leg.items() if l == leg]
        info = dict(tarsal_geoms=len(geoms), contacts=0, normal_magnitude=0.0, vertical_force=0.0,
                    min_contact_distance=None, foot_geom_z_min=None, foot_geom_z_max=None)
        for i in range(d.ncon):
            c = d.contact[i]
            if int(c.geom1) in geoms or int(c.geom2) in geoms:
                mujoco.mj_contactForce(model, d, i, buf)
                info['contacts'] += 1
                info['normal_magnitude'] += float(np.linalg.norm(buf[:3]))
                frame = np.asarray(c.frame).reshape(3, 3)
                info['vertical_force'] += float((frame.T @ buf[:3])[2])
                dist = float(c.dist)
                info['min_contact_distance'] = dist if info['min_contact_distance'] is None else min(info['min_contact_distance'], dist)
        zs = [float(d.geom_xpos[g][2]) for g in geoms]
        info['foot_geom_z_min'], info['foot_geom_z_max'] = min(zs), max(zs)
        out[leg] = info
    return out


def main(report):
    import mujoco
    base, pose, _, joints, geom_leg = closure.prepare_body()
    built = closure.build_neural_tier(closure.TierConfig(background_mean_nA=0, background_sd_nA=0))
    names = [mujoco.mj_id2name(base, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(base.nu)]
    manifest = closure.build_motor_manifest(json.loads(closure.POOLS.read_text()), built['tier']['root_ids'], names)
    groups, _ = closure.channel_index_map(int(built['n']), built['tier']['root_ids'])
    total_mass = float(base.body_mass.sum())
    weight = total_mass * abs(float(base.opt.gravity[2]))
    weight_per_leg = weight / 6.0
    cases, arrays = {}, {}
    buf = np.zeros(6)
    initial_leg_detail = {'canonical_100us': leg_detail(base, pose, geom_leg, buf)}
    for us, xml in CASES:
        model = mujoco.MjModel.from_xml_path(str(xml))
        dt = us * 1e-6
        assert np.isclose(model.opt.timestep, dt, rtol=0, atol=1e-15)
        summary, rec, qpos = run_case(model, pose, joints, geom_leg, built, manifest, groups, dt)
        z = rec['grf_z_B']
        supported = z > 0.5 * weight
        first_loss = float(rec['time_s'][np.argmax(~supported)]) if (~supported).any() else None
        cases[f'{us}us_normal'] = dict(
            body_dt_s=dt, body_steps=int(round(.2 / dt)), neural_steps=400,
            escape_time_s=summary['escape_time_s'], final_height_model=summary['final_height_model'],
            samples=int(rec['time_s'].size), sample_dt_median_s=float(np.median(np.diff(rec['time_s']))),
            weight_model_units=weight, weight_per_leg_model_units=weight_per_leg,
            initial_root_z_mm=float(rec['root_z'][0]),
            initial_grf_z_convention_A=float(rec['grf_z_A'][0]), initial_grf_z_convention_B=float(rec['grf_z_B'][0]),
            initial_grf_z_over_weight=float(rec['grf_z_B'][0] / weight),
            initial_ncon=int(rec['ncon'][0]), initial_nfoot=int(rec['nfoot'][0]),
            initial_per_leg_vertical_force={leg: float(v) for leg, v in zip(closure.LEGS, rec['leg_z_A'][0])},
            max_grf_z_over_weight=float(np.max(rec['grf_z_B']) / weight),
            min_grf_z_over_weight=float(np.min(rec['grf_z_B']) / weight),
            mean_grf_z_over_weight=float(np.mean(rec['grf_z_B']) / weight),
            final_grf_z_over_weight=float(rec['grf_z_B'][-1] / weight),
            time_support_below_half_weight_s=first_loss,
            final_root_z_mm=float(rec['root_z'][-1]),
            nonfoot_contacts_initial=int(rec['ncon'][0] - rec['nfoot'][0]),
            finite=bool(np.isfinite(np.concatenate([rec['grf_z_A'], rec['grf_z_B'], rec['root_z']])).all()),
        )
        arrays.update({f'{us}us_{k}': v for k, v in rec.items()})
        arrays[f'{us}us_body_qpos'] = qpos
        # Continuity check with previously published zero-floor trace at 100 us.
        if us == 100:
            with np.load(HERE / 'outputs/model_branches/zero_floor/zero_floor_traces.npz') as old:
                cases['100us_normal']['root_z_matches_saved_zero_floor'] = bool(
                    np.array_equal(rec['root_z'], old['normal_heights_mm'][:, 0]))
    OUT.mkdir(parents=True, exist_ok=True)
    npz = OUT / 'support_audit_traces.npz'
    np.savez_compressed(npz, **arrays)
    result = dict(
        status='MEASURED_GROUND_REACTION_FORCE_VS_WEIGHT',
        label='manual read-only load-support audit, not a standing or walking result',
        question='Does the modelled ground contact carry the body weight, and when is support lost?',
        frame_convention='A: world = frame @ f_local ; B: world = frame.T @ f_local. Both recorded; B is used for ratios.',
        weight_and_mass=dict(total_mass_model_units=total_mass, gravity_z=float(base.opt.gravity[2]), weight_model_units=weight,
                             weight_per_leg_model_units=weight_per_leg,
                             note='Model units as MuJoCo sees them; absolute SI calibration of mass is still unresolved.'),
        cases=cases, initial_leg_detail=initial_leg_detail,
        provenance=dict(models={str(us): str(x) for us, x in CASES}, npz=str(npz),
                        npz_sha256=hashlib_sha(npz)),
        limitations=['Two normal episodes only; no silence arm and no new timesteps.',
                     'Contact force convention is validated by the physical requirement that support is upward while resting.',
                     'A support ratio is not a stability proof: transient support can still accompany falling.',
                     'No controller, gain, or parameter was changed; frozen canonical inputs untouched.'],
        memory_guard=report('completed'), wall_s=time.time() - START)
    if not np.isfinite(result['memory_guard'].get('previous_soft_limit_mb', 0.0)):
        result['memory_guard']['previous_soft_limit_mb'] = 'unlimited'
    (OUT / 'support_audit_results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(cases, indent=2))
    return result


def hashlib_sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == '__main__':
    import mujoco
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('60s support-audit deadline')))
    signal.alarm(60)
    with closure.mg.guarded_run('support_audit', **closure.mg.BUDGETS['neural_muscle_closure']) as report:
        main(report)
    signal.alarm(0)
