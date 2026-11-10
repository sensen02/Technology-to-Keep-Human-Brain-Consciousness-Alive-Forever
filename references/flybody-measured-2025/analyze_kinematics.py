#!/usr/bin/env python3
"""Bounded measured-keypoint-fitted reference analysis. No physics or training.
Run with the existing flybody-teacher .venv/bin/python -B. Only writes the
sibling KINEMATICS.json and selected_reference.npz. Never transfers actions.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import xml.etree.ElementTree as ET
import h5py
import numpy as np

BASE = Path(__file__).resolve().parent
OFFICIAL = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-teacher/20261107T174640Z/official-flybody/flybody')
H5 = BASE / 'walking_reference_small.hdf5'
EXPECTED_SHA256 = '5b573757be3505eb609eb7a2bbdef75e3de7573ec285845fdfa94240aed2cd63'
LEGS = ['RF', 'RM', 'RH', 'LH', 'LM', 'LF']
PARTS = ['T1_right', 'T2_right', 'T3_right', 'T3_left', 'T2_left', 'T1_left']
DOFS = ['coxa_abduct', 'coxa_twist', 'coxa', 'femur_twist', 'femur', 'tibia', 'tarsus']


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stats(a):
    a = np.asarray(a)
    if not a.size:
        return None
    return dict(min=float(np.min(a)), median=float(np.median(a)),
                p95=float(np.percentile(a, 95)), p99=float(np.percentile(a, 99)),
                max=float(np.max(a)))


def rotate(v, q):
    """Scalar-first active rotation of local vectors into world frame."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    xyz = q[..., 1:]
    t = 2 * np.cross(xyz, v)
    return v + q[..., :1] * t + np.cross(xyz, t)


def world(root, local):
    return root[:, None, :3] + rotate(local, root[:, None, 3:7])


def swing_peaks(height, dt):
    # Five-sample (10 ms) smoothing; peak must exceed its local +/- 40 ms
    # minima by >= 0.003 cm and 20% of the leg's P95-P05 height range.
    smooth = np.convolve(np.pad(height, (2, 2), mode='edge'), np.ones(5)/5, mode='valid')
    threshold = max(.003, .20 * float(np.ptp(np.percentile(height, [5, 95]))))
    radius = max(2, round(.040 / dt))
    candidates = []
    for k in range(1, len(smooth)-1):
        if smooth[k] > smooth[k-1] and smooth[k] >= smooth[k+1]:
            lo, hi = max(0, k-radius), min(len(smooth), k+radius+1)
            prom = smooth[k] - max(np.min(smooth[lo:k+1]), np.min(smooth[k:hi]))
            if prom >= threshold:
                candidates.append(k)
    chosen = []
    for k in sorted(candidates, key=lambda k: -smooth[k]):
        if all(abs(k-j)*dt >= .035 for j in chosen):
            chosen.append(k)
    return np.array(sorted(chosen), dtype=int)


def plane_height(feet):
    # Fit z=a*x+b*y+c to each leg's lowest 20% of samples. This estimated
    # lower envelope is not a measured ground surface or contact label.
    low = feet[:, :, 2] <= np.percentile(feet[:, :, 2], 20, axis=0)[None, :]
    pts = feet[low]
    design = np.c_[pts[:, :2], np.ones(len(pts))]
    coef = np.linalg.lstsq(design, pts[:, 2], rcond=None)[0]
    pred = feet[..., 0]*coef[0] + feet[..., 1]*coef[1] + coef[2]
    signed = (feet[..., 2]-pred) / np.sqrt(1+np.sum(coef[:2]**2))
    return coef, signed, float(np.sqrt(np.mean((design@coef-pts[:, 2])**2)))


def motion(root, dt):
    xaxis = rotate(np.array([1., 0., 0.]), root[:, 3:7])
    yaw = np.unwrap(np.arctan2(xaxis[:, 1], xaxis[:, 0]))
    forward = xaxis[0, :2] / np.linalg.norm(xaxis[0, :2])
    lateral = np.array([-forward[1], forward[0]])
    delta = root[-1, :2]-root[0, :2]
    duration = (len(root)-1)*dt
    velocity = np.diff(root[:, :2], axis=0)/dt
    fwd = float(delta@forward)
    lat = float(delta@lateral)
    return dict(duration_s=duration, sample_count=len(root),
                net_forward_cm=fwd, net_lateral_cm=lat,
                lateral_to_forward_ratio=abs(lat)/max(abs(fwd), 1e-9),
                yaw_range_deg=float(np.ptp(yaw)*180/np.pi),
                yaw_net_deg=float((yaw[-1]-yaw[0])*180/np.pi),
                mean_signed_forward_speed_cm_s=fwd/duration,
                horizontal_speed_cm_s=stats(np.linalg.norm(velocity, axis=1)),
                backward_velocity_fraction=float(np.mean(velocity@forward < 0))), forward


def joint_stats(qvel, names):
    result = {}
    indices = []
    for leg, part in zip(LEGS, PARTS):
        rows = []
        for dof in DOFS:
            name = f'{dof}_{part}'
            idx = names.index(name)
            indices.append(idx)
            rows.append(dict(name=name, index=idx, abs_native_rad_s=stats(np.abs(qvel[:, idx]))))
        result[leg] = rows
    arr = np.abs(qvel[:, indices])
    per_joint_p99 = np.percentile(arr, 99, axis=0)
    return dict(per_leg=result, pooled_abs_native_rad_s=stats(arr),
                max_per_joint_p99_rad_s=float(np.max(per_joint_p99)),
                peak_rad_s=float(np.max(arr))), indices


def main():
    assert digest(H5) == EXPECTED_SHA256
    spec = importlib.util.spec_from_file_location('official_quaternions', OFFICIAL/'quaternions.py')
    oq = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oq)  # numpy-only official utility, no simulator imports.
    xml = ET.parse(OFFICIAL/'fruitfly/assets/fruitfly.xml').getroot()
    assert xml.find('compiler').attrib['angle'] == 'radian'
    gravity = [float(v) for v in xml.find('option').attrib['gravity'].split()]
    assert gravity == [0., 0., -981.]
    with h5py.File(H5, 'r') as f:
        dt = float(f['timestep_seconds'][()])
        assert dt == .002 and len(f['trajectories']) == 100
        names = [v.decode() for v in f['id2name/qpos'][()]]
        sites = [v.decode() for v in f['id2name/sites'][()]]
        assert names == [v.decode() for v in f['id2name/joints'][()]]
        assert sites == [f'claw_{p}' for p in PARTS]
        all_velocity = []
        candidates = []
        norm_errors = []
        for key, g in f['trajectories'].items():
            root = g['root_qpos'][()]
            local = g['root2site'][()]
            assert root.shape == (len(root), 7) and local.shape == (len(root), 6, 3)
            norm_errors.append(np.max(np.abs(np.linalg.norm(root[:, 3:], axis=1)-1)))
            all_velocity.append(g['qvel'][()])
            # All fixed 0.6 s windows at 50 ms spacing, plus complete snippets
            # when 0.5 <= endpoint duration <= 1.0 s. No unlimited clip search.
            spans = set()
            count = round(.6/dt)+1
            if len(root) >= count:
                for start in list(range(0, len(root)-count+1, 25)) + [len(root)-count]:
                    spans.add((start, start+count))
            if .5 <= (len(root)-1)*dt <= 1.0:
                spans.add((0, len(root)))
            for start, end in sorted(spans):
                r, l = root[start:end], local[start:end]
                m, _ = motion(r, dt)
                feet = world(r, l)
                counts = [len(swing_peaks(feet[:, i, 2], dt)) for i in range(6)]
                complete = [max(0, n-1) for n in counts]
                eligible = (m['net_forward_cm'] > .1 and m['yaw_range_deg'] <= 10
                            and m['lateral_to_forward_ratio'] <= .15 and min(complete) >= 2)
                score = (m['net_forward_cm'] / (1+4*m['lateral_to_forward_ratio']
                         + m['yaw_range_deg']/10) * min(1., min(complete)/3))
                candidates.append(dict(trajectory_id=key, start_index=start,
                    end_index_exclusive=end, eligible=eligible, score=score,
                    complete_peak_intervals_per_leg=complete, **m))
        ranked = sorted(candidates, key=lambda v: (v['eligible'], v['score']), reverse=True)
        chosen = ranked[0]
        assert chosen['eligible'], 'No clip passes declared straight/cycle criteria.'
        g = f['trajectories'][chosen['trajectory_id']]
        sl = slice(chosen['start_index'], chosen['end_index_exclusive'])
        raw = {k: g[k][sl] for k in ('root_qpos','root_qvel','root2site','qpos','qvel')}
    feet = world(raw['root_qpos'], raw['root2site'])
    # Numerical equality to official transform and inverse roundtrip; no FK,
    # dynamics or assumptions about target morphology enter these checks.
    # Explicitly broadcast for official utility's np.tile heuristic, which
    # otherwise repeats the time dimension when given (T,1,4) and (T,6,3).
    quat_sites = np.broadcast_to(raw['root_qpos'][:, None, 3:7], (len(feet), 6, 4)).copy()
    official_feet = raw['root_qpos'][:, None, :3] + oq.rotate_vec_with_quat(
        raw['root2site'], quat_sites)
    recovered = oq.get_egocentric_vec(raw['root_qpos'][:, None, :3], feet, quat_sites)
    transform_error = float(np.max(np.abs(feet-official_feet)))
    inverse_error = float(np.max(np.abs(recovered-raw['root2site'])))
    assert transform_error < 1e-6 and inverse_error < 1e-6
    coef, height, plane_rmse = plane_height(feet)
    _, forward = motion(raw['root_qpos'], dt)
    speed = np.linalg.norm(np.gradient(feet, dt, axis=0), axis=-1)
    stance_speed_threshold = max(.1, chosen['mean_signed_forward_speed_cm_s']*.4)
    stance = (height <= .005) & (height >= -.005) & (speed <= stance_speed_threshold)
    leg_metrics = {}
    for i, leg in enumerate(LEGS):
        peaks = swing_peaks(height[:, i], dt)
        periods = np.diff(peaks)*dt
        strides = np.diff(feet[peaks, i, :2], axis=0)@forward
        peak_to_trough = [float(np.max(height[a:b+1, i])-np.min(height[a:b+1, i]))
                          for a, b in zip(peaks[:-1], peaks[1:])]
        leg_metrics[leg] = dict(site=sites[i], swing_peak_sample_indices=peaks.tolist(),
            complete_peak_intervals=len(periods), periods_s=periods.tolist(),
            cadence_hz=stats(1/periods), signed_peak_to_peak_stride_cm=stats(strides),
            clearance_peak_to_trough_cm=stats(peak_to_trough),
            signed_height_above_inferred_plane_cm=stats(height[:, i]),
            swing_peak_height_above_inferred_plane_cm=stats(height[peaks, i]),
            near_plane_low_speed_candidate_fraction=float(np.mean(stance[:, i])),
            world_foot_speed_cm_s=stats(speed[:, i]))
    selected_joints, indices = joint_stats(raw['qvel'], names)
    dataset_joints, _ = joint_stats(np.concatenate(all_velocity), names)
    fd = np.gradient(raw['qpos'], dt, axis=0)[2:-2, indices]
    native = raw['qvel'][2:-2, indices]
    root_fd = np.gradient(raw['root_qpos'][:, :3], dt, axis=0)[2:-2]
    root_native = raw['root_qvel'][2:-2, :3]
    timewarp = {}
    for scope, js in [('selected_clip', selected_joints), ('all_100_trajectories', dataset_joints)]:
        timewarp[scope] = dict(
            conservative_p99_slowdown_factor=max(1., js['max_per_joint_p99_rad_s']/2),
            peak_slowdown_factor=max(1., js['peak_rad_s']/2),
            pooled_p99_over_2=js['pooled_abs_native_rad_s']['p99']/2,
            definition='t_target=factor*t_reference; qvel_target=qvel_reference/factor. Conservative P99 uses maximum of 42 individual P99, not pooled P99.')
    npz = BASE/'selected_reference.npz'
    np.savez_compressed(npz, **raw, foot_world_cm=feet,
        time_s=np.arange(len(feet))*dt,
        source_time_s=(chosen['start_index']+np.arange(len(feet)))*dt,
        timestep_seconds=np.array(dt), qpos_names=np.array(names), site_names=np.array(sites),
        leg_order=np.array(LEGS), selected_42_indices=np.array(indices),
        height_above_inferred_plane_cm=height, inferred_plane_z_abc=coef,
        near_plane_low_speed_candidate=stance,
        label=np.array('real_keypoint_fitted_reference; kinematic_inference_only; NOT simulated forces'),
        source_trajectory_id=np.array(chosen['trajectory_id']))
    report = dict(
        label='real_keypoint_fitted_reference', goal_stage='S2/S5 real-reference kinematic analysis only',
        source=dict(figshare_api='https://api.figshare.com/v2/articles/25309105',
            official_code='https://github.com/TuragaLab/flybody',
            paper='https://www.nature.com/articles/s41586-025-09029-4',
            hdf5=str(H5), hdf5_sha256=EXPECTED_SHA256,
            archive_md5='de7d5367850430adbe03a3783967c712', source_metadata=str(BASE/'SOURCE.json'),
            description='Official full-body trajectories fitted to real measured keypoints; fitted joint states are not all directly observed joint measurements.'),
        units_frames=dict(length='cm', time='s', joint_angle='rad', joint_velocity='rad/s',
            gravity_native=gravity, quaternion='wxyz, Hamilton active root-local to world rotation',
            root2site='root-local displacement; world_site=root_xyz+R(root_wxyz)*root2site',
            forward='initial root +X projected onto world XY and normalized; signed',
            source_evidence=[dict(path=str(OFFICIAL/'tasks/trajectory_loaders.py'), lines='241-253: root_qpos is concatenated unchanged; only initial root XY subtracted'),
                dict(path=str(OFFICIAL/'tasks/rewards.py'), lines='40-43,68-77: egocentric vectors compared to dataset root2site'),
                dict(path=str(OFFICIAL/'quaternions.py'), lines='20-35,114-168: scalar-first quaternion and local/world transform'),
                dict(path=str(OFFICIAL/'fruitfly/assets/fruitfly.xml'), lines='2-4: radians and gravity 0 0 -981; positive root X front leg anchors at 460 versus hind at 640'),
                dict(path=str(OFFICIAL/'tasks/arenas/hills.py'), lines='98-99: arena length explicitly in cm')],
            caveat='HDF5 does not itself provide unit attributes; cm is established by official loader/asset contract and explicit arena cm documentation, not by foot-scale guessing.'),
        schema=dict(trajectories=100, timestep_seconds=dt, qpos_columns=102,
            site_order=sites, leg_order=LEGS, duration_definition='(sample_count-1)*dt; excludes fictitious last interval'),
        selection=dict(method='Search full snippets of endpoint duration 0.5..1.0 s and 0.6 s windows every 25 samples plus final window; require positive forward >0.1 cm, yaw range <=10 deg, abs lateral/forward <=0.15 and >=2 complete swing-peak intervals on every leg. Rank net_forward/(1+4*lateral_ratio+yaw_range_deg/10)*min(1,min_cycles/3).',
            candidates=len(candidates), eligible=sum(c['eligible'] for c in candidates),
            selected=chosen, top10=ranked[:10]),
        verification=dict(hdf5_sha256_verified=True,
            max_quaternion_norm_error=float(max(norm_errors)),
            official_world_transform_max_abs_error_cm=transform_error,
            official_inverse_roundtrip_max_abs_error_cm=inverse_error,
            selected_qvel_vs_central_qpos_derivative_abs_error_rad_s=stats(np.abs(fd-native)),
            selected_qvel_derivative_correlation=float(np.corrcoef(fd.ravel(),native.ravel())[0,1]),
            root_linear_qvel_vs_world_position_derivative_abs_error_cm_s=stats(np.abs(root_fd-root_native)),
            note='Derivative comparison is diagnostic; native qvel is retained unchanged. Transform roundtrip verifies implementation against official convention, not independent reconstruction of experimental ground truth.'),
        foot_analysis=dict(method='Cadence and signed stride use consecutive smoothed height swing maxima. Five-sample smoothing, >=35 ms separation, local prominence >=max(0.003 cm,0.20*(P95-P05)), +/-40 ms neighborhood. Stride is world XY displacement projected on initial forward; no root-relative displacement substitution.',
            inferred_plane=dict(equation='z=a*x+b*y+c in cm; perpendicular signed distance retained, including negative residuals',
                abc=coef.tolist(), low_point_fit_rmse_cm=plane_rmse,
                method='Least-squares fit to each leg lowest 20% world-z samples; inferred lower envelope, not measured ground. Clearance peak-to-trough also reported to reduce plane-offset dependence.'),
            stance_candidate=dict(height_band_cm=[-.005,.005], max_world_speed_cm_s=stance_speed_threshold,
                rule='abs(height)<=0.005 cm AND world foot speed <= max(0.1,0.4*mean signed root forward speed) cm/s',
                caution='Kinematic near-plane low-speed candidate only; not measured contact, force, load, support, adhesion, friction or no-slip verification.'),
            per_leg=leg_metrics),
        joint_velocity=dict(selection_dofs_per_leg=DOFS,
            excluded='tarsus2/tarsus3/tarsus4/tarsus5 and all non-leg joints',
            selected_clip=selected_joints, all_100_trajectories=dataset_joints),
        retarget_boundary=dict(interface_speed_limit_rad_s=2., timewarp_candidates=timewarp,
            angle_mapping='NOT_ESTABLISHED', direct_42_joint_transfer='FORBIDDEN',
            caution='Homologous names do not establish axis directions, angle zero offsets, limits, morphology or target velocity equivalence. Timewarps are candidate reference slowdowns only; they establish neither physical feasibility nor target tracking. No force measurements or simulation/training performed.'),
        outputs=dict(script=str(Path(__file__).resolve()), json=str(BASE/'KINEMATICS.json'),
                     selected_reference=str(npz), selected_reference_sha256=digest(npz)))
    (BASE/'KINEMATICS.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(selected=chosen, per_leg={k:dict(cadence_hz=v['cadence_hz']['median'], stride_cm=v['signed_peak_to_peak_stride_cm']['median'], clearance_cm=v['clearance_peak_to_trough_cm']['median']) for k,v in leg_metrics.items()}, timewarps=timewarp, verification=report['verification'], outputs=report['outputs']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
