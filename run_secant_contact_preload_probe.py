"""S145/P1: exactly one STATIC secant-offset complete-body forward.

Inert import. No endpoint reruns, integration, neural ticks or dynamic invocation.
The full-precision actual S144 NPZ COM accelerations determine the sole offset.
Native physical units NOT_ESTABLISHED; static balance is not support success.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json

ROOT = Path(__file__).resolve().parent
OUTPUT_PARENT = Path('/home/sensen/Desktop/engine修复验收/s145-secant-contact-preload')
S143 = Path('/home/sensen/Desktop/engine修复验收/s143-initial-contact-preload/20261024T202354366515Z')
S144 = Path('/home/sensen/Desktop/engine修复验收/s144-downward-contact-preload/20261024T203038384807Z')
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
THRESHOLDS = dict(height_ratio_min=.85, angle_rad_max=.35, contact_legs_min=3,
                  load_ratio_min=.8, nonfoot_depth_max=1e-12, limit_excess_max=1e-6)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='auto', help='auto or fresh absolute directory')
    args = p.parse_args()
    out = (OUTPUT_PARENT/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
           if args.output == 'auto' else Path(args.output))
    if not out.is_absolute() or out.exists():
        raise ValueError('fresh absolute output required; no resume')
    import numpy as np
    import mujoco as mj
    from run_native_rk4_gate_training import (sha, atomic_json, atomic_npz,
        callback_snapshot, require_no_callbacks, numeric_model_snapshot)
    from run_native_rk4_tick_consumption import MODEL, INITIAL
    from run_neural_initialized_startup import WEIGHTS
    from run_initialized_translation_training import mesh_dependencies
    from run_com_contact_probe import com_contact_record
    from run_constraint_row_recording import constraint_record, constraint_metadata
    from run_local_pool_response import moment_record
    if mj.__version__ != '3.9.0':
        raise ValueError('requires MuJoCo 3.9.0')
    prior = json.loads((S144/'RESULT.json').read_text())
    prior143 = json.loads((S143/'RESULT.json').read_text())
    proofs = [S143/'independent/STATIC_PRELOAD_INDEPENDENT_CHECK.json',
              S143/'independent/NATIVE_SELECTION_CHECK.json',
              S144/'DOWNWARD_STATIC_INDEPENDENT_CHECK.json']
    for path in proofs:
        proof = json.loads(path.read_text())
        assert proof['passed'] is True, path
        for filename, expected in proof.get('hashes', {}).items():
            assert sha(Path(filename)) == expected, filename
    for directory, report in ((S143, prior143), (S144, prior)):
        assert report['status'] == 'STATIC_COMPLETE' and report['dynamic_executed'] is False
        for filename, expected in json.loads((directory/'OUTPUT_HASHES.json').read_text()).items():
            assert sha(Path(filename)) == expected, filename
        for filename, expected in report['source_hashes'].items():
            assert sha(Path(filename)) == expected, filename
    ddepth = float(prior143['d'])
    assert np.isfinite(ddepth) and ddepth > 0 and ddepth == prior['d']
    with np.load(INITIAL, allow_pickle=False) as z:
        original_q = z['new_qpos'].copy()
        sensory_reference_q = z['sensory_reference_qpos'].copy()
    endpoints = []
    for index in (0, 1):
        filename = S144/f'STATIC_{index}.npz'
        assert sha(filename) == prior['results'][index]['snapshot_sha256']
        with np.load(filename, allow_pickle=False) as z:
            assert all(np.isfinite(z[k]).all() for k in z.files)
            assert not z['warnings'].any() and z['six_gates'].all()
            assert not any(z[k].any() for k in ('qvel','act','ctrl','qfrc_applied','xfrc_applied'))
            offset = float(z['dz'])
            expected = original_q.copy(); expected[2] += offset
            assert np.array_equal(z['qpos'], expected) and float(z['time']) == 0
            az = float(z['com_acceleration_world'][2])
            assert offset == (0. if index == 0 else -ddepth/4)
            assert az == prior['results'][index]['com_az']
            assert prior['results'][index]['eligible'] and prior['results'][index]['reliable']
            residual = float(np.max(np.abs(z['com_mass_times_acceleration_world']-z['external_force_world'])))
            scale = max(float(np.max(np.abs(z['com_mass_times_acceleration_world']))), float(np.max(np.abs(z['external_force_world']))), 1e-12)
            assert residual <= 1e-8 and residual/scale <= 1e-6
            endpoints.append(dict(index=index, dz=offset, com_az=az, snapshot=str(filename), sha256=sha(filename)))
    z0, z1 = (e['dz'] for e in endpoints)
    a0, a1 = (e['com_az'] for e in endpoints)
    denominator = a1-a0
    # Stop before any model call; there is no fallback offset or retry.
    assert np.isfinite([z0,z1,a0,a1,denominator]).all() and denominator != 0
    zstar = z0-a0*(z1-z0)/denominator
    assert np.isfinite(zstar) and min(z0,z1) < zstar < max(z0,z1)
    factor = zstar/ddepth
    calculation = dict(formula='zstar=z0-a0*(z1-z0)/(a1-a0)', endpoints=endpoints,
        z0=z0,z1=z1,a0=a0,a1=a1,denominator=denominator,zstar=zstar,
        static_factor=factor,static_factor_definition='zstar/d',
        d=ddepth,d_source=str(S143/'RESULT.json'),strictly_between_endpoints=True,
        float_hex={k:float(v).hex() for k,v in dict(z0=z0,z1=z1,a0=a0,a1=a1,zstar=zstar,static_factor=factor).items()})
    sources = list(dict.fromkeys([MODEL, INITIAL, WEIGHTS, Path(__file__).resolve(),
        ROOT/'run_downward_contact_preload_probe.py', ROOT/'verify_downward_contact_preload_probe.py',
        ROOT/'run_initial_contact_preload_probe.py', ROOT/'verify_initial_contact_preload_probe.py',
        ROOT/'verify_static_preload_native.py', *proofs,
        *[directory/name for directory in (S143,S144) for name in
          ('RESULT.json','PREREG.json','OUTPUT_HASHES.json','STATIC_0.npz','STATIC_1.npz','DEPTH_MEASUREMENT.json')],
        ROOT/'run_free_physical_prewarm_probe.py', ROOT/'run_extended_initialization_probe.py',
        ROOT/'run_com_contact_probe.py', ROOT/'run_constraint_row_recording.py',
        ROOT/'run_local_pool_response.py', ROOT/'run_native_rk4_gate_training.py',
        ROOT/'run_native_rk4_tick_consumption.py', ROOT/'run_neural_initialized_startup.py',
        ROOT/'run_initialized_translation_training.py', *ROOT.glob('engine/**/*.py'),
        *mesh_dependencies()]))
    identities = {str(s): sha(s) for s in sources}
    out.mkdir(parents=True, exist_ok=False)
    freeze = out/'SOURCE_FREEZE'; freeze.mkdir()
    for i, source in enumerate(sources):
        if source.suffix in ('.py', '.xml', '.json'):
            (freeze/f'{i:04d}_{source.name}').write_bytes(source.read_bytes())
    atomic_json(out/'INPUT_HASHES.json', identities)
    atomic_json(out/'SECANT_CALCULATION.json', calculation)
    atomic_json(out/'PREREG.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        protocol='S145 STATIC ONLY; exactly one secant state; no endpoint reruns or other search',
        model=str(MODEL), initial_pose=str(INITIAL), weights_unchanged=str(WEIGHTS),
        prior_S143=str(S143), prior_S144=str(S144), d=ddepth, d_source=str(S143/'RESULT.json'),
        prior_independent_proofs=list(map(str,proofs)),secant=calculation,
        static_factor=factor,dz=zstar,
        original_root_z=float(original_q[2]), original_root_quaternion=original_q[3:7].tolist(),
        offsets='only full-precision zstar; root qpos[2] only; no rounded factor used',
        gate_names=['height','angle','legs','load','nonfoot','limit'], six_gate_thresholds=THRESHOLDS,
        legs_gate='ORIGINAL gate: >=3 legs with actual negative-distance floor-foot contacts; NOT bearing count',
        reference='height ORIGINAL rootz, angle ORIGINAL quaternion, never per-offset rebaseline',
        reliability='all arrays finite, zero warnings/callbacks/external loads, COM closure abs<=1e-8 AND relative<=1e-6',
        selection='all original six gates AND reliable AND abs(actual native COM az)<abs(exact old STATIC_0 COM az)',
        expected_calls=dict(model_load=1,fresh_data=1,forward=1,step=0,neural_ticks=0),
        early_stop='nonfinite secant or not strictly between old endpoints: stop before model; no fallback',
        model_edits=[],qvel_act_ctrl='fresh MjData zeros',independent_static_verification_required_before_dynamic=True,
        dynamic_file_created=False,physical_units='NOT_ESTABLISHED',behavior_claim='NONE',
        initial_balance_is_not_support_success=True,S119_same_condition_candidate=False,source_hashes=identities))
    atomic_npz(out/'ORIGINAL_INPUT.npz',qpos=original_q,sensory_reference_qpos=sensory_reference_q)
    ledger = dict(model_load=0,forward=0,step=0,neural_ticks=0,fresh_data=0)
    def reserve(**delta):
        for k,v in delta.items(): ledger[k] += v
        atomic_json(out/'CALL_LEDGER.json',ledger)
    reserve()
    results = []
    try:
        require_no_callbacks(callback_snapshot(mj))
        reserve(model_load=1)
        m = mj.MjModel.from_xml_path(str(MODEL))
        assert (m.nq,m.nv,m.nu,m.nbody,m.nmesh) == (49,48,90,73,71)
        before = numeric_model_snapshot(m)
        model_arrays = {}
        for prefix,obj in [('model',m),('opt',m.opt)]:
            for name in dir(obj):
                if not name.startswith('_'):
                    value = getattr(obj,name)
                    if isinstance(value,np.ndarray): model_arrays[prefix+'__'+name] = value.copy()
        atomic_npz(out/'NATIVE_MODEL_ARRAYS.npz',**model_arrays)
        atomic_json(out/'MODEL_NUMERIC_BEFORE.json',before)
        floors = set(map(int,np.flatnonzero(m.geom_type == mj.mjtGeom.mjGEOM_PLANE)))
        floor = mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'floor')
        assert floor >= 0 and floors == {floor}
        geom_leg = {}
        for g in range(m.ngeom):
            gn = (mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g) or '').lower()
            bn = (mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g])) or '').lower()
            for leg in LEGS:
                if 'tars' in gn+bn and (gn.startswith(leg) or bn.startswith(leg)):
                    geom_leg[g] = leg
        metadata = constraint_metadata(mj,m)
        metadata.update(foot_order=list(LEGS),geom_leg={str(k):v for k,v in geom_leg.items()},
            floors=sorted(floors),gravity=m.opt.gravity.tolist(),
            clocks='STATIC_SECANT fresh MjData physical time 0; constraint source-clock not dynamic',
            acceleration_method='S112 native mj_jacBodyCom@qacc + mj_jacDot@qvel; direct objects retained',
            normal_force='contact_external_sign * frame[0] * raw6[0]; full includes friction')
        atomic_json(out/'METADATA.json',metadata)
        limited = np.flatnonzero(m.jnt_limited); qa = m.jnt_qposadr[limited]
        weight = float(m.body_mass.sum()*abs(m.opt.gravity[2])); assert weight > 0
        index = 0
        dz = zstar
        q = original_q.copy(); q[2] += dz
        reserve(fresh_data=1)
        data = mj.MjData(m); data.qpos[:] = q
        require_no_callbacks(callback_snapshot(mj))
        assert not any(a.any() for a in (data.qvel,data.act,data.ctrl,data.qfrc_applied,data.xfrc_applied))
        reserve(forward=1); mj.mj_forward(m,data)
        assert data.time == 0 and np.array_equal(data.qpos,q)
        assert np.array_equal(data.qpos[np.arange(m.nq)!=2],original_q[np.arange(m.nq)!=2])
        assert not any(a.any() for a in (data.qvel,data.act,data.ctrl,data.qfrc_applied,data.xfrc_applied))
        row = com_contact_record(mj,m,data,floors)
        row.update({'constraint__'+k:v for k,v in constraint_record(mj,m,data,index).items()})
        row.update(moment_record(m,data))
        for name in ('qpos','qvel','act','ctrl','qacc','qM','qacc_warmstart','qfrc_actuator',
                     'qfrc_passive','qfrc_constraint','qfrc_bias','qfrc_applied','xfrc_applied',
                     'actuator_force','actuator_length','actuator_velocity','energy'):
            row[name] = np.asarray(getattr(data,name)).copy()
        Ma = np.zeros(m.nv); mj.mj_mulM(m,data,Ma,data.qacc)
        row.update(Ma=Ma,RHS=data.qfrc_actuator+data.qfrc_passive+data.qfrc_constraint-data.qfrc_bias,
                   time=np.asarray(data.time),dz=np.asarray(dz),original_qpos=original_q.copy(),
                   warnings=np.asarray([w.number for w in data.warning]))
        n = int(data.ncon); bearing = np.zeros(n,dtype=bool)
        normal_world = np.zeros((n,3)); foot_index = np.full(n,-1,dtype=int)
        foot_force = np.zeros((6,3)); negative_counts = np.zeros(6,dtype=int)
        nonfoot_depth = 0.
        for ci,c in enumerate(data.contact):
            g1,g2 = int(c.geom1),int(c.geom2)
            if floor not in (g1,g2): continue
            foot = g2 if g1 == floor else g1
            sign = 1. if g1 == floor else -1.
            if foot not in geom_leg:
                nonfoot_depth = max(nonfoot_depth,-float(c.dist)); continue
            li = LEGS.index(geom_leg[foot]); foot_index[ci] = li
            normal_world[ci] = sign*row['contact_frame'][ci,0]*row['contact_force_raw6'][ci,0]
            bearing[ci] = normal_world[ci,2] > 0
            foot_force[li] += sign*row['contact_force_on_geom2_world'][ci]
            negative_counts[li] += int(c.dist < 0)
        bearing_counts = np.asarray([np.count_nonzero(bearing & (foot_index==li)) for li in range(6)])
        excess = np.maximum(np.maximum(m.jnt_range[limited,0]-q[qa],q[qa]-m.jnt_range[limited,1]),0)
        angle = float(2*np.arccos(np.clip(abs(q[3:7]@original_q[3:7])/(np.linalg.norm(q[3:7])*np.linalg.norm(original_q[3:7])),0,1)))
        load = float(foot_force[:,2].sum()/weight)
        gates = np.asarray([q[2]/original_q[2]>=.85,angle<=.35,
            np.count_nonzero(negative_counts)>2,load>=.8,nonfoot_depth<=1e-12,excess.max(initial=0)<=1e-6])
        residual = float(np.max(np.abs(row['com_mass_times_acceleration_world']-row['external_force_world'])))
        scale = max(float(np.max(np.abs(row['com_mass_times_acceleration_world']))),float(np.max(np.abs(row['external_force_world']))),1e-12)
        row.update(bearing_contact_mask=bearing,contact_normal_force_on_foot_world=normal_world,
            contact_foot_leg_index=foot_index,bearing_contacts=bearing_counts,contacts=negative_counts,
            foot_force=foot_force,load_ratio=np.asarray(load),nonfoot_depth=np.asarray(nonfoot_depth),
            limit_excess=np.asarray(excess.max(initial=0)),joint_limit_excess=excess,
            root_angle_rad=np.asarray(angle),height_ratio=np.asarray(q[2]/original_q[2]),six_gates=gates,
            com_force_residual=np.asarray(residual))
        reliable = bool(all(np.isfinite(v).all() for v in row.values()) and not row['warnings'].any()
                        and residual<=1e-8 and residual/scale<=1e-6)
        require_no_callbacks(callback_snapshot(mj))
        assert before == numeric_model_snapshot(m)
        atomic_npz(out/'STATIC_SECANT.npz',**row)
        result = dict(index='SECANT',dz=float(dz),six_gates=gates.tolist(),reliable=reliable,
            eligible=bool(gates.all() and reliable),load_ratio=load,
            com_az=float(row['com_acceleration_world'][2]),com_force_residual=residual,
            snapshot_sha256=sha(out/'STATIC_SECANT.npz'))
        results.append(result); atomic_json(out/'STATIC_RESULTS.json',results)
        strict = bool(result['eligible'] and abs(result['com_az']) < abs(a0))
        selected = result if strict else prior['results'][0]
        assert identities == {str(s):sha(s) for s in sources}
        assert before == numeric_model_snapshot(m)
        atomic_json(out/'MODEL_NUMERIC_AFTER.json',numeric_model_snapshot(m))
        report = dict(status='STATIC_COMPLETE',d=ddepth,dz=zstar,static_factor=factor,secant=calculation,
            results=results,selected=selected,zero_valid=True,zero_reference=prior['results'][0],
            zero_reference_snapshot=str(S144/'STATIC_0.npz'),nonzero_strict_improvement=strict,
            dynamic_eligible_pending_independent_verifier=strict,dynamic_executed=False,
            independent_gate_passed=False,ledger=ledger,physical_units='NOT_ESTABLISHED',
            behavior_claim='NONE',support_success=False,S119_same_condition_candidate=False,source_hashes=identities)
        atomic_json(out/'RESULT.json',report)
        atomic_json(out/'OUTPUT_HASHES.json',{str(f):sha(f) for f in out.rglob('*') if f.is_file()})
        print(json.dumps(dict(output=str(out),selected=selected,dynamic_eligible_pending_independent_verifier=strict,ledger=ledger)))
        return report
    except BaseException as exc:
        atomic_json(out/'FAILED.json',dict(error=repr(exc),ledger=ledger,completed_static=len(results)))
        raise


if __name__ == '__main__':
    main()
