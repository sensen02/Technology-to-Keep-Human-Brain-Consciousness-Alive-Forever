"""S142/P1/S2-S4: ONE free full-body diagnostic, at most 10 ms.

Import is inert. Only main executes physics; no search, resume, or assistance.
Original S115 initialization is preserved byte-for-byte in private compiled source.
"""
from pathlib import Path
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import types

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT / 'run_extended_initialization_probe.py'
RECORDER = ROOT / 'run_constraint_row_recording.py'
OUTPUT_PARENT = Path('/home/sensen/Desktop/engine修复验收/s142-free-physical-prewarm')
THRESHOLDS = dict(height_ratio_min=.85, angle_rad_max=.35, contact_legs_min=3,
                  load_ratio_min=.8, nonfoot_depth_max=1e-12, limit_excess_max=1e-6)

# Replace only original acquisition loop + fixed-length reporting tail. All
# construction, initialization and sample() code above this boundary is intact.
TAIL = '''            rows = [sample()]; currents,spikes,features,commands,schedule = [],[],[],[],[]
            stop_reasons = s142_reasons(rows[-1], q, m)
            refresh_times = []
            for tick in range(TICKS):
                if stop_reasons: break
                if tick % 10 == 0:
                    observed = observe_quantities(m,d,sensory_ref,joints,geom_leg)
                    current, _ = enc.encode(observed)
                    refresh_times.append(float(d.time))
                    atomic_json(dest/f'OBSERVATION_{tick:02d}.json',raw_observation_record(observed)[0])
                currents.append(current.copy()); reserve(neural_ticks=1)
                sp = net.step(current_nA=current).copy()
                start,end = (PRE+tick)*.0005,(PRE+tick+1)*.0005
                u,info = dec.consume(sp[None,:],spike_dt_s=.0005,start_s=start,end_s=end)
                assert abs(net.t_ms/1000-end) < 1e-12 and dec.interval == PRE+tick
                assert abs(d.time-tick*.0005) < 1e-12
                spikes.append(sp); features.append(info['pool_features']); commands.append(u.copy())
                schedule.append(dict(tick=tick,source_start_s=start,source_end_s=end,
                    physical_start_s=float(d.time),planned_physical_end_s=(tick+1)*.0005,
                    source_offset_s=.08,consume_id=dec.interval))
                d.ctrl[:] = u
                for sub in range(substeps):
                    require_no_callbacks(callback_snapshot(mj))
                    assert not d.qfrc_applied.any() and not d.xfrc_applied.any()
                    reserve(step=1,forward=1); mj.mj_step(m,d); mj.mj_forward(m,d)
                    row = sample(); rows.append(row)
                    stop_reasons = s142_reasons(row, q, m)
                    if stop_reasons: break
                schedule[-1]['actual_physical_end_s'] = float(d.time)
            arrays = stack_rows(rows)
            arrays.update(spikes=np.asarray(spikes),features=np.asarray(features),
                applied=np.asarray(commands),decoded=np.asarray(commands),current=np.asarray(currents),
                weights=arm_weights,actuator_pool=dec.actuator_pool.copy())
            weight_native = float(m.body_mass.sum()*abs(m.opt.gravity[2]))
            load = arrays['foot_force'][:,:,2].sum(1)/weight_native
            quat = arrays['qpos'][:,3:7]; norms = np.linalg.norm(quat,axis=1)
            angle = 2*np.arccos(np.clip(np.abs(quat@quat[0])/(norms*norms[0]),0,1))
            gates = np.column_stack((arrays['qpos'][:,2]/q[2] >= .85,angle <= .35,
                np.count_nonzero(arrays['contacts'] > 0,axis=1) >= 3,load >= .8,
                arrays['nonfoot_depth'] <= 1e-12,arrays['limit_excess'] <= 1e-6))
            arrays.update(six_gates=gates,all_six_gates=gates.all(1),load_ratio=load,root_angle_rad=angle)
            # Actual-time trapezoid impulse, including stop sample; NOT work or elastic energy.
            for force_key, output_key in [('external_force_world','external_impulse_world'),
                                         ('contact_force_world','contact_impulse_world')]:
                f = arrays[force_key]
                arrays[output_key] = np.vstack([np.zeros(3),np.cumsum(
                    .5*(f[1:]+f[:-1])*np.diff(arrays['time'])[:,None],axis=0)])
            arrays['com_momentum_change_world'] = arrays['com_momentum_world']-arrays['com_momentum_world'][0]
            atomic_npz(dest/'TRAJECTORY.npz',**arrays)
            atomic_json(dest/'SCHEDULE.json',schedule)
            atomic_json(dest/'NEURAL_FINAL_META.json',neural_meta(net))
            atomic_npz(dest/'NEURAL_FINAL.npz',**{k:v.copy() for k,v in vars(net).items() if isinstance(v,np.ndarray)},
                FIR_history=dec.history,release_features=release_features,release_decoded=release_ctrl)
            atomic_npz(dest/'STOP_FULL_STATE.npz',**rows[-1])
            reason = stop_reasons or ['MAX_DURATION_10MS']
            assert ledger['step'] <= 400 and ledger['model_load'] == 1
            assert len(rows) == ledger['step']+1 and len(s142_records) == len(rows)
            assert numeric_model_snapshot(m) == after
            record = dict(label=label,status='STOPPED' if stop_reasons else 'MAX_DURATION_REACHED',
                stop_reasons=reason,physical_final_s=float(d.time),actual_duration_ms=float(d.time)*1000,
                requested_max_duration_ms=10,samples=len(rows),calls=ledger['per_arm'][label].copy(),
                initialization_ticks=PRE,initialization_ms=80,initialization_is_physical_prewarm=False,
                actual_sensory_observation_times_s=refresh_times,
                sensory_5ms_refresh_reached=any(t >= .005-1e-12 for t in refresh_times),
                physical_prewarm_accepted=False,behavior_claim='NONE',physical_units='NOT_ESTABLISHED',
                neural_final_s=net.t_ms/1000,FIR_final_s=dec.elapsed_s,
                closure_status='NOT_EVALUATED: original fixed-121 reporting not applicable; offline parent verification required',
                trajectory_sha256=sha(dest/'TRAJECTORY.npz'))
            atomic_json(dest/'STOP.json',record)
            hashes = {str(f):sha(f) for f in dest.iterdir() if f.is_file()}
            record['hashes'] = hashes
            atomic_json(dest/'RESULT.json',record); results.append(record)
            atomic_json(out/'RESULTS.json',results)
            ledger['completed'].append(label); ledger['active'] = None; reserve()
        assert identities == {str(s):sha(s) for s in sources}
        atomic_json(out/'COMPLETE.json',dict(status='DIAGNOSTIC_RECORDED',ledger=ledger,
            source_chain_hashes=identities,physical_units='NOT_ESTABLISHED',
            physical_prewarm_accepted=False,behavior_claim='NONE',selection=False,deployment=False))
        print(json.dumps(results))
'''


def private_source(text):
    """Exact, count-checked textual changes; no disk edits to old sources."""
    start = '            rows = [sample()]; currents,spikes,features,commands,schedule = [],[],[],[],[]\n'
    end = '    except BaseException:\n'
    assert text.count(start) == text.count(end) == 1
    old_tail = text[text.index(start):text.index(end)]
    changes = [
        ('PRE, TICKS = 160, 6', 'PRE, TICKS = 160, 20'),
        ("EXPECTED_PER_ARM = {'init80ms': dict(model_load=1, forward=121, step=120, neural_ticks=166)}",
         "EXPECTED_PER_ARM = {'init80ms': dict(model_load=1, forward=401, step=400, neural_ticks=180)}"),
        ('EXPECTED = dict(model_load=1, forward=121, step=120, neural_ticks=166)',
         'EXPECTED = dict(model_load=1, forward=401, step=400, neural_ticks=180)'),
        (old_tail, TAIL)]
    for old,new in changes:
        assert text.count(old) == 1
        text = text.replace(old,new,1)
    ast.parse(text)
    return text, [dict(old=old,new=new) for old,new in changes]


def stop_reasons(row, initial_q, m):
    import numpy as np
    reasons = []
    if not all(np.isfinite(v).all() for v in row.values()): reasons.append('NONFINITE_SAMPLE')
    if np.asarray(row['warnings']).any(): reasons.append('MUJOCO_WARNING')
    if np.asarray(row['qfrc_applied']).any() or np.asarray(row['xfrc_applied']).any():
        reasons.append('EXTERNAL_LOAD_GUARD')
    quat = row['qpos'][3:7]; q0 = initial_q[3:7]
    angle = 2*np.arccos(np.clip(abs(quat@q0)/(np.linalg.norm(quat)*np.linalg.norm(q0)),0,1))
    tests = [
        ('HEIGHT_RATIO_LT_0.85',row['qpos'][2]/initial_q[2] < .85),
        ('ROOT_ANGLE_GT_0.35_RAD',angle > .35),
        ('CONTACT_LEGS_LT_3',np.count_nonzero(row['contacts'] > 0) < 3),
        ('LOAD_RATIO_LT_0.8',row['foot_force'][:,2].sum()/(m.body_mass.sum()*abs(m.opt.gravity[2])) < .8),
        ('NONFOOT_DEPTH_GT_1E-12',row['nonfoot_depth'] > 1e-12),
        ('LIMIT_EXCESS_GT_1E-6',row['limit_excess'] > 1e-6)]
    return reasons + [name for name,failed in tests if failed]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,help='fresh absolute path; default real UTC under S142 desktop parent')
    args = p.parse_args()
    out = args.output or OUTPUT_PARENT/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if not out.is_absolute() or out.exists(): raise ValueError('fresh absolute --output required; no resume')
    # Imports are deliberately inside main: no model construction on import.
    import numpy as np
    import run_constraint_row_recording as recorder
    source, changes = private_source(ORIGINAL.read_text())
    module = types.ModuleType('_s142_private_s115')
    module.__file__ = str(ORIGINAL)
    exec(compile(source,str(ORIGINAL),'exec'),module.__dict__)
    module.OUTPUT_PARENT = out
    save_json,save_npz = module.atomic_json,module.atomic_npz
    records,metadata,live = [],{},{}
    module.s142_records = records
    module.s142_reasons = stop_reasons
    old_capture = module.com_contact_record
    commitments = {str(f):module.sha(f) for f in (Path(__file__).resolve(),ORIGINAL,RECORDER)}
    private_digest = hashlib.sha256(source.encode()).hexdigest()
    def capture(mj,m,d,floors):
        live.update(m=m,d=d)
        result = old_capture(mj,m,d,floors)
        index = len(records)
        constraint = recorder.constraint_record(mj,m,d,index)
        # S121 per-record copy has no 121-sample assumption; pack dynamically here.
        records.append(constraint)
        if not metadata:
            metadata.update(recorder.constraint_metadata(mj,m))
            metadata.update(schema='dynamic n samples; sampleNNN_ records; row_offsets + flat efc arrays',
                clocks='PRE160=.08s; 20 substeps/tick; stop may be mid-tick; source interval may lead physical stop',
                sampling='after original COM recorder at every actual forward sample; no added forward/step')
            model_arrays = {name:np.asarray(getattr(m,name)).copy() for name in
                ('body_mass','body_inertia','body_ipos','body_iquat','dof_damping','dof_frictionloss',
                 'jnt_stiffness','qpos_spring','actuator_gainprm','actuator_biasprm','actuator_dynprm',
                 'actuator_lengthrange','actuator_acc0','actuator_length0','actuator_actadr',
                 'actuator_actnum','actuator_gaintype','actuator_biastype','actuator_dyntype')}
            save_npz(out/'init80ms/NATIVE_MODEL_ARRAYS.npz',**model_arrays)
        result.update(root_vz=np.asarray(d.qvel[2]),native_mass_packed=d.qM.copy(),
            native_qacc_warmstart=d.qacc_warmstart.copy(),native_energy=d.energy.copy(),
            signed_power_passive=np.asarray(d.qfrc_passive@d.qvel),
            signed_power_actuator=np.asarray(d.qfrc_actuator@d.qvel),
            signed_power_constraint=np.asarray(d.qfrc_constraint@d.qvel),
            signed_power_muscle_by_actuator=d.actuator_force*d.actuator_velocity)
        types_ = constraint['efc_type']
        contact_types = [int(v) for k,v in mj.mjtConstraint.__members__.items() if k.startswith('mjCNSTR_CONTACT_')]
        contact_mask = np.isin(types_,contact_types)
        contact_qfrc = constraint['efc_J_dense'][contact_mask].T@constraint['efc_force'][contact_mask]
        result.update(qfrc_contact_rows=contact_qfrc,
            signed_power_contact_rows=np.asarray(contact_qfrc@d.qvel),
            signed_power_noncontact_constraint=np.asarray((d.qfrc_constraint-contact_qfrc)@d.qvel))
        return result
    def json_hook(path,value):
        path = Path(path)
        if path.name == 'INPUT_HASHES.json':
            value = dict(value,**commitments)
            frozen = out/'SOURCE_FREEZE'; frozen.mkdir()
            for f in (Path(__file__).resolve(),ORIGINAL,RECORDER):
                (frozen/f.name).write_bytes(f.read_bytes())
            (frozen/'PRIVATE_RUNNER_SOURCE.py').write_text(source)
            save_json(frozen/'EXACT_CHANGES.json',changes)
            save_json(frozen/'HASHES.json',dict(input_hashes=commitments,private_runner_sha256=private_digest))
        elif path.name == 'PREREG.json':
            value = dict(value,source_hashes=dict(value['source_hashes'],**commitments),
                protocol='S142 ONE bounded free physical diagnostic; no assistance or activation assignment',
                physical_duration_ms=10,physical_steps_per_arm={'init80ms':400},
                physical_samples_per_arm={'init80ms':401},physical_neural_ticks_per_arm=20,
                physical_counts_are_maxima=True,clocks='PRE160 neural-only 80ms; body0..<=10ms, source80..<=90ms',
                initialization_duration_intervention_ms=0,initialization_is_physical_prewarm=False,
                selection_rule='NONE; one diagnostic only; no parameter search or checkpoint replacement',
                early_window_s=[0,.010],early_aggregation='NOT_USED: no optimization or admission score',
                closure_criteria='NOT_EVALUATED online; parent offline verification required',
                original_actual_hard_guards=['no callbacks','zero applied loads','finite samples','zero MuJoCo warnings'],
                original_six_gates_were_posthoc_not_hardstops=True,
                added_conservative_hard_stop_thresholds=THRESHOLDS,
                stop_policy='check release and every25us sample; stop first failure and retain full failing sample; no continuation',
                sensory='live observe_quantities at physical0 and5ms only if reached; no replay; hold current between',
                feedback_acceptance='if stop before5ms, no refreshed physical feedback evidence; not accepted',
                fixed_source_note='private source and exact replacements archived before model load',
                private_source_sha256=private_digest,physical_prewarm_accepted=False,
                energy_note='native d.energy copied but energy flag unchanged; NOT validated total or elastic potential; no elastic-energy claim',
                powers='native force dot native velocity; actuator force*transmission velocity; contact uses actual efc contact rows; NOT work; physical units NOT_ESTABLISHED',
                decomposition='qfrc_passive joint/tendon/fluid aggregate is NOT passive muscle; actuator includes native active/passive muscle terms, not split',
                retained_reference_is_context_only=True)
        elif path.name == 'FAILED.json':
            if live:
                d = live['d']
                save_npz(out/'EMERGENCY_STATE.npz',**{name:np.asarray(getattr(d,name)).copy() for name in
                    ('qpos','qvel','qacc','act','ctrl','qacc_warmstart','qfrc_actuator','qfrc_passive','qfrc_constraint','qfrc_applied','xfrc_applied')},time=np.asarray(d.time))
                value = dict(value,actual_duration_ms=float(d.time)*1000,stop_reason='EXCEPTION_OR_ORIGINAL_HARD_GUARD',
                    emergency_state='EMERGENCY_STATE.npz',captured_constraint_samples=len(records))
                if records: save_records(out/'EMERGENCY_CONSTRAINT_ROWS.npz')
        elif path.name == 'COMPLETE.json':
            assert commitments == {name:module.sha(Path(name)) for name in commitments}
            value = dict(value,source_chain_hashes=dict(value['source_chain_hashes'],**commitments),private_source_sha256=private_digest)
        return save_json(path,value)
    def save_records(path):
        packed = {name:np.asarray([r[name] for r in records]) for name in
            ('time','sample_index','physical_step','physical_tick','source_start_s','source_end_s','nv','nefc','ncon','is_sparse','nJ')}
        packed['row_offsets'] = np.r_[0,np.cumsum(packed['nefc'],dtype=np.int64)]
        for name in ('efc_type','efc_id','efc_force'): packed[name] = np.concatenate([r[name] for r in records])
        for i,row in enumerate(records): packed.update({f'sample{i:03d}_{k}':v for k,v in row.items()})
        save_npz(path,**packed)
    def npz_hook(path,**arrays):
        path = Path(path)
        if path.name == 'TRAJECTORY.npz':
            assert np.array_equal(arrays['time'],np.asarray([r['time'] for r in records]))
            save_records(path.parent/'CONSTRAINT_ROWS.npz')
            save_json(path.parent/'CONSTRAINT_ROW_METADATA.json',metadata)
        return save_npz(path,**arrays)
    module.com_contact_record = capture
    module.atomic_json,module.atomic_npz = json_hook,npz_hook
    # Original parser sees only a fresh absolute output and performs one arm.
    import sys
    previous = sys.argv
    try:
        sys.argv = [str(Path(__file__).resolve()),'--output',str(out)]
        module.main()
    finally:
        sys.argv = previous


if __name__ == '__main__':
    main()
