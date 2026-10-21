"""One frozen 550ms MN/body arm; requires a separate human-scope receipt.

Preparation/tests do NOT execute this script. No replay, push, fitting or added
motor drive. Each invocation runs only one motor period in a fresh directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import run_neural_muscle_closure as closure
from engine.embodied.motor_cadence_identity import require_frozen_inputs, sha256
from run_motor_body_integration import require_legal_initial_pose, BodyObserver
from run_motor_cadence_single import ROOT, BATCH, WEIGHTS, POSE, BODY_DT, json_save

DURATION_S = .55
BODY_STEPS = 88000
NEURAL_TICKS = 1100


def require_scope(out, period_ms, authorization):
    """Reject every 550ms run unless a future direct-human scope is recorded.

    The receipt must be externally supplied after the human confirms this exact
    two-arm scope; this entry never generates or modifies one.
    """
    if type(period_ms) is not int or period_ms not in (1, 5):
        raise ValueError('motor period must be exactly 1 or 5ms')
    out = Path(out).resolve()
    if not out.is_dir() or {p.name for p in out.iterdir()} != {'PREREGISTRATION.md'}:
        raise ValueError('fresh output containing only PREREGISTRATION.md required')
    authorization = Path(authorization).resolve()
    if not authorization.is_file() or authorization.parent == out:
        raise ValueError('separate human-scope receipt required')
    scope = json.loads(authorization.read_text())
    if (scope.get('authority') != 'direct-human' or
        scope.get('objective') != 'two frozen motor cadence arms' or
        scope.get('duration_s_per_arm') != DURATION_S or
        scope.get('maximum_body_steps_total') != 2*BODY_STEPS or
        sorted(scope.get('motor_periods_ms', [])) != [1,5] or
        not isinstance(scope.get('human_confirmation'), str) or
        not scope['human_confirmation'].strip() or
        not isinstance(scope.get('preregistration_sha256'), str) or
        scope['preregistration_sha256'] != sha256(out/'PREREGISTRATION.md')):
        raise ValueError('human-scope receipt differs from registered run')
    return out, authorization, scope


def verify_observations(evidence, commands, steps, result, period_ms, n_neurons):
    """Validate actual source/command chronology; returns common numeric arrays."""
    intervals = 110 if period_ms == 5 else 550
    if len(steps) != BODY_STEPS or len(commands) != intervals:
        raise AssertionError('body or motor window count mismatch')
    body = np.asarray(steps)
    np.testing.assert_allclose(body[:,0], np.arange(1,BODY_STEPS+1)*BODY_DT,
                               rtol=0, atol=1e-9)
    controls = np.asarray(commands)
    if np.any(controls[0]) or not np.isfinite(body).all() or not np.isfinite(controls).all():
        raise AssertionError('first command nonzero or nonfinite observations')
    sensory_time = np.arange(110)*.005
    if period_ms == 1:
        start=np.asarray(evidence['motor_start_s']); end=np.asarray(evidence['motor_end_s'])
        applied=np.asarray(evidence['motor_applied_controls'])
        decoded=np.asarray(evidence['motor_decoded_next_controls'])
        sample=np.asarray(evidence['sensory_sample_start_s'])
        current=np.asarray(evidence['sensory_current_nA'])
        activation=np.asarray(evidence['motor_activation_end'])
        sources=np.asarray(evidence['motor_source_end_tick'])
        tick_rows=evidence['motor_event_complete_ticks']
        id_rows=evidence['motor_event_local_indices']
        if (not np.array_equal(evidence['motor_sensory_index'],np.repeat(np.arange(110),5))
            or not np.array_equal(sources,np.r_[-1,np.arange(2,1100,2)])):
            raise AssertionError('motor source/sensory mapping mismatch')
    else:
        start=np.asarray(evidence['interval_start_s']); end=np.asarray(evidence['interval_end_s'])
        applied=controls; decoded=np.asarray(evidence['decoded_next_controls'])
        sample=start; current=np.asarray(evidence['sensory_current_nA'])
        activation=np.asarray(evidence['activation'])
        sources=np.asarray(evidence['applied_source_interval'])
        if not np.array_equal(sources,np.r_[-1,np.arange(109)]):
            raise AssertionError('5ms source interval mismatch')
        probe=np.asarray(result['probe_local_indices'],dtype=int)
        tick_rows=[];id_rows=[]
        for k,record in enumerate(result['records']):
            events=np.asarray(record['probe_emitted_spike_events_step_column'],dtype=int).reshape(-1,2)
            # Probe contains mapped MNs plus sensory cells: keep only mapped MNs.
            motor_ids=set(int(i) for row in result['motor_manifest_rows'] for i in row['local_indices'])
            selected=[(k*10+int(t)+1,int(probe[int(col)])) for t,col in events
                      if int(probe[int(col)]) in motor_ids]
            tick_rows.append(np.array([tick for tick,_ in selected],dtype=np.int64))
            id_rows.append(np.array([idx for _,idx in selected],dtype=np.int64))
    dt=period_ms/1000.
    np.testing.assert_allclose(sample,sensory_time,rtol=0,atol=1e-10)
    np.testing.assert_allclose(start,np.arange(intervals)*dt,rtol=0,atol=1e-10)
    np.testing.assert_allclose(end,np.arange(1,intervals+1)*dt,rtol=0,atol=1e-9)
    np.testing.assert_array_equal(applied,controls)
    np.testing.assert_array_equal(applied[1:],decoded[:-1])
    if current.shape!=(110,n_neurons) or activation.shape[0]!=intervals:
        raise AssertionError('sensory/activation evidence shape mismatch')
    if len(tick_rows)!=intervals or len(id_rows)!=intervals:
        raise AssertionError('incomplete MN event chronology')
    offsets=np.r_[0,np.cumsum([len(row) for row in tick_rows])]
    ticks=np.concatenate(tick_rows)
    indices=np.concatenate(id_rows)
    for k,(tick_row,id_row) in enumerate(zip(tick_rows,id_rows)):
        if (len(tick_row)!=len(id_row) or np.any(tick_row <= k*(period_ms*2)) or
            np.any(tick_row > (k+1)*(period_ms*2)) or
            np.any(id_row < 0) or np.any(id_row >= n_neurons)):
            raise AssertionError('future, stale, or invalid MN event')
    for arr in (current,decoded,activation):
        if not np.isfinite(arr).all():
            raise AssertionError('nonfinite sensory/command/activation evidence')
    return dict(body_time_height_act_ctrl=body, sensory_sample_start_s=sample,
                sensory_current_nA=current,motor_start_s=start,motor_end_s=end,
                motor_applied_controls=applied,motor_decoded_next_controls=decoded,
                motor_activation_end=activation,motor_source=np.asarray(sources),
                motor_event_offsets=offsets,
                motor_event_complete_ticks_flat=ticks,
                motor_event_local_indices_flat=indices)


def execute(*, out, period_ms, authorization, duration_s=DURATION_S, replay=None, push=None):
    if duration_s != DURATION_S or replay is not None or push is not None:
        raise ValueError('only normal 550ms without replay/push is registered')
    out,authorization,scope=require_scope(out,period_ms,authorization)
    identity=require_frozen_inputs(WEIGHTS,POSE,BATCH/'source_hashes_before.json',
                                    ROOT/'run_neural_muscle_closure.py')
    if not identity['fresh_5ms_baseline_required']:
        raise ValueError('historical source identity unexpectedly changed')
    with np.load(POSE,allow_pickle=False) as z: base=z['qpos'].copy()
    model,seated,gate,joints,geom_leg=closure.prepare_body(initial_qpos=base,body_dt_s=BODY_DT)
    require_legal_initial_pose(gate)
    if not np.array_equal(base,seated):
        raise ValueError('initial qpos was changed')
    np.random.seed(0)
    built=closure.build_neural_tier(closure.TierConfig(background_mean_nA=0,
                         background_sd_nA=0,background_seed=0))
    import mujoco
    names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(model.nu)]
    manifest=closure.build_motor_manifest(json.loads(closure.POOLS.read_text()),
                                            built['tier']['root_ids'],names)
    groups,_=closure.channel_index_map(int(built['n']),built['tier']['root_ids'])
    files=(Path(__file__), ROOT/'run_motor_cadence_single.py',
           ROOT/'run_neural_muscle_closure.py',ROOT/'run_motor_body_integration.py',
           ROOT/'engine/embodied/motor_translation.py',
           ROOT/'engine/embodied/motor_cadence_schedule.py',
           ROOT/'engine/embodied/motor_cadence_identity.py',
           closure.cp.CANONICAL_XML,closure.POOLS,WEIGHTS,POSE)
    source_hashes={str(p.resolve()):sha256(p) for p in files}
    model_identity=dict(nq=int(model.nq),nv=int(model.nv),nu=int(model.nu),
                        timestep_s=float(model.opt.timestep),
                        mass_sha256=hashlib.sha256(np.asarray(model.body_mass).tobytes()).hexdigest())
    config=dict(mode='single_normal_550ms',duration_s=DURATION_S,
                motor_dt_s=period_ms/1000.,sensory_dt_s=.005,neural_dt_s=.0005,
                body_dt_s=BODY_DT,seed=0,weight_path=str(WEIGHTS),pose_path=str(POSE),
                model_identity=model_identity,identity=identity,source_hashes=source_hashes,
                scope_sha256=sha256(authorization),
                environment={k:os.environ.get(k) for k in
                             ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MUJOCO_GL')})
    if period_ms == 1:
        baseline=out.parent/'arm5'
        if not (baseline/'SUMMARY.json').is_file() or not (baseline/'OBSERVATIONS.npz').is_file():
            raise ValueError('fresh completed 5ms baseline required before 1ms arm')
        baseline_summary=json.loads((baseline/'SUMMARY.json').read_text())
        baseline_config=json.loads((baseline/'CONFIG.json').read_text())
        if (baseline_summary.get('status') != 'PASS_ENGINEERING_RUN' or
            baseline_summary.get('body_steps') != BODY_STEPS or
            baseline_config.get('motor_dt_s') != .005 or
            {k:v for k,v in baseline_config.items() if k!='motor_dt_s'} !=
            {k:v for k,v in config.items() if k!='motor_dt_s'}):
            raise ValueError('completed 5ms baseline identity differs from 1ms arm')
        with np.load(baseline/'OBSERVATIONS.npz',allow_pickle=False) as z:
            if z['body_substep_observations'].shape != (BODY_STEPS,5):
                raise ValueError('5ms baseline support observations incomplete')
    json_save(out/'CONFIG.json',config)
    steps=[]
    contact_observer=BodyObserver(model,geom_leg)
    def observer(m,d):
        contact_observer(m,d)
        steps.append((float(d.time),float(d.xpos[m.body('Thorax').id,2]),
                      float(np.mean(d.act)),float(np.mean(d.ctrl))))
    evidence={}
    result,trajectory,commands,spikes=closure.episode(
        model,seated,joints,geom_leg,built,manifest,groups,'normal',DURATION_S,
        stride=10,full_detail_intervals=0,external_touch=False,push=None,
        evidence=evidence,motor_translator=WEIGHTS,body_observer=observer,
        body_dt_s=BODY_DT,motor_dt_s=period_ms/1000.)
    # Source metadata for the default 5ms path, without altering the episode.
    result['motor_manifest_rows']=manifest['rows']
    arrays=verify_observations(evidence,commands,steps,result,period_ms,int(built['n']))
    contact=contact_observer.array()
    if contact.shape!=(BODY_STEPS,5) or not np.isfinite(contact).all():
        raise AssertionError('incomplete foot contact or tilt observations')
    np.testing.assert_allclose(contact[:,0],arrays['body_time_height_act_ctrl'][:,0],rtol=0,atol=1e-9)
    np.testing.assert_allclose(contact[:,3],arrays['body_time_height_act_ctrl'][:,1],rtol=0,atol=1e-9)
    arrays.update(trajectory=trajectory,commands=commands,spike_counts=spikes,
                  body_substep_observations=contact,
                  model_weight=np.asarray(float(model.body_mass.sum()*np.linalg.norm(model.opt.gravity))),
                  initial_height_model=np.asarray(float(result['initial_height_model'])))
    np.savez_compressed(out/'OBSERVATIONS.npz',**arrays)
    json_save(out/'SUMMARY.json',dict(status='PASS_ENGINEERING_RUN',
              motor_period_ms=period_ms,body_steps=BODY_STEPS,neural_ticks=NEURAL_TICKS,
              sensory_samples=110,motor_windows=len(commands),initial_gate=gate,
              final_height_model=float(result['final_height_model']),
              behavior_success='NOT_EVALUATED',units='NOT_ESTABLISHED'))
    return config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--motor-period-ms',type=int,choices=[1,5],required=True)
    parser.add_argument('--human-scope-receipt',type=Path,required=True)
    args=parser.parse_args()
    execute(out=args.output,period_ms=args.motor_period_ms,
            authorization=args.human_scope_receipt)

if __name__=='__main__':
    main()
