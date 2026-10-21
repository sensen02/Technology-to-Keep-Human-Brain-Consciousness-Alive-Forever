"""One fresh, bounded normal MN->muscle episode, no fitting or extra drive.

CLI exposes motor 5/1ms with fixed 5ms sensory and 0.5ms neural clocks.
A separate preregistered output is required for every arm. Manual research.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

import run_neural_muscle_closure as closure
from engine.embodied.motor_cadence_identity import require_frozen_inputs, sha256
from engine.embodied.closure_clock import grid_steps
from run_motor_body_integration import require_legal_initial_pose

ROOT = Path(__file__).resolve().parent
BATCH = Path('/run/media/sensen/Data2/engine修复验收/pool-gain-quarter-resume2-20261018T164139Z')
WEIGHTS = BATCH/'baseline_weights.json'
POSE = Path('/home/sensen/Desktop/engine修复验收/六足内带显式约束-20261018T031500Z/candidate_qpos.npz')
BODY_DT = 6.25e-6


def json_save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False)+'\n')


def execute(*, out, motor_period_ms, duration_s):
    out = Path(out).resolve()
    if motor_period_ms not in (1,5):
        raise ValueError('motor period must be exactly 1 or 5ms')
    if duration_s not in (.005, .01):
        raise ValueError('single-arm smoke entry permits exactly 5 or 10ms; longer runs require another registered entry')
    sensory_intervals = grid_steps(duration_s, .005)
    body_count = grid_steps(duration_s, BODY_DT)
    neural_count = grid_steps(duration_s, .0005)
    if not out.is_dir() or not (out/'PREREGISTRATION.md').is_file():
        raise ValueError('new existing output directory with PREREGISTRATION.md required')
    if any(out.iterdir()) and set(p.name for p in out.iterdir()) != {'PREREGISTRATION.md'}:
        raise FileExistsError('output must be fresh and contain only PREREGISTRATION.md')
    identity = require_frozen_inputs(WEIGHTS, POSE, BATCH/'source_hashes_before.json',
                                      ROOT/'run_neural_muscle_closure.py')
    # These are new-source runs; never claim equivalence to an old-source body trajectory.
    assert identity['fresh_5ms_baseline_required']
    with np.load(POSE, allow_pickle=False) as z:
        base=z['qpos'].copy()
    model, seated, gate, joints, geom_leg = closure.prepare_body(initial_qpos=base,
                                                                  body_dt_s=BODY_DT)
    require_legal_initial_pose(gate)
    if not np.array_equal(base,seated):
        raise ValueError('seated qpos differs from frozen explicit initial pose')
    np.random.seed(0)
    built=closure.build_neural_tier(closure.TierConfig(background_mean_nA=0,
                         background_sd_nA=0,background_seed=0))
    import mujoco
    names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,i)
           for i in range(model.nu)]
    manifest=closure.build_motor_manifest(json.loads(closure.POOLS.read_text()),
                                            built['tier']['root_ids'],names)
    groups,_=closure.channel_index_map(int(built['n']),built['tier']['root_ids'])
    model_identity=dict(nq=int(model.nq),nv=int(model.nv),nu=int(model.nu),
                        timestep_s=float(model.opt.timestep),
                        mass_sha256=hashlib.sha256(np.asarray(model.body_mass).tobytes()).hexdigest())
    source_files=(Path(__file__), ROOT/'run_neural_muscle_closure.py',
                  ROOT/'engine/embodied/motor_translation.py',
                  ROOT/'engine/embodied/motor_cadence_schedule.py',
                  ROOT/'engine/embodied/motor_cadence_identity.py',
                  ROOT/'run_motor_body_integration.py', closure.cp.CANONICAL_XML,
                  closure.POOLS, WEIGHTS, POSE)
    source_hashes={str(p.resolve()):sha256(p) for p in source_files}
    config=dict(mode='single_normal_smoke',duration_s=duration_s,
                motor_dt_s=motor_period_ms/1000.,sensory_dt_s=.005,neural_dt_s=.0005,
                body_dt_s=BODY_DT,seed=0,weight_path=str(WEIGHTS),pose_path=str(POSE),
                identity=identity,model_identity=model_identity,source_hashes=source_hashes,
                environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MUJOCO_GL')})
    json_save(out/'CONFIG.json',config)
    # Observations are captured after each native body step, without modifying data.
    steps=[]
    def observer(m,d):
        steps.append((float(d.time),float(d.xpos[m.body('Thorax').id,2]),
                      float(np.mean(d.act)),float(np.mean(d.ctrl))))
    evidence={}
    result,trajectory,commands,spikes=closure.episode(
        model,seated,joints,geom_leg,built,manifest,groups,'normal',duration_s,
        external_touch=False, push=None, evidence=evidence, motor_translator=WEIGHTS,
        body_observer=observer,body_dt_s=BODY_DT,motor_dt_s=motor_period_ms/1000.)
    if len(steps)!=body_count or len(commands)!=sensory_intervals*(5 if motor_period_ms==1 else 1):
        raise AssertionError('unexpected body or motor window count')
    if commands[0].any():
        raise AssertionError('initial motor command not zero')
    expected_sensory = np.arange(sensory_intervals)*.005
    if motor_period_ms==1:
        if (not np.allclose(evidence['sensory_sample_start_s'],expected_sensory,rtol=0,atol=1e-12) or
            np.any(evidence['motor_source_end_tick'][1:] >
                   np.rint(evidence['motor_start_s'][1:]/.0005).astype(int))):
            raise AssertionError('sensory hold or completed MN causality broken')
        if not np.array_equal(evidence['motor_sensory_index'],
                              np.repeat(np.arange(sensory_intervals),5)):
            raise AssertionError('sensory hold index changed within a 5ms block')
        event_ticks = evidence['motor_event_complete_ticks']
        event_indices = evidence['motor_event_local_indices']
        if any(np.any(ticks > int(round(end/.0005))) or len(ticks)!=len(indices)
               for ticks,indices,end in zip(event_ticks,event_indices,evidence['motor_end_s'])):
            raise AssertionError('motor events not completed before decoder endpoint')
        # Archive ragged event sources without object dtype or pickle.
        offsets=np.r_[0,np.cumsum([len(ticks) for ticks in event_ticks])]
        event_arrays=dict(motor_event_offsets=offsets,
                          motor_event_complete_ticks_flat=np.concatenate(event_ticks),
                          motor_event_local_indices_flat=np.concatenate(event_indices))
    else:
        if not np.allclose(evidence['interval_start_s'],expected_sensory,rtol=0,atol=1e-12):
            raise AssertionError('legacy 5ms sensory update schedule changed')
        probe=np.asarray(result['probe_local_indices'],dtype=np.int64)
        motor_ids={int(i) for row in manifest['rows'] for i in row['local_indices']}
        events=[]
        for k,record in enumerate(result['records']):
            events.append([(k*10+int(tick)+1,int(probe[int(column)]))
                           for tick,column in record['probe_emitted_spike_events_step_column']
                           if int(probe[int(column)]) in motor_ids])
        event_arrays=dict(motor_event_offsets=np.r_[0,np.cumsum([len(row) for row in events])],
                          motor_event_complete_ticks_flat=np.asarray(
                              [tick for row in events for tick,_ in row],dtype=np.int64),
                          motor_event_local_indices_flat=np.asarray(
                              [index for row in events for _,index in row],dtype=np.int64))
    np.savez_compressed(out/'OBSERVATIONS.npz',body_time_height_act_ctrl=np.asarray(steps),
                        trajectory=trajectory,commands=commands,spike_counts=spikes,
                        **event_arrays,
                        **{key:value for key,value in evidence.items() if isinstance(value,np.ndarray)})
    json_save(out/'SUMMARY.json',dict(status='PASS_ENGINEERING_SMOKE',
              motor_period_ms=motor_period_ms,body_steps=len(steps),neural_ticks=neural_count,
              sensory_samples=sensory_intervals,motor_windows=len(commands),
              command_first_zero=True, initial_gate=gate,
              final_height_model=float(result['final_height_model']),
              behavior_success=False,units='NOT_ESTABLISHED'))
    return config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--motor-period-ms',type=int,choices=[1,5],required=True)
    parser.add_argument('--duration',type=float,default=.005)
    args=parser.parse_args()
    execute(out=args.output,motor_period_ms=args.motor_period_ms,duration_s=args.duration)

if __name__=='__main__':
    main()
