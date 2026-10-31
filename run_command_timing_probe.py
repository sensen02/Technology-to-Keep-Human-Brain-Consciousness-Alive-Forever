"""S159/P1: one S154 contrast, application delayed exactly one 0.5ms tick.
Preparation/import/--stage-only are inert; no historical source is modified.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, ast, hashlib, json, sys
import run_activation_rate_probe as base
ROOT=Path(__file__).resolve().parent
REFERENCE=Path('/home/sensen/Desktop/engine修复验收/s154-activation-rate/20261024T215751775489Z')
OUTPUT_PARENT=Path('/home/sensen/Desktop/engine修复验收/s159-command-timing')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

LOOP='''            # s159: fresh generation is queued; pending command is not applied in this interval.
            rows = [sample()]; currents,spikes,features,commands,schedule = [],[],[],[],[]
            applied_commands, applied_decoded, applied_source_ticks = [],[],[]
            stop_reasons = s142_reasons(rows[-1], q, m)
            refresh_times = []
            for tick in range(TICKS):
                if stop_reasons: break
                if tick % 10 == 0:
                    observed = observe_quantities(m,d,sensory_ref,joints,geom_leg)
                    current, _ = enc.encode(observed)
                    sensory_sample_actual_s = float(d.time)
                    refresh_times.append(sensory_sample_actual_s)
                    atomic_json(dest/f'OBSERVATION_{tick:02d}.json',raw_observation_record(observed)[0])
                currents.append(current.copy()); reserve(neural_ticks=1)
                sp = net.step(current_nA=current).copy()
                start,end = (PRE+tick)*.0005,(PRE+tick+1)*.0005
                u,info = dec.consume(sp[None,:],spike_dt_s=.0005,start_s=start,end_s=end)
                assert abs(net.t_ms/1000-end) < 1e-12 and dec.interval == PRE+tick
                assert abs(d.time-tick*.0005) < 1e-12
                spikes.append(sp); features.append(info['pool_features']); commands.append(u.copy())
                applied_source_tick = tick-1
                applied_raw = np.zeros_like(u) if tick == 0 else commands[tick-1].copy()
                applied_actual = .5 * applied_raw
                if tick:
                    assert schedule[tick-1]['command_available_physical_s'] <= float(d.time)+1e-12
                    schedule[tick-1]['command_actual_application_start_s'] = float(d.time)
                schedule.append(dict(tick=tick,source_start_s=start,source_end_s=end,
                    decoder_validfrom_s=end,command_available_physical_s=end-.08,
                    sensory_sample_actual_s=sensory_sample_actual_s,
                    sensory_held_from_s=sensory_sample_actual_s,last_observation_actual_s=sensory_sample_actual_s,
                    neural_tick_physical_start_s=tick*.0005,
                    physical_start_s=float(d.time),planned_physical_end_s=(tick+1)*.0005,
                    command_planned_application_start_s=(tick+1)*.0005,
                    command_planned_application_end_s=(tick+2)*.0005,
                    command_actual_application_start_s=None,command_actual_application_end_s=None,
                    applied_source_tick=applied_source_tick,
                    applied_source_kind='INITIAL_ZERO_NO_PRE_COMMAND' if tick == 0 else 'RELEASE_NEURAL_TICK',
                    applied_decoded=applied_raw.tolist(),applied_gain=.5,applied_actual_ctrl=applied_actual.tolist(),
                    source_offset_s=.08,consume_id=dec.interval))
                applied_commands.append(applied_actual.copy()); applied_decoded.append(applied_raw.copy())
                applied_source_ticks.append(applied_source_tick)
                d.ctrl[:] = applied_actual
                for sub in range(substeps):
                    require_no_callbacks(callback_snapshot(mj))
                    assert not d.qfrc_applied.any() and not d.xfrc_applied.any()
                    reserve(step=1,forward=1); mj.mj_step(m,d); mj.mj_forward(m,d)
                    row = sample(); rows.append(row)
                    stop_reasons = s142_reasons(row, q, m)
                    if stop_reasons: break
                schedule[-1]['actual_physical_end_s'] = float(d.time)
                if tick: schedule[tick-1]['command_actual_application_end_s'] = float(d.time)
            pending = [r['tick'] for r in schedule if r['command_actual_application_start_s'] is None]
            atomic_json(dest/'PENDING_COMMAND_EVENT.json',dict(event='UNCONSUMED_GENERATED_COMMAND_AT_STOP',
                stop_reasons=stop_reasons,physical_stop_s=float(d.time),pending_source_ticks=pending,
                postcommand_alignment='AVAILABLE' if any(k >= 0 for k in applied_source_ticks) else 'NOT_AVAILABLE',
                nonzero_ctrl_ever_applied=any(np.any(v) for v in applied_commands)))
            atomic_npz(dest/'PENDING_COMMANDS.npz',source_ticks=np.asarray(pending,dtype=np.int64),
                decoded=np.asarray([commands[k] for k in pending]),gain_applied_when_released=np.asarray(.5))
'''

def prepare():
    old,binding,baseline,changes=base.prepare(base.SELECTION)
    assert sha(base.ROOT/'run_activation_rate_probe.py') == sha(REFERENCE/'SOURCE_FREEZE/S146_run_activation_rate_probe.py')
    refs=[REFERENCE/n for n in ('PREREG.json','COMPLETE.json','init80ms/NATIVE_MODEL_ARRAYS.npz','init80ms/INITIALIZATION.npz','init80ms/TRAJECTORY.npz','SOURCE_FREEZE/PRIVATE_RUNNER_SOURCE.py')]
    binding['hashes'].update({str(p):sha(p) for p in [*refs,Path(__file__).resolve(),ROOT/'verify_command_timing_probe.py']})
    prior=old.private_source
    def transform(text):
        source,changes=prior(text)
        start=source.index('            rows = [sample()]')
        end=source.index('            arrays = stack_rows(rows)',start)
        replacement=(source[start:end],LOOP)
        source,extra=base.inert(base.S149,'_s159_replace').replace_exact(source,[replacement,
            ('applied=0.5*np.asarray(commands),decoded=np.asarray(commands),current=np.asarray(currents),',
             'applied=np.asarray(applied_commands),applied_decoded=np.asarray(applied_decoded),applied_source_tick=np.asarray(applied_source_ticks,dtype=np.int64),decoded=np.asarray(commands),current=np.asarray(currents),'),
            ("str(ROOT/'run_activation_rate_probe.py')]:", "str(ROOT/'run_activation_rate_probe.py'),str(ROOT/'run_command_timing_probe.py'),str(ROOT/'verify_command_timing_probe.py')]:"),
            ("applied_interface='all90 ctrl=0.5*decoded raw u after decode; no act overwrite; activation tau .001 only'", "applied_interface='S159 all90 ctrl[k]=.5*decoded[k-1]; ctrl[0]=ZERO no PRE command; tau .001/.0004 identical S154'"),
            ("sole_model_change='activation tau .0005 -> .001 all90'", "sole_model_change='NONE versus S154; native XML tau patch .0005 -> .001 exactly as S154', sole_contrast='application delay one .5ms tick; zero initial interval, no PRE command'"),
            ("interface='unchanged ctrl=.5 raw decoded all90; no act overwrite/filter/replay'", "interface='ctrl[k]=.5 raw decoded[k-1], k0 ZERO; no act overwrite/filter/replay'"),
        ])
        compile(source,str(old.ORIGINAL),'exec'); return source,changes+extra
    module_text=base.S142.read_text().replace("protocol='S142 ONE bounded free physical diagnostic; no assistance or activation assignment'", "protocol='S159 ONE bounded one-tick application-delay contrast versus S154', EXPERIMENT='COMMAND_TIMING', application_delay_ticks=1, application_delay_s=.0005, initial_interval_ctrl='ZERO no PRE command', candidate=False")
    module=base.inert(base.S142,'_s159_private',module_text)
    module.private_source=transform; module.stop_reasons=old.stop_reasons
    source,changes=transform(module.ORIGINAL.read_text())
    assert 'd.ctrl[:] = 0.5 * u' not in source
    return module,binding,source,changes

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',default='auto'); p.add_argument('--stage-only',action='store_true'); a=p.parse_args()
    module,binding,source,changes=prepare()
    assert binding['hashes']=={n:sha(n) for n in binding['hashes']}
    if a.stage_only:
        print(json.dumps(dict(status='INERT_COMPILE_OK',private_source_sha256=hashlib.sha256(source.encode()).hexdigest(),model_load=0,forward=0,step=0,neural_ticks=0))); return
    out=OUTPUT_PARENT/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') if a.output=='auto' else Path(a.output)
    if not out.is_absolute() or out.exists(): raise ValueError('fresh absolute output only; no resume/retry')
    previous=sys.argv
    try: sys.argv=[str(Path(__file__).resolve()),'--output',str(out)]; module.main()
    finally: sys.argv=previous
if __name__=='__main__': main()
