"""Parent post-audit saved-stream diagnostics; no new NN/Core/physics."""
import runtime as r

def summarize(folder,result):
    import numpy as np
    paths=[folder/'trace_1ms.npz',folder/'neural_decisions.npz']
    hashes={str(p):r.sha(p) for p in paths}
    with np.load(paths[0],allow_pickle=False) as t, np.load(paths[1],allow_pickle=False) as d:
        if t['time'].shape!=(r.TOTAL_MS,) or d['time'].shape!=(r.DECISIONS,):
            raise ValueError('Full saved streams required')
        windows=[]
        events=result['autonomous_cycles']['events']
        legs=list(result['autonomous_cycles']['complete_per_leg'])
        for k in range(4):
            lo=5800+k*9600;hi=lo+9600
            td=slice(lo,hi);dd=slice(lo//5,hi//5)
            counts={leg:sum(e['type']=='complete' and e['leg']==leg and lo*.001-1e-8<=float(e['time'])<hi*.001-1e-8 for e in events) for leg in legs}
            target=t['target'][td]
            windows.append(dict(autonomous_interval_s=[k*9.6,(k+1)*9.6],absolute_interval_ms=[lo,hi],sample_count=hi-lo,root_x_displacement=float(t['qpos'][hi-1,0]-t['qpos'][lo,0]),COM_x_displacement=float(t['com'][hi-1,0]-t['com'][lo,0]),action_RMS=float(np.sqrt(np.mean(d['actually_applied_action'][dd]**2))),target_step_RMS_rad=float(np.sqrt(np.mean(np.diff(target,axis=0)**2))),target_net_RMS_rad=float(np.sqrt(np.mean((target[-1]-target[0])**2))),motor_hidden_step_RMS=float(np.sqrt(np.mean((d['hidden_after'][dd]-d['hidden_before'][dd])**2))),strict_cycles_completed_by_end_time=counts,all_six_completed_in_window=all(v>0 for v in counts.values())))
        diagnostic=dict(windows=windows,last9p6=windows[-1],last_brain_command=d['command'][-1].tolist() if 'command' in d.files else None,missing_command_reason=None if 'command' in d.files else 'NOT_AVAILABLE_IN_SAVED_SCHEMA',classification='DESCRIPTIVE_ONLY_NO_NEW_ACCEPTANCE_THRESHOLD',hypotheses=dict(flat_also_decays='supports a general long-term old336 closed-loop issue; cannot assign obstacle-only cause',flat_keeps_progress='supports scene association; cannot distinguish visual versus mechanical effects'),causal_result='PENDING_PARENT_COMPARISON_WITH_ACTUAL_MATCHED_ROUGH',strict_cycle_window_definition='Original full-run sensed complete event time falls in half-open 9.6s absolute window; event may have begun in earlier window',source_hashes=hashes)
    for p,digest in hashes.items():
        if r.sha(p)!=digest: raise ValueError('Diagnostic source drift '+p)
    return diagnostic
