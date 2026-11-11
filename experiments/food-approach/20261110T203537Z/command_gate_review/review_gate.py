"""READONLY algebraic action envelopes; no optimization or physical backend."""
from pathlib import Path
import sys,json
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from fast_motor import FastMotor,load_base,sha256,verify_hashes
from fast_data import prepare


def floor_stats(mean,y,radius):
    low=np.tanh(mean-radius[:,None]);high=np.tanh(mean+radius[:,None])
    error=np.maximum(low-y,0)+np.maximum(y-high,0)
    return dict(MSE_lower_bound=float(np.mean(error**2)),outside_fraction=float(np.mean(error>0)),
                max_unavoidable_action_error=float(error.max()),radius_min=float(radius.min()),radius_max=float(radius.max()))


def main():
    torch.set_num_threads(1)
    model=FastMotor(load_base())
    episodes,data=prepare(model)
    sources={str(ROOT/n):sha256(ROOT/n) for n in ('fast_motor.py','fast_data.py')}
    sources[str(HERE/'review_gate.py')]=sha256(HERE/'review_gate.py')
    results={}
    windows=[('onset0_to_0p2',0,.2),('onset0p2_to_0p5',.2,.5),('onset0p5_to_1',.5,1.),('post1_to_4p8',1.,4.8),('post4p8_to_9p6',4.8,9.6)]
    windows += [(f'provenance_bin_{k:02d}',k*.8,(k+1)*.8) for k in range(12)]
    for e in episodes:
        m=e['mean'][960:].numpy();y=e['y'][960:].numpy();u=e['u'][960:].numpy()
        norm=u/np.array([.15,1.]);old=np.tanh((norm**2).sum(1))
        # Two independently bounded heads add their possible latent intervals.
        # On the registered separate axes this is |turn| or |-forward/.15|.
        linear=np.abs(norm).sum(1)
        if np.any((np.abs(norm)>1e-12).sum(1)>1):raise ValueError('expected separate command axes')
        base_action=np.tanh(m);delta=np.abs(y-base_action)
        time=np.arange(len(m))*.005
        case=dict(whole_post_window=dict(old_norm=floor_stats(m,y,old),candidate_linear_heads=floor_stats(m,y,linear)),
                  windows={},onset_samples=[],small_command_groups={},exact_teacher_saturation_count=int(np.sum(np.abs(y)==1)),
                  all_post_target_delta=dict(mean_abs=float(delta.mean()),max_abs=float(delta.max()),rms=float(np.sqrt(np.mean(delta**2)))))
        for label,a,b in windows:
            mask=(time>=a-1e-12)&(time<b-1e-12)
            case['windows'][label]=dict(start_s=a,end_s=b,decisions=int(mask.sum()),old_norm=floor_stats(m[mask],y[mask],old[mask]),candidate_linear_heads=floor_stats(m[mask],y[mask],linear[mask]))
        for t in (0,.005,.025,.05,.1,.2,.5,.75,.995):
            i=round(t/.005)
            case['onset_samples'].append(dict(elapsed_s=float(time[i]),raw_command=u[i].tolist(),normalized_abs=float(linear[i]),old_gate=float(old[i]),linear_radius=float(linear[i]),
                bounded_target_delta_max=float(delta[i].max()),bounded_target_delta_rms=float(np.sqrt(np.mean(delta[i]**2))),
                old_floor=floor_stats(m[i:i+1],y[i:i+1],old[i:i+1]),linear_floor=floor_stats(m[i:i+1],y[i:i+1],linear[i:i+1])))
        for label,a,b in [('zero',0,0),('positive_to_0p1',0,.1),('0p1_to_0p25',.1,.25),('0p25_to_0p5',.25,.5)]:
            mask=(linear==0) if label=='zero' else (linear>a)&(linear<=b)&(time<1)
            if mask.any():case['small_command_groups'][label]=dict(decisions=int(mask.sum()),mean_abs_command=float(linear[mask].mean()),old_radius_mean=float(old[mask].mean()),linear_radius_mean=float(linear[mask].mean()),target_delta_rms=float(np.sqrt(np.mean(delta[mask]**2))),target_delta_max=float(delta[mask].max()),old_floor=floor_stats(m[mask],y[mask],old[mask]),linear_floor=floor_stats(m[mask],y[mask],linear[mask]))
        case['first_post_decision']=dict(command=u[0].tolist(),base_to_teacher_max_abs=float(delta[0].max()),base_to_teacher_action_MSE=float(np.mean(delta[0]**2)),unavoidable_both_gates=float(np.mean(delta[0]**2)))
        # No use of atanh at saturated labels: bounded action interval is exact.
        results[e['name']]=case
    verify_hashes(sources);verify_hashes(data)
    normalized_demo=np.array([0.,.01,.1,.5,2/3,1.])
    report=dict(scope='read-only algebraic candidate comparison on four authenticated teacher streams; candidate heads are hypothetical',
                cases=results,aggregate_equal_episode_floor={key:float(np.mean([r['whole_post_window'][key]['MSE_lower_bound'] for r in results.values()])) for key in ('old_norm','candidate_linear_heads')},
                scaling_examples=[dict(normalized_command=float(v),old_radius=float(np.tanh(v*v)),candidate_linear_radius=float(v)) for v in normalized_demo],
                derivatives=dict(old_norm_gate_at_neutral=0.,old_bounded_output_command_derivative_at_neutral=0.,candidate_turn_derivative='tanh(Fturn(features at u0))',candidate_forward_derivative='-tanh(Fspeed(features at u0))/.15',candidate_head_parameter_derivative_at_neutral=0.,notes='Candidate derivative requires nonzero head outputs. If both heads are identically zero, command derivative is also zero. Policy gradient with stochastic commands differs from deterministic chain gradients.'),
                caveats=['Linear envelope is larger on registered normalized axes0<|u|<=1; this is a range change as well as conditioning change.',
                         'A command-independent signed turn head forces an odd correction; symmetric or even corrections require head command dependence or another expressly neural representation.',
                         'Per-input interval floor allows independently optimal outputs at every observation and ignores finite shared-network capacity/learning/generalization.',
                         'Source clock and teacher event windows are diagnostic provenance only, never neural inputs.',
                         'No atanh inference is used for exact teacher±1 targets; their finite latent target is not asserted.',
                         'No physical or learned improvement follows from a lower algebraic floor.',
                         'A future range-matched signed gate coefficient sign(v)*tanh(v²) retains old axis envelope but has zero derivative at0; exact oldquadratic radius and nonzero derivative atneutral cannot both hold for bounded heads.'],
                physical_steps=0,optimizer_steps=0,model_artifacts_created=0,source_hashes=sources,data_hashes=data)
    out=HERE/'GATE_REVIEW.json'
    with out.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(equal_episode_floor=report['aggregate_equal_episode_floor'],cases={n:{key:r['whole_post_window'][key] for key in ('old_norm','candidate_linear_heads')} for n,r in results.items()},neutral_boundary={n:r['first_post_decision'] for n,r in results.items()}),indent=2))

if __name__=='__main__':main()
