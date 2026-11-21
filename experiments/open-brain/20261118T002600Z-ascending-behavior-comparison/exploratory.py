"""Supplementary candidate-vs-no-up ablation; primary criteria are unchanged."""
from neural_runtime import *


def main():
    result=read(HERE/'FINAL_BEHAVIOR.json');rows=[]
    for item in result['rows']:
        s=item['metrics']['translated'];z=item['metrics']['no_up'];seed=item['seed']
        folder=HERE/'FORMAL'/f'seed{seed}'
        with np.load(folder/'NEURAL.npz',allow_pickle=False) as a:
            differences=int(np.count_nonzero(a['dn_emitted'][:,2]!=a['dn_emitted'][:,3]))
        dy=s['yaw_degrees']-z['yaw_degrees'];dv=s['active_forward_mm_s']-z['active_forward_mm_s']
        rows.append(dict(seed=seed,dn_event_differences=differences,yaw_difference_degrees=dy,
              active_speed_difference_mm_s=dv,yaw_threshold=abs(dy)>=5.,
              speed_threshold=abs(dv)>=.1*max(abs(z['active_forward_mm_s']),1e-12)))
    same_yaw=all(r['yaw_threshold'] and r['dn_event_differences']>0 for r in rows) and rows[0]['yaw_difference_degrees']*rows[1]['yaw_difference_degrees']>0
    same_speed=all(r['speed_threshold'] and r['dn_event_differences']>0 for r in rows) and rows[0]['active_speed_difference_mm_s']*rows[1]['active_speed_difference_mm_s']>0
    write(HERE/'EXPLORATORY_SENSORY_EFFECT.json',dict(status='EXPLORATORY_ABLATION_NOT_PRIMARY_ACCEPTANCE',rows=rows,
       repeated_candidate_yaw_effect=bool(same_yaw),repeated_candidate_speed_effect=bool(same_speed),
       primary_behavior_expectation=result['broad_behavior_expectation_met'],
       interpretation='Candidate and no-up share boundary ownership and upstream stimuli. This establishes '
       'observed influence of this input pipeline in these trials, not correct sensory semantics, obstacle avoidance '
       'or equivalence to the original reference. Added after inspecting the formal metrics; primary gates unchanged.'))
    print('exploratory candidate yaw effect',bool(same_yaw),[(r['seed'],r['yaw_difference_degrees']) for r in rows],flush=True)


if __name__=='__main__':main()
