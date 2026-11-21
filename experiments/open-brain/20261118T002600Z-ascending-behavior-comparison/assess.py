"""Behavior agreement and actual sensory influence are separate endpoints."""
from neural_runtime import *

MODES=('native_full','reference_boundary','translated','no_up')


def arrays(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}


def relative(reference,student):
    xy=np.linalg.norm(np.array(reference['final_xy'])-np.array(student['final_xy']))
    gates=dict(final_xy=float(xy)<=.5,
      yaw=abs(student['yaw_degrees']-reference['yaw_degrees'])<=max(10.,.25*abs(reference['yaw_degrees'])),
      active_speed=abs(student['active_forward_mm_s']-reference['active_forward_mm_s'])<=max(1.,.25*abs(reference['active_forward_mm_s'])),
      height=student['min_height_mm']>=reference['min_height_mm']-.1,
      student_safety=student['safety_pass'])
    return dict(pass_=all(gates.values()),gates=gates,final_xy_difference_mm=float(xy),
       yaw_difference_degrees=student['yaw_degrees']-reference['yaw_degrees'],
       active_speed_difference_mm_s=student['active_forward_mm_s']-reference['active_forward_mm_s'])


def main():
    rows=[];direction=[]
    for seed in (10901,11311):
        folder=HERE/'FORMAL'/f'seed{seed}';meta=read(folder/'RESULT.json');verify(meta['source_sha256'])
        assert sha(folder/'NEURAL.npz')==meta['neural_sha256']
        neural=arrays(folder/'NEURAL.npz')
        modes={m:read(folder/m/'RESULT.json') for m in MODES}
        for m,r in modes.items():
            assert sha(folder/m/'TRACE.npz')==r['trace_sha256']
        comparisons={m:relative(modes[m],modes['translated']) for m in ('native_full','reference_boundary')}
        speed=modes['reference_boundary']['active_forward_mm_s']-modes['no_up']['active_forward_mm_s']
        yaw=modes['reference_boundary']['yaw_degrees']-modes['no_up']['yaw_degrees']
        dn_difference=int(np.count_nonzero(neural['dn_emitted'][:,1]!=neural['dn_emitted'][:,3]))
        speed_effect=abs(speed)>=.10*max(abs(modes['no_up']['active_forward_mm_s']),1e-12)
        yaw_effect=abs(yaw)>=5.
        influence=dict(dn_event_differences=dn_difference,speed_difference_mm_s=speed,yaw_difference_degrees=yaw,
             speed_threshold_pass=bool(speed_effect),yaw_threshold_pass=bool(yaw_effect),
             pass_=bool(dn_difference>0 and (speed_effect or yaw_effect)))
        direction.append((speed,yaw,speed_effect,yaw_effect,dn_difference))
        rows.append(dict(seed=seed,metrics={m:{k:r[k] for k in ('safety_pass','forward_dx_mm','yaw_degrees','active_forward_mm_s',
                'stand_rms_mm_s','advance_pass','final_stand_pass','min_height_mm')} for m,r in modes.items()},
                comparisons=comparisons,reference_sensory_influence=influence,
                dn_totals=neural['dn_emitted'].sum((0,2)).tolist(),
                actual_candidate_an_arrivals=neural['an_arrived'][:,2].sum(0).tolist(),
                native_full_an_spikes=neural['full_an_spikes'].sum(0).tolist()))
    repeatable_speed=all(d[2] and d[4]>0 for d in direction) and direction[0][0]*direction[1][0]>0
    repeatable_yaw=all(d[3] and d[4]>0 for d in direction) and direction[0][1]*direction[1][1]>0
    original=HERE/'FORMAL/seed10901';replay=HERE/'REPLAY/seed10901'
    replay_result=read(replay/'RESULT.json');verify(replay_result['source_sha256'])
    pairs={}
    for mode in MODES:
        x=arrays(original/mode/'TRACE.npz');y=arrays(replay/mode/'TRACE.npz')
        pairs[mode]={k:bool(np.array_equal(x[k],y[k])) for k in x}
    x=arrays(original/'NEURAL.npz');y=arrays(replay/'NEURAL.npz')
    pairs['neural']={k:bool(np.array_equal(x[k],y[k])) for k in x}
    agreement=all(r['comparisons']['reference_boundary']['pass_'] for r in rows)
    native_agreement=all(r['comparisons']['native_full']['pass_'] for r in rows)
    behavior_influence=bool(repeatable_speed or repeatable_yaw)
    result=dict(project='P1_open_source_brain',status='BEHAVIOR_TEST_COMPLETED',rows=rows,
       controlled_reference_agreement=agreement,native_full_reference_agreement=native_agreement,
       repeated_reference_sensory_behavior_influence=behavior_influence,
       repeated_speed_influence=bool(repeatable_speed),repeated_yaw_influence=bool(repeatable_yaw),
       independent_replay=pairs,replay_exact=all(all(v.values()) for v in pairs.values()),
       broad_behavior_expectation_met=bool(agreement and behavior_influence),
       biological_behavior_validation='NOT_CLAIMED',optimizer_updates=0,
       native_body_steps=240000,body_episodes=12,brain_batch_ticks=63200,brain_lane_ticks=252800,
       neck_batch_ticks=60000,neck_lane_ticks=240000,local_reference_lif_ticks=60000,native_receptor_lif_ticks=60000,
       interpretation='Agreement with a reference and actual sensory-driven behavioral influence are independent endpoints. '
          'Original full-brain control uses a newly declared engineering sensory interface, not the unpublished Eon body interface.')
    write(HERE/'FINAL_BEHAVIOR.json',result)
    print('controlled agreement',agreement,'native-full agreement',native_agreement,'repeated sensory influence',behavior_influence,
          'exact replay',result['replay_exact'],flush=True)


if __name__=='__main__':main()
