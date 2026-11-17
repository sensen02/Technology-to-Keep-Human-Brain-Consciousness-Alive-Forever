"""Recheck paired reflex ablations in the same common policy entry."""
import argparse
from pathlib import Path

import numpy as np

from compose import checksum, read, sources, write


def same_prefix(a,b,fields,prefix):
    return {k:bool(np.array_equal(a[k][:prefix],b[k][:prefix])) for k in fields}


def audit(live,ablation,output):
    a0,b0=(read(folder/'ASSEMBLY_AUDIT.json') for folder in (live,ablation))
    a1,b1=(read(folder/'RESULT.json') for folder in (live,ablation))
    a2,b2=(read(folder/'NATIVE_AUDIT.json') for folder in (live,ablation))
    assert a0['passed'] and a0['task']==b0['ablation'] and a0['task']==b0['task']
    first=a0['first_modified_action_ms']
    for root,figures,physical in ((live,a1,a2),(ablation,b1,b2)):
        assert checksum(root/'TRACE.npz')==figures['trace_sha256']
        assert checksum(root/'DECISIONS.npz')==figures['decision_sha256']
        assert checksum(root/'NATIVE.npz')==physical['native_sha256']
    with np.load(live/'DECISIONS.npz',allow_pickle=False) as a, np.load(ablation/'DECISIONS.npz',allow_pickle=False) as b:
        np.testing.assert_array_equal(a['ms'],b['ms'])
        decision_prefix=same_prefix(a,b,('action',),first//5)
        through_event=same_prefix(a,b,('x','intent','local','base'),first//5+1)
        assert not np.array_equal(a['action'][first//5],b['action'][first//5])
    with np.load(live/'TRACE.npz',allow_pickle=False) as a, np.load(ablation/'TRACE.npz',allow_pickle=False) as b:
        state_prefix=same_prefix(a,b,('time','qpos','qvel','loads','com'),first+1)
        if a0['task']=='gap':
            _,gap,_=sources()
            sensors=gap.bench
            saved_stand=[]
            for record in (a,b):
                end=len(record['qpos'])-1
                saved_stand.append(sensors.standing(record,end-200,end))
        else:
            saved_stand=[a1['terminal_stand'],b1['terminal_stand']]
    with np.load(live/'NATIVE.npz',allow_pickle=False) as a, np.load(ablation/'NATIVE.npz',allow_pickle=False) as b:
        n=first*80
        native_prefix=same_prefix(a,b,a.files,n)
    actors = dict(on=a0['selected_decisions'],off=b0['selected_decisions'])
    other='gap' if a0['task']=='stumble' else 'stumble'
    no_false_branch=not actors['on'][other] and not actors['off'][other] and not actors['off'][a0['task']]
    proof=bool(all((*decision_prefix.values(),*through_event.values(),*state_prefix.values(),*native_prefix.values()))
               and no_false_branch)
    key='first_target_floor_reload_ms' if a0['task']=='stumble' else 'first_upper_floor_reload_ms'
    events={state:dict(first_reload_ms=run['event'].get(key),standing_at_endpoint_200ms=stand['passed'],
                       first_resisting_touch_ms=run['event'].get('first_resisting_touch_ms'),
                       actual_hole_loss_ms=run['event'].get('first_pad_drop_ms'))
            for state,run,stand in zip(('on','off'),(a1,b1),saved_stand,strict=True)}
    if a0['task']=='stumble':
        for state,run in zip(('on','off'),(a2,b2),strict=True):
            events[state]['resisting_impulse_W_s']=run['obstacle_resisting_impulse_Ws']
    write(output,dict(matched_scene={role:str(folder) for role,folder in zip(('on','off'),(live,ablation),strict=True)},
                      task=a0['task'],first_action_difference_ms=first,decision_before=decision_prefix,
                      sensor_through_first_action=through_event,same_physical_state=state_prefix,
                      same_raw_native_before=native_prefix,actual_routes=actors,events=events,
                      passed=proof,control_engineering_checkpoint_still_loaded=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('on',type=Path)
    p.add_argument('off',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    audit(a.on.resolve(),a.off.resolve(),a.output.resolve())
