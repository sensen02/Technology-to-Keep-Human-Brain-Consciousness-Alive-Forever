"""Qualify live walking/flight choices and their anatomical cut contrast."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CASES = ('HIGH', 'LOW', 'NO_ADAPTER', 'CUT')


def read(case, root=HERE):
    run = root / case
    result = json.loads((run/'RESULT.json').read_text())
    traces = dict(zip(result['columns'], np.load(run/'TRACE.npy',allow_pickle=False).T,strict=True))
    with np.load(run/'NEURAL.npz',allow_pickle=False) as events:
        values = {k:events[k] for k in events.files}
    return result, traces, values


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def longest(mask, dt):
    edges = np.flatnonzero(np.diff(np.r_[False,mask,False].astype(int)))
    return float(np.max(edges[1::2]-edges[::2],initial=0)*dt)


def run(root):
    records = {name:read(name, root) for name in CASES}
    initial = np.load(root/'HIGH/INITIAL.npy',allow_pickle=False)
    gates = {}
    data = {}
    for case,(result,trace,events) in records.items():
        assert result['external_force_max']==0.
        assert np.array_equal(initial,np.load(root/case/'INITIAL.npy',allow_pickle=False))
        assert abs(1.8-result['fruit_remaining']-result['intake']) < 1e-9
        dnp11_memory = events['emitted'][:,:2]
        arrivals = events['arrived'][:,:2]
        if case!='CUT':
            assert not arrivals[:13].any() and np.array_equal(arrivals[13:],dnp11_memory[:-13])
        good=(trace['fruit_feet']==6)&(trace['floor_feet']==0)&(trace['nonfoot']==0)&(trace['landing_phase']==4)
        taste=trace['mouth_contact']!=0
        raised=result['mode']=='elevated'
        raw_gate=np.flatnonzero(events['values'][:,:2].min(axis=1)>=.22)
        air=result['first_air_s']
        touch=result['first_fruit_contact_s']
        across=longest(good,result['native_dt_s'])
        x=trace['x_cm']
        y=trace['y_cm']
        data[case]=dict(intake=result['intake'],first_intake_s=result['first_intake_s'],
                        takeoff_s=result['takeoff_s'],first_air_s=air,touch_s=touch,
                        first_bilateral_threshold_s=float(raw_gate[0]*.0001) if len(raw_gate) else None,
                        brain_pulses=dnp11_memory.sum(axis=0).tolist(),
                        neck_arrivals=arrivals.sum(axis=0).tolist(),
                        max_height_mm=10*result['max_height_cm'],
                        final_support=result['last_support'],continuous_six_s=across,
                        minimum_horizontal_distance_to_food_mm=float(10 * np.min(np.hypot(
                            x-result['geometry']['x_cm'],y-result['geometry']['y_cm']))),
                        file_sha256={name:sha(root/case/name) for name in (
                            'INITIAL.npy','QPOS.npy','TRACE.npy','OBS.npy','NEURAL.npz','RESULT.json')})
        if case=='HIGH':
            gates['elevated_eon_takeoff']=bool(len(raw_gate) and raw_gate[0]*.0001 <= result['takeoff_s']+1e-8
                    and result['takeoff_s']-raw_gate[0]*.0001 < .00025 and air is not None and touch is not None
                    and air<touch and min(dnp11_memory.sum(axis=0))>=2 and raised)
            gates['elevated_perch_and_intake']=bool(across>=.30 and result['last_support']==[0,6,0]
                    and result['intake']>=.07 and result['first_intake_s']>touch
                    and taste.any() and np.any(events['mn9']>0))
        elif case=='LOW':
            observation=np.load(root/case/'OBS.npy',allow_pickle=False)
            saturated=observation[(observation[:,2]==1) & (observation[:,4]>=254)]
            gates['clipped_low_target_remains_visible_without_flight_evidence']=bool(
                len(saturated) and np.all(saturated[:,1]==0.))
            gates['low_walk_and_intake']=bool(not dnp11_memory.any() and not len(raw_gate)
                    and result['takeoff_s'] is None and not raised and result['intake']>=.07
                    and x.max()>.70*result['geometry']['x_cm'] and np.any(events['mn9']>0))
        elif case=='NO_ADAPTER':
            gates['elevated_without_adapter_walks']=bool(not dnp11_memory.any()
                    and not len(raw_gate) and result['takeoff_s'] is None and raised and result['intake']==0.
                    and x.max()>.30*result['geometry']['x_cm'])
        else:
            pos=records['HIGH'][2]['emitted'][:,:2]
            k=round(records['HIGH'][0]['takeoff_s']/.0001)
            gates['cut_preserves_sources_blocks_takeoff']=bool(np.array_equal(pos[:k],dnp11_memory[:k])
                    and dnp11_memory.any() and not arrivals.any() and not len(raw_gate)
                    and result['takeoff_s'] is None and raised and result['intake']==0.)
    prior=root/'HIGH_BEFORE_CLIP_GATE'
    if prior.is_dir():
        current=records['HIGH'][2]
        with np.load(prior/'NEURAL.npz',allow_pickle=False) as old:
            neural_exact=all(np.array_equal(old[k],current[k])
                             for k in ('emitted','arrived','values','ground','pre','mn9','feed'))
        qpos_exact=np.array_equal(np.load(prior/'QPOS.npy',allow_pickle=False),
                                  np.load(root/'HIGH/QPOS.npy',allow_pickle=False))
        gates['high_physical_and_neural_replay_identical']=bool(neural_exact and qpos_exact)
    s=dict(status='PASS_SHARED_BODY_LOCOMOTION_CHOICE' if all(gates.values()) else 'PARTIAL_SHARED_BODY_LOCOMOTION_CHOICE',
           same_initial_physical_state=True,closed_loop_block_s=.02,scheduled_dnp11=False,brain_weight_updates=0,
           gates=gates,results=data)
    (root/'LIVE_CHOICE_CURRENT.json').write_text(json.dumps(s,indent=2,allow_nan=False)+'\n')
    print(json.dumps(s,indent=2,allow_nan=False))
    return all(gates.values())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=HERE)
    raise SystemExit(0 if run(parser.parse_args().root) else 1)
