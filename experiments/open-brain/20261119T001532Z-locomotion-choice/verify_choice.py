"""Verify photographed selection, matched cuts, and physically replayed response."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DT = .0001


def read(name):
    return json.loads((HERE / name / 'RESULT.json').read_text())


def choice(name):
    return np.load(HERE / name / 'CHOICE.npz', allow_pickle=False)


def arr(name, filename):
    return np.load(HERE / name / f'{filename}.npy', allow_pickle=False)


def ground(name):
    d = read(name)
    p = np.load(HERE / name / 'TRACE.npz', allow_pickle=False)
    x = p['trace'][:,d['qpos_columns']+1:d['qpos_columns']+3]
    horizontal = np.linalg.norm(x - np.array(d['fruit'][:2]),axis=1)
    return dict(min_target_horizontal_mm=float(horizontal.min()), low_food_intake=d['ingested'],
                path_length_mm=float(np.linalg.norm(np.diff(x,axis=0),axis=1).sum()),
                endogenous_dnp11_spikes=d['endogenous_dnp11_spikes'],
                takeoff_gate_s=d.get('takeoff_gate_s'), odor_outside_samples=d.get('odor_samples_outside_field',0))


def check_flight(side):
    neural, guide, fly, meal = [root + '_' + side for root in ('SELECT','GUIDE','FLY','MEAL')]
    n, g, b, m = map(read, (neural, guide, fly, meal))
    q = choice(neural)
    assert n['takeoff_gate_s'] is not None
    assert b['takeoff_s'] >= n['takeoff_gate_s'] - 1e-8
    assert b['takeoff_s'] - n['takeoff_gate_s'] < .00025
    assert b['visual_trigger_s'] == g['visual_trigger_s']
    assert b['landing_s'] >= g['visual_trigger_s'] + .0014 - 1e-8
    k = round(g['visual_trigger_s']/.0002)
    for tag, prefix in [('QPOS',k),('ACTION',k),('TRACE',k*4)]:
        assert np.array_equal(arr(guide,tag)[:prefix],arr(fly,tag)[:prefix])
    assert np.array_equal(arr(guide,'INITIAL'),arr(fly,'INITIAL'))
    assert np.array_equal(arr(guide,'EYE'),arr(fly,'EYE'))
    x = arr(fly,'TRACE')
    signal=dict(zip(b['columns'],x.T,strict=True))
    touch = b['first_fruit_contact_s']
    assert touch is not None and b['takeoff_s'] < touch
    air=(signal['time_s']>b['takeoff_s']+.035)&(signal['time_s']<touch)
    assert air.any() and np.all(signal['feet'][air] == 0)
    post=signal['time_s']>=touch
    assert not signal['floor_feet'][post].any() and not signal['nonfoot_contacts'][post].any()
    settled=(signal['perch_ready']>0)&(signal['fruit_feet']==6)
    edges=np.flatnonzero(np.diff(np.r_[False,settled,False].astype(int)))
    continuous=int(np.max(edges[1::2]-edges[::2],initial=0))
    dwell=float(continuous*b['native_dt_s'])
    assert dwell >= .30 and b['end_fruit_feet'] == 6
    assert b['external_force_max']==0 and m['intake']>=.07
    assert m['first_intake_s'] >= b['first_taste_s'] and abs(m['nutrient_balance_error'])<1e-9
    assert m['trace_sha256']==b['files']['TRACE.npy']
    path=HERE/('FLIGHT_NECK_'+side)/'COMMANDS.npz'
    d=np.load(path,allow_pickle=False)
    s=len(q['emitted'])
    assert np.array_equal(d['emitted'][:s,0,:2],q['emitted'][:,:2])
    assert np.array_equal(d['emitted'][:,0,:2],d['emitted'][:,1,:2])
    assert not d['arrived'][:,1,:2].any()
    first_spikes = [np.flatnonzero(q['emitted'][:,i])[0]*DT for i in range(2)]
    return dict(neural_source=neural, actual_body=fly, first_bilateral_arrival_s=[
                    float(np.flatnonzero(q['arrived'][:,i])[0]*DT) for i in range(2)],
                first_brain_dnp11_pulse_s=first_spikes, original_eon_spikes=q['emitted'][:,:2].sum(axis=0).tolist(),
                actual_bilateral_threshold_s=n['takeoff_gate_s'], physical_kick_s=b['takeoff_s'],
                actual_fruit_touch_s=touch, continuous_six_feet_s=dwell,
                ingested=m['intake'], first_meal_s=m['first_intake_s'],
                stable_windows=int(len(edges)//2))


def main():
    native={k:ground(k) for k in ('GROUND_L','AIR_L','AIR_R','NATIVE_REPLAY')}
    assert native['GROUND_L']['low_food_intake']>=.3
    for key in ('AIR_L','AIR_R','NATIVE_REPLAY'):
        assert native[key]['endogenous_dnp11_spikes']==[0,0]
        assert native[key]['low_food_intake']==0. and native[key]['path_length_mm']>2.
    for key in ('trace','emitted','arrived','ascending','input_events'):
        a,b=[np.load(HERE/n/'TRACE.npz',allow_pickle=False)[key] for n in ('AIR_L','NATIVE_REPLAY')]
        assert np.array_equal(a,b),key
    assert read('NATIVE_REPLAY')['dn_flight_arrivals']==[0,0,0,0]
    native_low = np.load(HERE/'GROUND_L/TRACE.npz',allow_pickle=False)['trace']
    adapted_low = np.load(HERE/'CONTROL_LOW/TRACE.npz',allow_pickle=False)['trace']
    assert np.array_equal(native_low[:len(adapted_low)],adapted_low)
    assert not choice('CONTROL_LOW')['emitted'].any()
    assert not choice('SELECT_MASK')['emitted'].any()
    assert np.array_equal(choice('SELECT_L')['emitted'],choice('SELECT_CUT')['emitted'])
    assert not choice('SELECT_CUT')['arrived'][:,:2].any()
    assert read('SELECT_CUT')['takeoff_gate_s'] is None
    physical_cut=read('FLY_CUT')
    assert physical_cut['takeoff_s'] is None and physical_cut['first_fruit_contact_s'] is None
    valid={side:check_flight(side) for side in ('L','R')}
    facts=dict(status='MODE_SELECTION_AND_PHYSICAL_REPLAY_PASS', eon_frozen=True,
               continuous_walk_to_flight=False, scheduled_dnp11=False,
               visual_empirical_fitted_head_adapter=True, non_eon_landing_trigger=True,
               native=native, paired=valid,
               controls=dict(low_with_adapter=ground('CONTROL_LOW'), vision_mask_zero=True,
                             cut_brain_spikes_unchanged=True, cut_no_flight=True,
                             repeated_native_five_arrays_identical=True))
    (HERE/'LOCOMOTION_CHOICE_CURRENT.json').write_text(json.dumps(facts,indent=2,allow_nan=False)+'\n')
    print(json.dumps(facts,indent=2))


if __name__=='__main__':
    main()
