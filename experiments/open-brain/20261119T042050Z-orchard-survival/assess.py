"""Independently check P1 orchard activity charges, receipts and matched controls."""
import argparse
import json
from pathlib import Path

import numpy as np

CASES = ('FRESH','EMPTY_GROUND','GUIDED')
ELEVATED = ('branch_outer','branch_inner','table_left','table_middle','table_right')
GROUND = ('ground_near','ground_far')
DELAYS = (13,13,14,14)


def meal_bouts(meals, cols):
    stamps = meals[:,cols['time_s']]
    intake = meals[:,cols['ingested']]
    reserve = meals[:,cols['reserve']]
    positive = np.flatnonzero(intake>0.)
    starts = positive[np.r_[True,np.diff(stamps[positive])>=.4]] if len(positive) else []
    events = []
    for start,end in zip(starts,list(starts[1:])+[len(meals)]):
        before = reserve[max(0,start-1)]
        gain = reserve[start:end].max()-before
        events.append(dict(time_s=float(stamps[start]),intake=float(intake[start:end].sum()),
                           peak_reserve_gain=float(gain)))
    return events


def read_case(directory):
    result = json.loads((directory/'RESULT.json').read_text())
    trace, meals = [np.load(directory/(name+'.npy'),allow_pickle=False) for name in ('TRACE','MEAL')]
    eye = [json.loads(line) for line in (directory/'EYES.jsonl').read_text().splitlines()]
    with np.load(directory/'NEURAL.npz',allow_pickle=False) as stream:
        emitted = stream['emitted']
        arrivals = stream['arrived']
        neck_levels = stream['values'][::2].copy()
        neural_delay = all(not arrivals[:delay,i].any() and
                           np.array_equal(arrivals[delay:,i],emitted[:-delay,i])
                           for i,delay in enumerate(DELAYS))
    cols = {name:i for i,name in enumerate(result['meal_columns'])}
    tc = {name:i for i,name in enumerate(result['columns'])}
    def energy(name):
        return meals[:,cols[name]]
    dt = .001
    rates = result['engineering_rates']
    modes = np.array(list(rates['activity'].values()))
    want = (rates['basal']+modes[energy('mode').astype(int)])*dt
    reserve_before = np.r_[result['reserve_initial'],energy('reserve')[:-1]]
    delta = energy('reserve')-reserve_before
    supplied = energy('absorbed')-energy('metabolized')
    expected_nutrient = result['reserve_initial']+.9*np.cumsum(energy('ingested'))
    nutrient_total = (energy('reserve')+energy('midgut_nutrient')+energy('crop_nutrient')
                      +np.cumsum(energy('excreted_nutrient')+energy('metabolized')))
    exhausted = np.flatnonzero(energy('reserve')<=1e-12)
    first_exhaustion = float(energy('time_s')[exhausted[0]]) if len(exhausted) else None
    contacts = energy('mouth_index').astype(int)
    actual = energy('ingested')
    bouts = meal_bouts(meals,cols)
    net_cycles = sum(b['peak_reserve_gain']>=.01 for b in bouts)
    case = result['case']
    success_names = GROUND if case=='FRESH' else ELEVATED if case=='EMPTY_GROUND' else ('branch_outer',)
    successful = [name for name in success_names if result['intake'][name]>=.01 and
                  (case=='FRESH' or (result['first_air_s'] is not None and
                                    result['fruit_stable_s'][name]>=.070))]
    takeoff, landing = result['takeoff_s'], result['landing_s']
    takeoff_index = round(takeoff/result['dt_s']) if takeoff is not None else None
    landing_index = round(landing/result['dt_s']) if landing is not None else None
    launches = np.flatnonzero(np.min(neck_levels[:,:2],axis=1)>=.22)
    wallet_checks = dict(
        per_interval_budget=bool(np.allclose(energy('metabolized')+energy('unmet'),want,rtol=0.,atol=1e-12)),
        paid_total=bool(np.allclose(energy('basal_paid')+energy('activity_paid'),energy('metabolized'),rtol=0.,atol=1e-12)),
        reserve_reconciliation=bool(np.allclose(delta,supplied,rtol=0.,atol=1e-12)),
        nutrient_reconciliation=bool(np.allclose(expected_nutrient,nutrient_total,rtol=0.,atol=2e-12)),
        stock_from_contact=all(abs(actual[contacts==i].sum()-result['intake'][name])<1e-12 and
                               abs(result['stocks_before'][name]-result['stocks_after'][name]
                                   -result['intake'][name])<1e-12
                               for i,name in enumerate(result['stocks_before'])),
        no_intake_without_touch=bool(np.all(actual[contacts<0]==0.)),
        no_credit_without_absorption=bool(np.all(delta[energy('absorbed')==0.]<=0.)),
        first_exhaustion_matches=first_exhaustion==result['exhausted_at_s'],
        native_neural_clock=result['native_steps']==len(trace)==2*result['neural_steps'],
        neck_delay=neural_delay,
        takeoff_trigger_from_neck=(takeoff_index==int(launches[0]) if len(launches) else takeoff_index is None),
        landing_trigger_from_neck=(landing_index is None or bool(np.min(neck_levels[landing_index,2:])>=.18)),
        no_external_force=bool(np.all(trace[:,tc['external_force']]==0.)),
    )
    return result,trace,meals,eye,dict(
        checks=wallet_checks,feeding_success=bool(successful),successful_fruit=successful,
        total_ingested=float(actual.sum()),bouts=bouts,net_recovery_cycles=net_cycles,
        stayed_energized=first_exhaustion is None,
        sustained_two_cycles=bool(net_cycles>=2 and first_exhaustion is None),
        first_exhaustion_s=first_exhaustion,minimum_reserve=float(energy('reserve').min()),
        peak_crop=float(energy('crop').max()),final_circulating=float(energy('circulating')[-1]),
        initial_hunger=1.-result['reserve_initial']/result['hunger_reference'],
        final_hunger=float(energy('hunger')[-1]),running_budget_by_mode={
            mode:float(energy('metabolized')[energy('mode')==i].sum())
            for i,mode in enumerate(rates['activity'])
        },
    )


def assess(root, cases=CASES):
    payload = {case:read_case(root/case) for case in cases}
    initial = [np.load(root/case/'INITIAL.npy',allow_pickle=False) for case in cases]
    common = {'same_initial_pose': all(np.array_equal(initial[0],saved) for saved in initial[1:])}
    if all(case in payload for case in CASES):
        fresh,empty,guided = [payload[case] for case in CASES]
        end = next((i for i,(left,right) in enumerate(zip(fresh[3],empty[3])) if left!=right),
                   min(len(fresh[3]),len(empty[3])))
        differential = fresh[3][end] if end<len(fresh[3]) else None
        start = round(end*.02/fresh[0]['native_dt_s'])
        geometry = [0,1,2,3,4,5,6,7,8,10,11,12]
        identical = np.all(fresh[1][:start,geometry]==empty[1][:start,geometry],axis=1)
        earliest = np.flatnonzero(~identical)
        common.update(
            only_ground_stocks_differ=all(fresh[0]['stocks_before'][k]==empty[0]['stocks_before'][k]
                                         for k in ELEVATED)
                and all(empty[0]['stocks_before'][k]==0.<fresh[0]['stocks_before'][k] for k in GROUND),
            guided_matches_empty_stocks=empty[0]['stocks_before']==guided[0]['stocks_before'],
            first_autonomous_packet_difference_s=differential['time_s'] if differential else None,
            matched_physical_prefix_s=float(fresh[1][earliest[0],0]) if len(earliest) else end*.02,
            same_motion_before_packet_divergence=not len(earliest),
        )
    valid = all(all(row[-1]['checks'].values()) for row in payload.values()) and all(
        value for value in common.values() if isinstance(value,bool))
    verdict = dict(status='INTEGRITY_PASS' if valid else 'INTEGRITY_FAIL',controls=common,
                   cases={case:row[-1] for case,row in payload.items()})
    (root/'ASSESSMENT.json').write_text(json.dumps(verdict,indent=2,allow_nan=False)+'\n')
    print(json.dumps(verdict,indent=2),flush=True)
    return valid


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--cases',nargs='+',choices=CASES,default=CASES)
    args = parser.parse_args()
    raise SystemExit(0 if assess(args.root,args.cases) else 1)
