"""Assess learned search, actual intake, and the matched resting control."""
import argparse
import json
from pathlib import Path

import numpy as np

NAMES=('HUNGRY','FULL','FADE')
HERE=Path(__file__).resolve().parent


def check_fitted_gate(receipts):
    fit=json.loads((HERE/'GATE_FIT.json').read_text())
    intercept,weight=fit['weights']
    logits=intercept+weight*(receipts[:,0]-fit['offset'])/fit['scale']
    expected=1./(1.+np.exp(-np.clip(logits,-50.,50.)))
    gate=np.zeros(len(expected),dtype=bool)
    previous=False
    for i,p in enumerate(expected):
        previous=p>=(fit['search_off'] if previous else fit['search_on'])
        gate[i]=previous
    return bool(np.allclose(receipts[:,1],expected,rtol=0.,atol=2e-15) and np.array_equal(receipts[:,2],gate))


def check_ledger(root,case,result,gates):
    ledger=np.load(root/case/'MEAL.npy',allow_pickle=False)
    available=np.r_[result['initial_reserve'],ledger[:-1,1]]
    energy=available+ledger[:,6]-ledger[:,7]
    last_circulation=np.r_[result['initial_circulating'],ledger[:-1,2]]
    expected_circulation=last_circulation-(np.minimum(ledger[:,1],1.)-last_circulation)*np.expm1(-.001/2.)
    packet_circulation=np.r_[result['initial_circulating'],ledger[19:-1:20,2]]
    by_name={name:float(ledger[ledger[:,4]==i,5].sum()) for i,name in enumerate(result['intake'])}
    return bool(np.allclose(energy,ledger[:,1],rtol=0.,atol=1e-12)
        and np.allclose(expected_circulation,ledger[:,2],rtol=0.,atol=1e-12)
        and np.array_equal(packet_circulation,gates[:,0])
        and np.all(ledger[ledger[:,4]<0,5]==0.)
        and all(abs(volume-result['intake'][name])<1e-12 for name,volume in by_name.items()))


def assess(root):
    result={name:json.loads((root/name/'RESULT.json').read_text()) for name in NAMES}
    pose=[np.load(root/name/'INITIAL.npy',allow_pickle=False) for name in NAMES]
    body={name:np.load(root/name/'TRACE.npy',allow_pickle=False) for name in NAMES}
    gates={name:np.load(root/name/'NUTRIENT_GATE.npy',allow_pickle=False) for name in NAMES}
    eyes={name:[json.loads(line) for line in (root/name/'EYES.jsonl').read_text().splitlines()]
          for name in NAMES}
    fade=result['FADE']
    full=result['FULL']
    hungry=result['HUNGRY']
    same_image=lambda a,b: a['candidates']==b['candidates'] and a['selection']==b['selection']
    checks=dict(
        same_body=all(np.array_equal(pose[0],current) for current in pose[1:]),
        identical_first_candidate=all(same_image(eyes[NAMES[0]][0],eyes[name][0]) for name in NAMES[1:]),
        same_taste_and_vision_while_resting=all(
            same_image(a,b) and a['contact']==b['contact']
            for a,b in zip(eyes['FULL'],eyes['FADE'])),
        exact_body_prefix_before_fade=bool(np.array_equal(body['FULL'],body['FADE'][:len(body['FULL'])])),
        no_external_force=all(r['max_external']==0. for r in result.values()),
        replay_fitted_hunger=all(check_fitted_gate(gates[name]) for name in NAMES),
        per_millisecond_ledger=all(check_ledger(root,name,result[name],gates[name]) for name in NAMES),
        full_no_search_or_stepping=full['first_search_s'] is None and full['ground_peak']==0.,
        hungry_search_and_eat_second=bool(hungry['first_walk_request_s']==0.
            and hungry['first_post_neck_walk_s'] is not None and hungry['intake']['ground_far']>=.01),
        fade_starts_after_drop=bool(fade['first_search_s'] is not None
            and fade['first_search_s']>=.3 and not gates['FADE'][0,2]
            and gates['FADE'][round(fade['first_search_s']/.02),0]<gates['FADE'][0,0]),
        fade_eats_second=bool(fade['first_search_s'] is not None and fade['first_post_neck_walk_s'] is not None
            and fade['first_post_neck_walk_s']>=fade['first_search_s']-1e-8
            and fade['intake']['ground_far']>=.01),
    )
    neck={}
    for name in NAMES:
        path=root/name/'GROUND_DN.npz'
        if not path.exists():
            continue
        with np.load(path,allow_pickle=False) as data:
            spikes,arrived=data['emitted'],data['arrived']
            delays=data['delay_ticks']
            timings=all(not arrived[:delay,i].any() and np.array_equal(spikes[:-delay,i],arrived[delay:,i])
                        for i,delay in enumerate(delays))
            neck[name]=dict(aligned=timings,emitted=int(spikes.sum()),arrived=int(arrived.sum()),
                            delay_ticks=delays.tolist())
    checks['recorded_neck_timings_match']=bool(neck) and all(row['aligned'] for row in neck.values())
    learned_start=checks['same_body'] and checks['identical_first_candidate'] and checks['same_taste_and_vision_while_resting'] and checks['replay_fitted_hunger'] and checks['full_no_search_or_stepping'] and checks['fade_starts_after_drop']
    verdict=dict(passed=all(checks.values()), learned_search_start_verified=learned_start,
                 checks=checks,neck=neck,cases=result)
    (root/'ASSESSMENT.json').write_text(json.dumps(verdict,indent=2)+'\n')
    print(json.dumps(verdict,indent=2),flush=True)
    return verdict['passed']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    raise SystemExit(0 if assess(parser.parse_args().root) else 1)
