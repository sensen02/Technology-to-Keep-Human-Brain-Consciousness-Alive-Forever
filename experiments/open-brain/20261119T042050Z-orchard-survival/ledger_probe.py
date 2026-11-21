"""Numerical positive control with two explicitly prescribed, valid contacts."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
from engine.embodied.feeding import EMPTY, FeedingBody, FeedingState, Food


def probe(result):
    rates=result['engineering_rates']
    engine=FeedingBody(basal_use=rates['basal'])
    initial=result['reserve_initial']
    output={}
    for mode in ('rest','walk','flight'):
        stocks=[Food(.9,.81), Food(1.1,.99)]
        intake=[0.,0.]
        state=FeedingState(EMPTY,reserve=initial,circulating=initial)
        ledger=[]
        for i in range(12000):
            patch=0 if i<2000 else 1 if 6000<=i<8000 else None
            contact=patch is not None
            food=stocks[patch] if contact else EMPTY
            receipt=engine.advance(replace(state,patch=food),.001,feed=contact,
                                   contact=contact,activity_use=rates['activity'][mode])
            state=receipt.state
            if contact:
                stocks[patch]=state.patch
                intake[patch]+=receipt.ingested.volume
            ledger.append(state.reserve)
        for old,food,volume in zip((.9,1.1),stocks,intake,strict=True):
            assert abs(old-food.volume-volume)<1e-12
        peak=[max(ledger[:6000])-initial,max(ledger[6000:])-ledger[5999]]
        output[mode]=dict(intake_volume=intake,reserve_final=ledger[-1],
                          peak_reserve_gains=peak,
                          first_zero_s=next(((i+1)*.001 for i,v in enumerate(ledger) if v<=1e-12),None))
    assert all(v>.01 for v in output['rest']['peak_reserve_gains'])
    assert all(v>.01 for v in output['walk']['peak_reserve_gains'])
    assert output['flight']['first_zero_s'] is not None
    return dict(scope='prescribed contacts only, without animal policy, physics or neural activity',
                contact_windows_s=[[0.,2.],[6.,8.]],horizon_s=12.,constant_rate_controls=output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fresh_result',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    result=probe(json.loads(args.fresh_result.read_text()))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
