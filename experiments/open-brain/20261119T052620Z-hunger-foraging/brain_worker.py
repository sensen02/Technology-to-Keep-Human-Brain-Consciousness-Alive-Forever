"""Trainable nutrient gate at the Eon interface. The frozen worker executes the pathway."""
import json
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
CHOICE=HERE.parent/'20261119T001532Z-locomotion-choice'
sys.path.insert(0,str(CHOICE))
from live_brain import Brain


def main():
    cfg=json.loads(sys.stdin.readline())
    fitted=json.loads((HERE/'GATE_FIT.json').read_text())
    model=Brain(cfg)
    update_neck=model.neck.update
    ground_emitted=np.zeros((model.ticks,len(model.neck.columns)),dtype=bool)
    ground_arrived=np.zeros_like(ground_emitted)
    delays=model.neck.model.neck_delay[model.neck.columns]

    def recorded_neck(tick, dn, front):
        ground_emitted[tick]=dn
        result=update_neck(tick,dn,front)
        ground_arrived[tick]=model.neck.last_dn
        return result

    model.neck.update=recorded_neck
    weights=fitted['weights']
    state=False
    receipts=[]
    print(json.dumps({'ready':True}),flush=True)
    for line in sys.stdin:
        obs=json.loads(line)
        # The translation layer receives a calibrated circulating proxy.
        # Crop, reserve, coordinates and stock IDs are outside this input.
        signal=obs.pop('circulating')
        fed=obs['contact']
        value=weights[0]+weights[1]*(signal-fitted['offset'])/fitted['scale']
        probability=float(1./(1.+np.exp(-np.clip(value,-50.,50.))))
        state=probability>=(fitted['search_off'] if state else fitted['search_on'])
        obs['seen']=obs['seen'] and state and not fed
        # Local tongue contact stops the present approach. Do not integrate
        # the duration of a meal into a later locomotion lockout.
        model.ground_taste=0.
        packet=model.advance(obs)
        receipt=(signal,probability,int(state),int(obs['seen']))
        receipts.append(receipt)
        packet['hunger']=receipt
        print(json.dumps(packet,separators=(',',':')),flush=True)
        if model.tick==model.ticks:
            break
    model.save(cfg['stream'])
    for j,d in enumerate(delays):
        assert not ground_arrived[:d,j].any()
        assert np.array_equal(ground_emitted[:-d,j],ground_arrived[d:,j])
    np.savez_compressed(Path(cfg['stream']).with_name('GROUND_DN.npz'),
                        emitted=ground_emitted,arrived=ground_arrived,delay_ticks=delays)
    np.save(Path(cfg['stream']).with_name('NUTRIENT_GATE.npy'),np.array(receipts),allow_pickle=False)


if __name__=='__main__':
    main()
