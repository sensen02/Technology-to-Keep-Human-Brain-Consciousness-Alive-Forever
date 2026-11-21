"""Apply an audited Eon weight branch and run the original BANC dynamics."""
import json
import os
from neck_profile import configure
from pathlib import Path
import sys

import numpy as np

PAIR=Path(__file__).resolve().parent.parent/'20261119T073818Z-sac-terrain-brain-pair'
sys.path.insert(0,str(PAIR))
from brain_probe import digest, patchable

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'20261119T001532Z-locomotion-choice'))
from live_brain import Brain


def main():
    settings=json.loads(sys.stdin.readline())
    model=Brain(settings)
    profile = Path(os.environ['P1_OMNISIM_PROFILE'])
    mode = os.environ['P1_OMNISIM_MODE']
    blocked, applied = configure(model.neck.model, profile, mode)
    output = Path(settings['stream']).parent
    (output / 'APPLIED.json').write_text(json.dumps(applied, indent=2) + '\n')
    calibration=json.loads(Path(settings['calibration']).read_text())
    weights=model.brain.outgoing.data
    assert digest(weights)==calibration['original_weights_sha256']
    if settings['arm']=='tuned':
        indices,edges=patchable(model.index,model.brain.outgoing,model.inputs,model.neck)
        assert edges==calibration['edges']
        weights[indices]*=calibration['tuned_gain']
        assert digest(weights)==calibration['tuned_weights_sha256']
    proposed=(0.,0.)
    original_rates=model.head.rates
    original_update=model.neck.update
    model.head.rates=lambda _pace,_turn,odor: original_rates(*proposed,odor)
    n=model.ticks
    emitted=np.zeros((n,len(model.neck.columns)),dtype=bool)
    arrived=np.zeros_like(emitted)
    ascending=np.zeros((n,2),dtype=bool)

    def capture(tick,dn,front):
        emitted[tick]=dn
        response=original_update(tick,dn,front)
        arrived[tick]=model.neck.last_dn
        ascending[tick]=model.neck.last_an
        return response

    model.neck.update=capture
    print(json.dumps(dict(ready=True,weights=digest(weights))),flush=True)
    for line in sys.stdin:
        row=json.loads(line)
        if row.get('finished'):
            break
        proposed=row.pop('intent')
        nutrient=row.pop('circulating')
        row['seen']=nutrient<.295 and (proposed[0]>.008 or abs(proposed[1])>.015)
        # Preserve the archived ground feeding behavior.
        model.ground_taste=0.
        first=model.tick
        packet=model.advance(row)
        packet['arrived_an']=ascending[first:model.tick].sum(axis=0).tolist()
        print(json.dumps(packet,separators=(',',':')),flush=True)
        if model.tick==n:
            break
    n=model.tick
    emitted,arrived,ascending=emitted[:n],arrived[:n],ascending[:n]
    delays=model.neck.model.neck_delay[model.neck.columns]
    for i,d in enumerate(delays):
        if blocked[model.neck.columns[i]]:
            assert not arrived[:,i].any()
            continue
        assert not arrived[:d,i].any()
        assert np.array_equal(emitted[:-d,i],arrived[d:,i])
    np.savez_compressed(settings['stream'],emitted=emitted,arrived=arrived,
                       ascending=ascending,delays=delays,roots=model.neck.root_names,blocked=blocked[model.neck.columns],drive=model.ground[:n],
                       feeding=model.feed[:n],weights=digest(weights))


if __name__=='__main__':
    main()
