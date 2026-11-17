"""Exercise the actual source queue and neural endpoints on held intent sequences."""
from collections import Counter

import numpy as np
import torch

from bench_runtime import HERE, turn
from trained_link import TrainedLink


def run(profile, seed):
    link = TrainedLink(HERE/"CODEC.pt",profile,seed)
    tracker = []
    active_forbidden = (950, 1270)
    values = (0,1,-1,0,-1,1,0)
    span = 270
    for tick in range(len(values)*span):
        cut = active_forbidden[0]<=tick<active_forbidden[1]
        link.switch(cut)
        intended = values[tick//span]
        if tick%20 == 0:
            link.send(intended)
        if tick%5 == 0:
            output = link.action()
            tracker.append((tick,cut,intended,output))
        link.advance()
    table = np.asarray(tracker)
    assert np.all(table[table[:,1]==1,3] == 0)
    finish = (table[:,0] % span >= 80) & (table[:,1]==0) & ((table[:,0]<1270)|(table[:,0]>=1350))
    counts = Counter((int(x[2]),int(x[3])) for x in table[finish])
    wrong_direction = int(sum(n for (a,b),n in counts.items() if a*b==-1))
    assert not wrong_direction
    return dict(profile=profile,seed=seed, decoder=int(len(link.measurements)),
                drops=sum(row[1] is None for row in link.measurements), cut_all_zero=True,
                steady_flags_match=float(np.mean(table[finish,2]==table[finish,3])),
                steady_matrix=[[counts[a,b] for b in (-1,0,1)] for a in (-1,0,1)],
                wrong_direction=wrong_direction,
                confirmed_transport=link.transport.verify_final())


if __name__ == "__main__":
    torch.set_num_threads(1)
    evidence = [run(profile,seed) for profile,seed in (("clean",1090),("combined",5092),("combined",892105))]
    assert min(r["steady_flags_match"] for r in evidence)>=.95
    turn.stand.put(HERE/"LINK_PREFLIGHT.json",evidence)
    print([(r["profile"],r["drops"],r["steady_flags_match"]) for r in evidence])
