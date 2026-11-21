"""Search the force transient at handoff, then add proprioceptive and VNC candidate blocks."""
import json
import numpy as np
import refine as r
from bench import HERE, Bench


def main():
    r.B=Bench()
    r.OUT=HERE/'transfer'
    r.OUT.mkdir()
    with np.load(HERE/'strict/trained.npz') as a:
        r.P={k:a[k] for k in a.files}
    r.OBJECTIVE='handoff'
    r.HORIZON=25.
    b=r.B
    arms=b.motor_rows
    derivative=json.loads((HERE/'HANDOFF.json').read_text())['original']['slope']
    ranking=np.argsort(-np.abs(derivative),kind='stable')
    # Largest first-step load derivatives; each selected row has its own CMA coordinate.
    for j in range(2):
        selected=ranking[18*j:18*(j+1)]
        idx=np.flatnonzero(np.isin(arms,selected))
        groups=np.array([list(selected).index(arm) for arm in arms[idx]])
        r.batch('motor_gain',idx,groups,'impact'+str(j),15400+j,9,.85)
    r.HORIZON=150.
    r.batch('motor_gain',np.arange(len(arms)),arms%15,'family',15403,7,.5)
    io=b.io
    r.SCALE=np.where(io.angle, .6,.004)  # nA/rad and nA/(rad/s), signed gated transduction
    r.batch('feco_gain',np.arange(len(io.feco)),io.feco_feet*2+io.angle,'feco',15405,4,.8)
    prep=b.prepare(r.P)
    with np.load(HERE.parent/'20261120T074805Z-linear-interfaces/VNC_CANDIDATES.npz') as a:
        count=a['published_count']
    rank=count*prep[3][io.pre]
    used=np.flatnonzero(rank>0)
    chosen=used[np.argsort(-rank[used],kind='stable')[:768]]
    if len(chosen):
        r.SCALE=.0005*np.minimum(12,count[chosen])
        r.batch('vnc_gain_uS',chosen,io.post[chosen]%12,'vnc',15407,4,.85)
    np.savez_compressed(r.OUT/'trained.npz',**r.P)


if __name__=='__main__':
    main()
