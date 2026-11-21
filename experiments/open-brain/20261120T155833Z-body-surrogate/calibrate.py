from paths import *
from electrode import library,render,extract

roots=np.load(HERE/'data/10901_intact/nerve.npz')['roots'].tolist()
wave,chol=library(roots)
rng=np.random.default_rng(681239)
hits=np.zeros((2,2),int)
blank=0
for repeat in range(8):
    src=np.zeros((20000,19),bool)
    # Independent, known stimulation times; these are separate from the walking trials.
    ticks=np.arange(25,19950,50)+rng.integers(-10,11,399)
    units=np.tile([17,18],200)[:len(ticks)]
    rng.shuffle(units)
    src[ticks,units]=True
    obs=extract(render(src,wave,chol,rng),wave,chol)
    for i in (17,18):
        hits[i-17] += [int((src[:,i]*obs[:,i]).sum()),int(src[:,i].sum())]
    blank+=int(obs[:,17:][~src[:,17:]].sum())
stats=dict(correct_and_total=hits.tolist(),sensitivity=(hits[:,0]/hits[:,1]).tolist(),
           unmatched_an_calls=blank, duration_s=16.)
(HERE/'calibration.json').write_text(json.dumps(stats,indent=2))
print(stats)
