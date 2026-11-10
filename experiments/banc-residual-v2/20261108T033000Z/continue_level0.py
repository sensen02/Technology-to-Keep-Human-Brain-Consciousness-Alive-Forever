"""Amendment 2: continue level 0 (v_target 0.05) from the saved residual weights."""
import sys, json, importlib.util
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('tr', str(HERE / 'train_residual_v2.py'))
tr = importlib.util.module_from_spec(spec); sys.modules['tr'] = tr; spec.loader.exec_module(tr)
z = np.load(HERE / 'MODEL_RESIDUAL_V2.npz')
orig_init = tr.Net.__init__
def init(self, rng, w=z):
    orig_init(self, rng)
    for k, key in (('w0','actor_w0'),('b0','actor_b0'),('w1','actor_w1'),('b1','actor_b1'),
                   ('w2','actor_w2'),('b2','actor_b2'),('wc','critic_w'),('bc','critic_b'),('logstd','logstd')):
        self.p[k] = w[key].astype(np.float64).copy()
tr.Net.__init__ = init
tr.CFG['updates_per_level'] = 32
tr.CFG['v_targets'] = [.05]
tr.HERE = HERE
tr.main()
