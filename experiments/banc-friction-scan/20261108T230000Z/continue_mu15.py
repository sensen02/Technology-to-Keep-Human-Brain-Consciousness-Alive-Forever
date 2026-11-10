"""Continue the mu=1.5 model for 64 more updates in the same world."""
import sys, importlib.util
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('tf', str(HERE / 'train_friction.py'))
tf = importlib.util.module_from_spec(spec); sys.modules['tf'] = tf; spec.loader.exec_module(tf)
z = np.load(HERE / 'MODEL_mu15.npz')
orig = tf.Net.__init__
def init(self, rng):
    orig(self, rng)
    for k, key in (('w0', 'actor_w0'), ('b0', 'actor_b0'), ('w1', 'actor_w1'), ('b1', 'actor_b1'),
                   ('w2', 'actor_w2'), ('b2', 'actor_b2'), ('wc', 'critic_w'), ('bc', 'critic_b'), ('logstd', 'logstd')):
        self.p[k] = z[key].astype(np.float64).copy()
tf.Net.__init__ = init
tf.CFG['v_targets'] = [.05]
tf.main()
