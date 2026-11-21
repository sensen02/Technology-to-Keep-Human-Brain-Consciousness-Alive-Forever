"""Training-only candidate: legacy motor readout, fine-tuned sensory weights."""
from common import *
from policy_plus import Policy as Legacy
class Policy(Legacy):pass

def read(path):
 saved=torch.load(path,weights_only=True,map_location='cpu');m=Policy()
 state={k:v for k,v in saved['state'].items() if k not in ('phase_points','phase_values')}
 m.load_state_dict(state);return m.eval()
