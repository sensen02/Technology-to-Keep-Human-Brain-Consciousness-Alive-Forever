"""ExactoriginalCNNGRU/production; only offlineobjective has episode scale argument."""
from common import module,V1,torch
_original=module('_neutral_original_visual_model',V1/'model.py')
VisualPolicy=_original.VisualPolicy
Scan=_original.Scan
make=_original.make
production=_original.production
def objective(turn,latent,target,weights,episode_scale):
 per_episode=(((turn-target).square()+.01*(latent-torch.atanh(target.clamp(-.95,.95))).square())*weights).sum(0)
 return (episode_scale*per_episode).mean()
