"""Exact parent artificialbrain/queue architecture; only offline training objective changes."""
from common import module,OLD
_original=module('_aggregation_exact_parent_architecture',OLD/'model.py')
NeuralBridge=_original.NeuralBridge;Scan=_original.Scan;make_scan=_original.make_scan;python_rollout=_original.python_rollout;torch=_original.torch

def objective(out,teacher,normalized,weights,model):
 turn,dn,an,recon,states=out;action=((turn-teacher).square()*weights).sum(0).mean();feedback=(recon-normalized).square().mean();activity=dn.square().mean();silent=model.decoder(dn.new_zeros((dn.shape[1],2))).square().mean()
 return action+.01*feedback+.001*activity+.1*silent,dict(turn_weighted_MSE=action,feedback_MSE=feedback,DN_activity_MSE=activity,silent_DN_decoder_MSE=silent)
