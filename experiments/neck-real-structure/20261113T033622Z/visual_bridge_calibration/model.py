"""Parent numerical backend. Exact bridge computation, brain-side visual scalar identity."""
from pathlib import Path
import importlib.util,sys
P=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261112T225331Z/neural_calibration/model.py');s=importlib.util.spec_from_file_location('_visual_neck_immutable_bridge_arch',P);original=importlib.util.module_from_spec(s);sys.modules[s.name]=original;s.loader.exec_module(original)
torch=original.torch
class NeuralBridge(original.NeuralBridge):
 def identity(self):
  d=super().identity();d.update(format='SOURCE_NECK_VISUAL_SCALAR_BRIDGE_V1',brain_input=['deliveredAN1','frozen_actual_RGB_visual_rawturn1'],visual_hidden=32,total_recurrent_state_with_motor=304,visual_feature_cache16_separate=True,vision_warm_hidden_held_zero=True,visual_sha256='996980806ac107137d6d5f4743de207fb3350e84a8a145dc5f52b8807398bb87');return d
Scan=original.Scan;make_scan=original.make_scan;python_rollout=original.python_rollout
def objective(out,target,weights,m,reference):
 turn,dn,an,recon,hidden=out;action=((turn-target).square()*weights).sum(0).mean();silent=m.decoder(dn.new_zeros((dn.shape[1],2))).square().mean();activity=dn.square().mean();named=dict(m.named_parameters());anchor=sum((named[k]-v).square().sum() for k,v in reference.items())/sum(v.numel() for v in reference.values())
 return action+.1*silent+.001*activity+.001*anchor,dict(turn_weighted_MSE=action,silent_DN_decoder_MSE=silent,DN_activity_MSE=activity,parent_trainable_anchor_MSE=anchor)
