"""Same learnedfusionGRU architecture; jointboundteacher exists only offline data generation."""
from pathlib import Path
import importlib.util,sys
P=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T045302Z/sensory_fusion_calibration/model.py');s=importlib.util.spec_from_file_location('_fusionagg_frozen_arch',P);old=importlib.util.module_from_spec(s);sys.modules[s.name]=old;s.loader.exec_module(old)
torch=old.torch
class NeuralBridge(old.NeuralBridge):
 def identity(self):
  d=super().identity();d.update(format='SOURCE_NECK_MULTISENSORY_AGGREGATION_BRIDGE_V1',odor_sha256=self.new_odor_sha256,odor_parameter_class_identity_unchanged=True,newodor_offset_repair=True,GRU_input3_column_mapping='all3db25columnsstrictcopy',runtime_fusion_rule=False);return d
Scan=old.Scan;make_scan=old.make_scan;python_rollout=old.python_rollout;objective=old.objective
def accumulated_backward(scan,m,x,u,y,w,ref):
 if x.shape[1]!=24:raise ValueError('Exactlyeffective24')
 total=0.;stats={}
 for start in (0,8,16):
  output=scan(x[:,start:start+8],u[:,start:start+8]);loss,terms=objective(output,y[:,start:start+8],w[:,start:start+8],m,ref);(loss/3.).backward();total+=float(loss.detach())/3.
  for k,v in terms.items():stats[k]=stats.get(k,0.)+float(v.detach())/3.
  del output,loss,terms
 return total,stats
