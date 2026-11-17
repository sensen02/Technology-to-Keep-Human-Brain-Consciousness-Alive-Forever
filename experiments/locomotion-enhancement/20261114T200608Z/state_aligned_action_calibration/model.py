"""Identity-only 41fa600 migration; deployment operations/tensors unchanged."""
from pathlib import Path
import importlib.util,sys
PARENT=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_order_repair_v4')
FORMAT='INDEPENDENT_MOTOR_STATE_ALIGNED_ACTION_V1'
PARENT_FORMAT='INDEPENDENT_MOTOR_TARGET_FEEDBACK_V1'
PARENT_SHA='41fa1e2232f5143fec0a3ebb5d506c819af0ba084b3a05041e829fd1477b906b'
LINEAGE='RESUME600_STATE_ALIGNED_ACTION_TO900'
_backend_cache=None

def backend():
 global _backend_cache
 if _backend_cache is None:
  spec=importlib.util.spec_from_file_location('_statealigned_exact_v4_deployment',PARENT/'model.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);_backend_cache=m
 return _backend_cache

def make_model(parent):
 import copy
 if parent.identity().get('format_version')!=PARENT_FORMAT or parent.identity().get('state_dims')!=[128,64,64,96] or tuple(parent.encoder.weight.shape)!=(96,412) or tuple(parent.head.weight.shape)!=(42,192):raise ValueError('Strictactual41fa412/352/8 parentrequired')
 parent.verify_old();parent.validate_architecture()
 class StateAlignedMotor(type(parent)):
  def identity(self):
   value=super().identity();value.update(format_version=FORMAT,resume_lineage=LINEAGE,resume_parent_step=600,resume_parent_sha256=PARENT_SHA,deployment_operations='exact41fa600 production operations; newFORMAT/lineage only',training_scope='statealigned action calibration; no body/realneck acceptance');return value
 child=copy.deepcopy(parent);child.__class__=StateAlignedMotor;child.verify_old();child.validate_architecture();return child

def make_scan(m):return backend().make_scan(m)
def make_residual_scan(m,script=True):return backend().make_residual_scan(m,script=script)
def state_digest(values):return backend().state_digest(values)
