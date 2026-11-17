"""Exact300-parent deployment parameters/operations; resume-order metadata only.
Numerical model code is lazy: importing this file alone uses standardlibrary only.
"""
from pathlib import Path
import importlib.util
import sys

PARENT=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_calibration')
FORMAT='INDEPENDENT_MOTOR_TARGET_FEEDBACK_V1'
LINEAGE='RESUME300_ORDER_REPAIR_TO600'
_backend_cache=None


def backend():
 global _backend_cache
 if _backend_cache is None:
  name='_orderrepair_exact_original_deployment';spec=importlib.util.spec_from_file_location(name,PARENT/'model.py');value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value);_backend_cache=value
 return _backend_cache


def make_model(parent):
 """Deepcopy preserved300tensors; no initializer or forward or RNGdraws."""
 import copy
 source=backend()
 class ResumeMotor(source.IndependentMotor):
  def identity(self):
   value=super().identity();value.update(resume_lineage=LINEAGE,resume_parent_step=300,resume_parent_sha256='b562fcc3e39c2aa56381a87b17f1e80124f203fa1a73e3cdf8888778de119f14',deployment_operations='exact checkpoint300 source operations; orderrepair is training scan only');return value
 child=copy.deepcopy(parent);child.__class__=ResumeMotor
 child.verify_old();child.validate_architecture()
 return child


def make_scan(m):return backend().make_scan(m)
def make_residual_scan(m,script=True):return backend().make_residual_scan(m,script=script)
def state_digest(values):return backend().state_digest(values)
