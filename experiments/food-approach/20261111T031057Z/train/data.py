"""Immutable teacher namespace plus aligned BEFORE-action RH load loss mask."""
import importlib.util,sys,json,re
from pathlib import Path
import numpy as np
import torch
from model_io import HERE,ROOT,OLD,BASE,legacy,sha256
binding=sys.modules.get('model_io')
try:
 sys.modules['model_io']=legacy
 spec=importlib.util.spec_from_file_location('_rh_weighted_original_data',OLD/'data.py');original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
finally:
 if binding is None:sys.modules.pop('model_io',None)
 else:sys.modules['model_io']=binding
invariance=original.invariance
cached=original.cached
TEACHER=ROOT.parent/'20261110T191449Z/teacher/left_full'
MAPPING=ROOT.parent/'20261111T024558Z/JOINT_MAPPING.json'


def prepare(model):
 e,hashes=original.prepare(model)
 trace_path=TEACHER/'TRACE_1MS.npz';sample_path=TEACHER/'SAMPLES_5MS.npz';report_path=TEACHER/'RESULT.json'
 report=json.loads(report_path.read_text())
 if sha256(trace_path)!=report['trace']['sha256']:raise ValueError('teacher trace hash drift')
 with np.load(trace_path,allow_pickle=False) as z:
  times=z['time'].copy();loads=z['loads'].copy();trace_action=z['action'].copy();phase=z['control_source'].copy()
 with np.load(sample_path,allow_pickle=False) as z:sample_times=z['actual_time'].copy()
 indices=999+5*np.arange(2880)
 if times.shape!=(15400,) or loads.shape!=(15400,6) or sample_times.shape!=(2880,):raise ValueError('full matched teacher trace/sample counts required')
 clock_error=float(np.abs(times[indices]-sample_times).max())
 nominal_error=float(np.abs(sample_times-(1.+np.arange(2880)*.005)).max())
 if clock_error>1e-8 or nominal_error>1e-8:raise ValueError('before-action timestamp alignment failed')
 # Importing immutable physics does not instantiate Core or integrate.
 sys.path.insert(0,str(BASE))
 import physics
 if Path(physics.__file__).resolve()!=BASE/'physics.py':raise ValueError('wrong frozen physics module')
 W=float(physics.W)
 mapping=json.loads(MAPPING.read_text());names=mapping['joint_names']
 if mapping['RH_rows']!=list(range(28,35)) or not all(n.startswith('joint_RH') for n in names[28:35]):raise ValueError('RH7 actuator mapping changed')
 if tuple(physics.LEGS)!=('LF','LH','LM','RF','RH','RM'):raise ValueError('RHloadcolumn4 leg ordering differs')
 if not np.array_equal(trace_action[1000:],np.repeat(e['y'].numpy(),5,axis=0)):raise ValueError('teacher action sample alignment differs')
 mask=loads[indices,4]>=.1*W
 weights=torch.ones((2880,42),dtype=torch.float64);weights[torch.tensor(mask),28:35]=4.
 e.update(weights=weights,loaded_mask=torch.tensor(mask),before_indices=indices,teacher_weight_W=W)
 for p in (trace_path,MAPPING,BASE/'physics.py',BASE/'sources/repair_core.py'):hashes[str(p)]=sha256(p)
 alignment=dict(decision_rows=2880,trace_rows=15400,before_action_index_formula='999+5*k',all_clock_max_error_s=clock_error,all_nominal_clock_max_error_s=nominal_error,load_column=4,actuator_rows=list(range(28,35)),RH_names=names[28:35],weight_W_native=W,load_threshold_native=.1*W,post_loaded_count=int(mask[960:].sum()),post_weight_sum=float(weights[960:].sum()),original_post_error_cells=1920*42,physics_Core_constructions=0,physics_integration_steps=0,trace_sha256=sha256(trace_path),mapping_sha256=sha256(MAPPING))
 return e,hashes,alignment


def objective(mean,e,weighted=True):
 errors=(mean[960:].tanh()-e['y'][960:]).square()
 if not weighted:return errors.mean()
 w=e['weights'][960:]
 return (errors*w).sum()/w.sum()
