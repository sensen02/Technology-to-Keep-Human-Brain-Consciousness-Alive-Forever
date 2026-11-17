"""One stdlib namespace layer over frozen original calibration; only HERE/FORMAT/lineage differ."""
from pathlib import Path
import importlib.util,sys,types
HERE=Path(__file__).resolve().parent;ORIGINAL=HERE.parent/'sensory_fusion_calibration'
_spec=importlib.util.spec_from_file_location('_fusion_recovery_frozen_common',ORIGINAL/'common.py');_old=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=_old;_spec.loader.exec_module(_old)
_g=dict(_old.__dict__);_g.update(HERE=HERE,ORIGINAL=ORIGINAL,FORMAT='SOURCE_NECK_VISUAL_OLFACTORY_FUSION_RECOVERY_V1')
for _name,_value in list(_g.items()):
 if isinstance(_value,types.FunctionType):_g[_name]=types.FunctionType(_value.__code__,_g,_value.__name__,_value.__defaults__,_value.__closure__)
globals().update({k:v for k,v in _g.items() if not k.startswith('__')});_inherited_sources=_g['sources']
def sources():
 paths=_inherited_sources()+[ORIGINAL/n for n in ('PREREG.json','common.py','model.py','data.py','preflight.py','train.py','verify_saved.py','evaluate.py','run.py','codecheck.py','CODEONLY_CHECK.json','DATASET.json','SENSORY_REPLAY.json','train.npz','development.npz','JOBS.json','PREFLIGHT_EXECUTION.log','PIPELINE.log')]+[HERE/'RECOVERY_DATA_PROOF.json',HERE/'ORIGINAL_FAILURE.json']
 return sorted(set(paths),key=str)
_g.update(sources=sources)
