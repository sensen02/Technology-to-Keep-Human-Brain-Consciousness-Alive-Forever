"""Exactlyoriginal serialstageorchestration withnewrecoveryoutputroot; no heavy defaults."""
from pathlib import Path
import importlib.util,sys,types
HERE=Path(__file__).resolve().parent;ORIGINAL=HERE.parent/'sensory_fusion_calibration'
_s=importlib.util.spec_from_file_location('_fusion_recovery_original_orchestrator',ORIGINAL/'run.py');_old=importlib.util.module_from_spec(_s);sys.modules[_s.name]=_old;_s.loader.exec_module(_old);_g=dict(_old.__dict__);_g.update(HERE=HERE)
for _name,_v in list(_g.items()):
 if isinstance(_v,types.FunctionType):_g[_name]=types.FunctionType(_v.__code__,_g,_v.__name__,_v.__defaults__,_v.__closure__)
main=_g['main']
if __name__=='__main__':main()
