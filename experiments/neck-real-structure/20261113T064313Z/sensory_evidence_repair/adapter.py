"""Bind the alreadyvalidated final multisensory renderer tree directly into new evidence globals."""
from pathlib import Path
import ast,importlib.util,sys
HERE=Path(__file__).resolve().parent;OLD=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T045302Z/sensory_evidence')
_sp=importlib.util.spec_from_file_location('_sensoryrepair_immutable_render_adapter',OLD/'adapter.py');_old=importlib.util.module_from_spec(_sp);sys.modules[_sp.name]=_old;_sp.loader.exec_module(_old)
def tree():return _old.tree()
def namespace(name):
 import contracts
 if name=='render.py':value,_changes=tree()
 elif name=='plot_sync.py':value=ast.parse((OLD/'plot_sync.py').read_text())
 else:raise ValueError('Registeredactualevidenceentry')
 ns=dict(__name__='_sensoryrepair_actual_'+name.replace('.','_'),__file__=str(HERE/name));exec(compile(value,str(HERE/name)+'::sameverified_multisensorycore_newnamespace','exec'),ns);ns.update({k:v for k,v in contracts.__dict__.items() if not k.startswith('_')});return ns
