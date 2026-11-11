"""Reuse immutable teacher preparation with its actual original namespace."""
import importlib.util,sys
from model_io import OLD,legacy
old_binding=sys.modules.get('model_io')
try:
 sys.modules['model_io']=legacy
 spec=importlib.util.spec_from_file_location('_continued_positive_original_data',OLD/'data.py');original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
finally:
 if old_binding is None:sys.modules.pop('model_io',None)
 else:sys.modules['model_io']=old_binding
prepare=original.prepare
cached=original.cached
invariance=original.invariance
