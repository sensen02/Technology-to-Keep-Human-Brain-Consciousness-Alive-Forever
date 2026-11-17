"""Original body helpers, direct single-layer newROOT/CAL/final900 namespace."""
from adapter import namespace as _namespace
_original=_namespace('common.py')
globals().update({name:value for name,value in _original.items() if not name.startswith('__')})
