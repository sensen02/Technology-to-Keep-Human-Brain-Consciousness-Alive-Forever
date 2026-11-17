"""Original helpers compiled once in new resume-repair body namespace."""
from adapter import namespace as _namespace
_original=_namespace('common.py')
globals().update({name:value for name,value in _original.items() if not name.startswith('__')})
