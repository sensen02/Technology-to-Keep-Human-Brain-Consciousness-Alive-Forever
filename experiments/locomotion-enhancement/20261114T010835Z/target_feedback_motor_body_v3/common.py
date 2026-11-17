"""Same original body helpers, new v3 ROOT/CAL namespace."""
from adapter import namespace as _namespace
_original=_namespace('common.py')
globals().update({name:value for name,value in _original.items() if not name.startswith('__')})
