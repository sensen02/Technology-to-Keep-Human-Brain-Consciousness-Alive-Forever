"""Exact inherited saved288 contracts, explicitly bound to new B-root and behavior."""
from adapter import contract_namespace
_BOUND=contract_namespace()
globals().update({key:value for key,value in _BOUND.items() if not key.startswith('__')})
