"""Frozen resume qualification relocated to actual CALv4 plus pinned metadata schema."""
from adapter import prepare_namespace
_original=prepare_namespace()
globals().update({name:value for name,value in _original.items() if not name.startswith('__')})
if __name__=='__main__':main()
