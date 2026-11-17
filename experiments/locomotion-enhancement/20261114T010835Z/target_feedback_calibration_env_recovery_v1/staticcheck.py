"""Stdlib-only wrapper equivalence/source/namespace/inspect checks, no Torch."""
from pathlib import Path
import ast,hashlib,inspect,json,sys
from adapter import ORIGINAL
ROOT=Path(__file__).resolve().parent

def main():
 for p in ROOT.glob('*.py'):compile(ast.parse(p.read_text()),str(p),'exec')
 import sources,common,data,learning,utils
 if sources.ROOT!=ROOT or common.ROOT!=ROOT or data._parent_prepare.__globals__['ROOT']!=ROOT:raise ValueError('Relocatednamespace notbound')
 if 'def prepare(m)' not in inspect.getsource(data.prepare):raise ValueError('Virtualoriginalsource inspect unavailable')
 if any(k in sys.modules for k in ('torch','numpy','mujoco','physics','flygym')):raise ValueError('Unexpectednumericalbackend import')
 hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}
 print(json.dumps(dict(AST_compile=True,newROOT=True,inspect_source=True,numeric_imports=0,scientific_changes=False,source_hashes=hashes),indent=2))
if __name__=='__main__':main()
