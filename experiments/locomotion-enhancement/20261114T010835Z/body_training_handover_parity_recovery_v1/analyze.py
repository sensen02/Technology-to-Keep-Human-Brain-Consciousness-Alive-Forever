"""One-layer unchanged saved-array analysis with repaired metadata namespace."""
from pathlib import Path
import ast
import hashlib

ROOT=Path(__file__).resolve().parent
ORIGINAL=ROOT.parent/'body_training_handover_parity/analyze.py'
with ORIGINAL.open('rb') as stream:
    if hashlib.file_digest(stream,'sha256').hexdigest()!='b8b3cac78bf43465fb260c93fdca6059e9ab1474d9d72d79dfdb8e2cebb0c1d8':raise ValueError('Frozen original array-analysis source drift')
_namespace={'__name__':'_handover_recovery_exact_saved_array_analysis','__file__':str(ROOT/'analyze.py')}
exec(compile(ast.parse(ORIGINAL.read_text()),str(ORIGINAL)+'::unchanged_single_entry_new_metadata','exec'),_namespace)
globals().update({key:value for key,value in _namespace.items() if not key.startswith('__')})
if __name__=='__main__':main()
