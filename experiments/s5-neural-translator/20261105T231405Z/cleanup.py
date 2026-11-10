"""Remove only byte-recorded redundant concatenations after exact reconstruction checks."""
from pathlib import Path
import numpy as np,json,hashlib
O=Path('/home/sensen/Desktop/engine修复验收/s5-neural-translator/20261105T231405Z')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def arrays(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
ledger=json.loads((O/'CLEANUP_LEDGER.json').read_text())
for number in [1,2]:
    target=O/'DATA'/f'training_union_{number}.npz';sources=[O/'DATA/closed_train.npz']+[O/'DATA'/f'aggregate_{i}.npz' for i in range(1,number+1)]
    original=arrays(target);parts=[arrays(p) for p in sources]
    assert all(np.array_equal(original[k],np.concatenate([part[k] for part in parts])) for k in original)
    ledger['deleted'].append(dict(path=str(target),sha256=sha(target),bytes=target.stat().st_size,classification='可再生临时',reason='Exact concatenation duplicates, no runtime or verifier reads this union file; actual component data retained and exact reconstruction confirmed before deletion.',reconstruct_from=[dict(path=str(p),sha256=sha(p)) for p in sources]));target.unlink()
(O/'CLEANUP_LEDGER.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n')
print('Removed two exactly reconstructible duplicate unions',flush=True)
