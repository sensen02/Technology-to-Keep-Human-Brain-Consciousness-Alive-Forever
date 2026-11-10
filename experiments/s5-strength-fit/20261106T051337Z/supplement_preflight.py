"""Additive post-freeze pre-search identity gate; original frozen files unchanged."""
import sys,pickle
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run as r
p=r.R/'run_integrated_repaired_closure.py'
if sys.argv[1]=='freeze':
    assert not (r.O/'SUPPLEMENT_INPUT_HASHES_BEFORE.json').exists()
    r.dump(r.O/'SUPPLEMENT_INPUT_HASHES_BEFORE.json',{str(p):r.sha(p),str(Path(__file__).resolve()):r.sha(__file__)})
    r.dump(r.O/'SUPPLEMENT_PREREGISTRATION.json',dict(reason='Parent source review after main freeze identified transitive body identity helper and explicit encoder comparison; additive gate before search, no search design change or frozen source rewriting',helper=str(p),gate='every cached encoder field exact old/fresh; main and supplemental hashes unchanged before and after search'))
else:
    hashes=r.js(r.O/'SUPPLEMENT_INPUT_HASHES_BEFORE.json');assert all(r.sha(k)==v for k,v in hashes.items())
    with Path(r.js(r.ARCH/'DATA/CACHE_IDENTITY.json')['file']).open('rb') as f:old=pickle.load(f)
    with (r.O/'DATA/PHYSICAL_CACHE.pkl').open('rb') as f:new=pickle.load(f)
    for ph,c in old.items():assert r.exact(vars(c['enc']),vars(new[ph]['enc'])),f'encoder mismatch {ph}'
    r.dump(r.O/('SUPPLEMENT_'+sys.argv[1].upper()+'.json'),dict(passed=True,encoder_exact_phases=sorted(old),hashes=hashes))
    print('SUPPLEMENT EXACT ENCODER',len(old),flush=True)
