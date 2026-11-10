"""Check an observed historical phase mismatch without modifying archived evidence."""
from pathlib import Path
import numpy as np,json,hashlib
OLD=Path('/home/sensen/Desktop/engine修复验收/s5-neural-joint-torque/20261105T182927Z')
O=Path('/home/sensen/Desktop/engine修复验收/s5-neural-translator/20261105T200021Z')
rows=[]
for phase in [1.3,1.225,1.25,1.35,1.375]:
    p=OLD/f'PHASE_{phase:.3f}'/'WARM_P0/NEURAL.npz';n=np.load(p)
    rows.append(dict(phase=phase,source=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),spikes_total=int(n['spikes'].sum()),current_abs_max=float(abs(n['currents']).max()),all_currents_zero=bool(not np.any(n['currents']))))
assert rows[1]['all_currents_zero'] and all(not r['all_currents_zero'] for i,r in enumerate(rows) if i!=1)
report=dict(finding='Historical PHASE_1.225 warmup had all sensory currents zero after best_sensory_cut from PHASE_1.300. DiagnosticTrial.sensory_cut was not reset before warm(). Later phases were reset by evaluate(sensory_cut=False).',affected='Historical1.225s phase is not a standard sensory-warmed phase comparison. Both arms shared an unintentionally sensory-deprived neural warmup; do not use it as clean phase generalization evidence.',evidence=rows,current_fix='New Runtime.warmup explicitly resets zero_sensory=False at start; no historical bytes or hashes changed.',importance='核心必需')
(O/'HISTORICAL_PHASE_CORRECTION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(rows,ensure_ascii=False),flush=True)
