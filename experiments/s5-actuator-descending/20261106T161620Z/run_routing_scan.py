import subprocess,sys,json,hashlib
from pathlib import Path
D=Path(__file__).parent;R=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z');O=R/'routing/20261106T225055Z';O.mkdir(parents=True,exist_ok=True);assert not (O/'PREREGISTRATION.json').exists()
audit=json.loads((R/'descending/PATH_AUDIT.json').read_text());cand=audit['ranked_candidate_indices'];new=D/'routing_scan_20261106T225055Z.py'
p=dict(importance='核心必需',stage='C2 per-cell routing',cells=cand,bias_nA=.03,current_per_cell_nA=.1,offset_rad=0.,dt_ms=.0078125,duration_ms=150,neural_runs=len(cand),physics_runs=0,purpose='the pooled tonic drives only the extensor pool; measure which individual audited descending cells reach the flexor pool and which reach the extensor pool',classify='flex-only, ext-only, both, neither by tail-window counts; a reciprocal graded command later requires at least one flex-driving and one ext-driving cell',no_invented_edges=True,source_sha256=hashlib.sha256(new.read_bytes()).hexdigest())
(O/'PREREGISTRATION.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
for k in cand:subprocess.run([sys.executable,'-B',str(new),str(k)],check=True)
