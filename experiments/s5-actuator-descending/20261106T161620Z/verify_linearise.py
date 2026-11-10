import subprocess,sys,json,hashlib
from pathlib import Path
import numpy as np
O=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z/linearise/20261106T225055Z');p=json.loads((O/'PREREGISTRATION.json').read_text());OFFS=p['offsets_rad'];SPAN=1e-3;results=[]
for b in p['biases_nA']:
 folder=O/f'b{b}';subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('verify_polarity_sweep.py')),str(folder)],check=True)
 s=json.loads((folder/'SUMMARY.json').read_text());recs={ (r['condition'],r['offset_rad']):r for r in s['records']}
 cut=[recs['sensory_cut',0.],recs['sensory_cut',.06]]
 cut_identical=bool(np.isclose(cut[0]['net_drive'],cut[1]['net_drive'],atol=1e-12))
 d0=recs['normal',0.]['net_drive']
 base_ext=bool(d0<-1e-6)
 vals=[recs['normal',o]['net_drive'] for o in OFFS]
 strictly_decreasing=bool(all(vals[i]>vals[i+1] for i in range(len(vals)-1)))
 span=float(vals[0]-vals[-1])
 satur=bool(abs(vals[3]-vals[4])<1e-6)
 graded=bool(base_ext and strictly_decreasing and span>=SPAN and not satur and cut_identical)
 results.append(dict(bias_nA=b,net_drive_by_offset=[float(v) for v in vals],baseline_ext_drive_present=base_ext,strictly_decreasing=strictly_decreasing,span_rad_units=span,span_threshold=SPAN,terminal_saturation=satur,cut_offsets_identical=cut_identical,graded_restoring=graded))
selected=[r['bias_nA'] for r in results if r['graded_restoring']]
l=p['change_ledger'];assert all(hashlib.sha256(Path(l[k+'_file']).read_bytes()).hexdigest()==l[k+'_sha256'] for k in ['old','new'])
report=dict(importance='核心必需',verified=True,neural_runs=len(results)*7,physics_runs=0,biases=results,selected_biases=selected,criterion='preregistered GRADED_RESTORING; physical recovery never used for selection',outcome=('candidates:'+str(selected)) if selected else 'no bias produced a graded monotone restoring map; preregistered fallback is a descending-tonic sweep at fixed bias .03')
(O/'VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report))
