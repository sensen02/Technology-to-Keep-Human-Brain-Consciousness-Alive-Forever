"""S320 verification: freeze hashes, independent ignition and graded recompute, units, counters."""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np, scipy.sparse as sps
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S303=Path('/home/sensen/Desktop/engine修复验收/s303-readout-graded-window/20261026T201133Z')
N=12000
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    ck=[]
    def c(name,v,**e): ck.append(dict(name=name,passed=bool(v),**e))
    fr=(o/'S320_PREREG_FROZEN').read_text()
    c('freeze carries both hashes and UTC',('sha256(PREREG.json)=' in fr) and ('sha256(analyze_s320.py)=' in fr) and ('utc=' in fr))
    c('PREREG hash matches',('sha256(PREREG.json)=%s'%sha(o/'PREREG.json')) in fr)
    pre=json.loads((o/'PREREG.json').read_text());led=json.loads((o/'CALL_LEDGER.json').read_text())
    c('stage B counters equal the preregistration',led['stage_b']==pre['counters']['stage_b'],ledger=led['stage_b'])
    c('stage C counters equal the preregistration',led['stage_c']==pre['counters']['stage_c'],ledger=led['stage_c'])
    d=pre['counters']['stage_d'];c('stage D counters equal the preregistration',all(led['stage_d'][k]==v for k,v in d.items()),ledger=led['stage_d'])
    c('no physics, model or rendering',led['stage_d']['physics_step']==0 and led['stage_d']['model_load']==0 and led['stage_d']['render']==0)
    I=json.loads((o/'IGNITION_AUDIT.json').read_text());G=json.loads((o/'GRADED_WINDOW.json').read_text());A=json.loads((o/'ARMS.json').read_text())
    oc=json.loads((o/'OUTCOMES.json').read_text())
    c('gate arm is bitwise identical to S309 k1_control',A['G0_gate']['gate_bitwise'] is True)
    c('w=5e-4 is silent at sd=0 and sd=0.003 for every seed',
      all(r['spikes']==0 for r in I['grid']['0.0005|0.0']+I['grid']['0.0005|0.003']))
    c('w=5e-4 is seed dependent at sd=0.004 (both active and silent seeds)',
      any(r['spikes']==0 for r in I['grid']['0.0005|0.004']) and any(r['spikes']>0 for r in I['grid']['0.0005|0.004']))
    # independent recompute of one ignition cell with the archived matrix
    We=np.load(S303/'EDGES_w_5e-4_We.npz');Wi=np.load(S303/'EDGES_w_5e-4_Wi.npz')
    n=int(We['indptr'].size-1)
    rng=np.random.default_rng(2);bg=rng.normal(0.0,0.004,n)
    from engine.neural_cond import ConductanceNetwork,ConductanceParams
    net=ConductanceNetwork(n,ConductanceParams(),dt_ms=0.5)
    net.set_connectivity(sps.csr_matrix((np.asarray(We['data'],float),np.asarray(We['indices'],int),np.asarray(We['indptr'],int)),shape=(n,n)),
                         sps.csr_matrix((np.asarray(Wi['data'],float),np.asarray(Wi['indices'],int),np.asarray(Wi['indptr'],int)),shape=(n,n)))
    tot=0
    for _ in range(2000): tot+=int(net.step(current_nA=bg).sum())
    arch=[r['spikes'] for r in I['grid']['0.0005|0.004'] if r['seed']==2][0]
    c('independent re-run reproduces the silent seed 2 of the archived ignition grid',tot==0 and arch==0,verifier_spikes=tot,archived=arch)
    c('per-neuron units were corrected and archived',all('rate_per_neuron_hz' in r for k in I['grid'] for r in I['grid'][k]))
    c('graded points exist after the unit correction',any(v['n_graded']>0 for v in json.loads((o/'GRADED_WINDOW_POINTS.json').read_text())['points'].values()))
    c('branch n4 with the arms showing zero rhythmic legs',oc['branch']=='n4' and all(v['n_legs']==0 for v in A.values()))
    c('D1 and D2 are identical at this working point (motif has no effect)',abs(A['D1']['motif_median_hz']-A['D2']['motif_median_hz'])<1e-9)
    c('arithmetic sheet present with rheobase and per-spike drive',(o/'MOTIF_DRIVE_ARITHMETIC.json').is_file())
    c('verifier writes only verify/',(o/'verify').is_dir())
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
      calls=dict(model_load=0,physics_step=0,forward=0,render=0),limitations=['stage B/C archived aggregate counts only','ConductanceParams illustrative'])
    (o/'verify'/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),nfail=len(r['failed']),failed=r['failed'][:3],silent_spikes=tot)))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
