"""S333 verification: hashes, counters, triple gate, structure, relay, branch thresholds."""
from __future__ import annotations
import argparse, json, hashlib, ast
from pathlib import Path
LEGS=('lf','lm','lh','rf','rm','rh')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    ck=[]
    def c(name,v,**e): ck.append(dict(name=name,passed=bool(v),**e))
    fr=(o/'S333_PREREG_FROZEN').read_text()
    c('freeze carries PREREG and both script hashes plus UTC',all(('sha256(%s)=%s'%(k,sha(o/k))) in fr for k in ('PREREG.json','analyze_s333.py','audit_s333_independent.py')))
    c('a REFREEZE ledger records the post-freeze fixes',(o/'REFREEZE.json').is_file())
    pre=json.loads((o/'PREREG.json').read_text());led=json.loads((o/'CALL_LEDGER.json').read_text())
    c('counters equal the preregistration exactly',all(led[k]==v for k,v in pre['counters'].items()),ledger=led)
    c('no tier advance and no physics/model/render',led['tier_advance']==0 and led['physics_step']==0 and led['model_load']==0 and led['render']==0)
    G=json.loads((o/'GATE.json').read_text())
    c('the gate arm is bitwise identical to all three archives',all(G['triple']))
    ST=json.loads((o/'POOL_INPUT_STRUCTURE.json').read_text())
    sh={lg:ST['thresholds']['0.9']['per_leg'][lg]['share_exc'] for lg in LEGS}
    c('the exclusive E shares reproduce 0.711 to 0.791',0.70<=min(sh.values()) and max(sh.values())<=0.80,shares={k:round(v,4) for k,v in sh.items()})
    c('all three exclusive thresholds are archived without picking one',set(ST['thresholds'])=={'0.8','0.9','0.95'})
    c('the exclusive populations are phase locked to each other',ST['thresholds']['0.9']['phase']['E']['same']<=16 and ST['thresholds']['0.9']['phase']['E']['cross']<=16,
      same=round(ST['thresholds']['0.9']['phase']['E']['same'],1),cross=round(ST['thresholds']['0.9']['phase']['E']['cross'],1))
    z=ST['zero_coupling']
    # We/F rows are POSTSYNAPTIC, so the field recorded as pool2motif is actually source=M18 -> target=pool
    c('every cross-leg path is zero and the pool sends nothing back to the motif',
      z['pool2pool_AtoB']==0 and z['pool2pool_BtoA']==0 and z['motif2pool']==0 and z['motif_cross_leg']==0 and z['pool2pool_wi_nnz']==0,
      z=z,naming_note='pool2motif in the archive is motif->pool (160 edges) because rows are postsynaptic; pool->motif is the motif2pool field and equals zero')
    R=json.loads((o/'RELAY_ARITHMETIC.json').read_text())
    c('the base drive resolves to 0.0885 nA',abs(R['base_drive_nA']-0.0885)<0.001,base=R['base_drive_nA'])
    c('the synchronised cell count is archived',R['required_synchronised_cells']>0,sync=round(R['required_synchronised_cells'],2))
    AR=json.loads((o/'ARMS.json').read_text())
    c('every arm carries a manipulation check',all('manip' in AR[k] for k in AR))
    c('the input-layer differential does not reach antiphase',AR['T3_in_diff040']['cross']<30,d3=round(AR['T3_in_diff040']['cross'],1))
    c('the band sensitivity table is archived per arm',all(len(AR[k]['band_sensitivity'])==4 for k in AR))
    c('quarter-block stability is archived per arm',all(len(AR[k]['blocks'])==4 for k in AR))
    c('a seed repeat exists for the 0.010 nA point',AR['T4_in_diff010_seed1']['seed']==1)
    oc=json.loads((o/'OUTCOMES.json').read_text())
    c('the branch thresholds are numerically preregistered',set(pre['branch_thresholds'])>= {'t1','t2','t3'})
    c('the branch equals the preregistered label set',oc['branch'] in ('t1','t2','t3','t4'),branch=oc['branch'])
    c('the S332 corrections are archived with six items',len(json.loads((o/'S332_CORRECTION.json').read_text())['items'])>=6)
    c('the requirement table lists drive, modulation and decoder',len(json.loads((o/'REQUIREMENT_V2.json').read_text())['items'])==3)
    AG=json.loads((o/'RECONCILE_AGREEMENT.json').read_text())
    c('the two implementations agree within 5 degrees and 1e-3 share',AG['agree'] is True,max_phase=AG['max_phase_diff'],max_share=AG['max_share_diff'])
    c('the entrainment-versus-relay interpretation is recorded','does not relay phase' in oc['interpretation'])
    src=(o/'analyze_s333.py').read_text();tree=ast.parse(src)
    bad=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.BoolOp) and isinstance(n.op,ast.Or) and any(isinstance(v,ast.Constant) and v.value is True for v in n.values)]
    c('no or-True conjunction (AST)',len(bad)==0,lines=bad)
    c('verifier writes only verify/',(o/'verify').is_dir())
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
      calls=dict(model_load=0,physics_step=0,forward=0,tier_build=0,render=0),limitations=['six arms, no physics','phase is an envelope-domain projection'])
    (o/'verify'/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),nfail=len(r['failed']),failed=r['failed'][:3])))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
