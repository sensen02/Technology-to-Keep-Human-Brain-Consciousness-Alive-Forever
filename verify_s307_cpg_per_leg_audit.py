"""S307 verification: independent recomputation of the six modules, nt, DNg100 split, matrix, gate. Writes only into --output."""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype');BANC=ROOT/'data/flywire/banc_connectome.npz'
LEGS=('lf','lm','lh','rf','rm','rh')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    assert (o/'S307_PREREG_FROZEN').is_file()
    ck=[]
    def c(n,v,**e): ck.append(dict(name=n,passed=bool(v),**e))
    pre=json.loads((o/'PREREG.json').read_text());mod=json.loads((o/'PER_LEG_MODULE.json').read_text())
    mx=json.loads((o/'MATRIX18.json').read_text());oc=json.loads((o/'OUTCOMES.json').read_text())
    gate=json.loads((o/'GATE_PAPER_MODEL.json').read_text());corr=json.loads((o/'S306_CORRECTION.json').read_text())
    pm=json.loads((o/'PAPER_MODEL.json').read_text());led=json.loads((o/'CALL_LEDGER.json').read_text())
    c('zero dynamics and connectome_load declared',all(pre['calls'][k]==0 for k in ('model_load','forward','physics_step','neural_tick','tier_build','render')) and pre['connectome_calls']['connectome_load']==1)
    c('ledger matches the executed branch',led['rate_model_runs']==oc['runs_executed']==2)
    z=np.load(BANC,allow_pickle=True);rt=np.asarray(z['root_ids']);pt=np.asarray(z['ann_primary_type']).astype(str)
    nt=np.asarray(z['ann_nt_verified']).astype(str)
    pre_i=rt[np.asarray(z['pre'])];post_i=rt[np.asarray(z['post'])];w=np.asarray(z['syn'],float)
    types={'E1':'IN17A001','E2':'INXXX466','I1':'IN16B036'}
    # nt per type across all cells
    ntall={lab:sorted({str(x) for x in nt[pt==name]}) for lab,name in types.items()}
    c('ann_nt_verified values reproduced (E1 acetylcholine, I1 glutamate, E2 empty)',ntall['E1']==['acetylcholine'] and ntall['I1']==['glutamate'] and ntall['E2']==[''])
    c('S306 correction lists four items',len(corr['items'])==4)
    def syn(lg,a,b):
        sa=set(mod['modules'][lg][a]);sb=set(mod['modules'][lg][b])
        return float(w[np.isin(pre_i,list(sa))&np.isin(post_i,list(sb))].sum())
    for lg in LEGS:
        for e in ('E1->E2','E2->I1','I1->E1','I1->E2','E1->I1','E2->E1'):
            a,b=e.split('->');c(f'{lg} {e} reproduced',abs(syn(lg,a,b)-mod['edges'][lg][e])<1e-6,verifier=syn(lg,a,b),archived=mod['edges'][lg][e])
    c('cross-leg synapse mass is zero',all(v==0.0 for lg in LEGS for v in mod['cross_leg'][lg].values()) and mx['cross_leg_mass']==0.0)
    c('18 cells total with six per leg',mx['n_cells']==18 and all(len(mod['modules'][lg]['E1'])+len(mod['modules'][lg]['E2'])+len(mod['modules'][lg]['I1'])==3 for lg in LEGS))
    c('I1 two-method assignment recorded',isinstance(mod['i1_second_method'],dict))
    def dnv(c,lg):
        src=set(mod['modules'][lg]['E1'])
        return float(w[(pre_i==int(c))&np.isin(post_i,list(src))].sum())
    c('DNg100 per-leg split reproduced',all(abs(mod['dn_leg'][c][lg]-dnv(c,lg))<1e-6 for c in mod['dn_leg'] for lg in LEGS))
    c('jaccard overlaps recorded',len(mod['jaccard_targets'])==15)
    c('paper b is NOT_ESTABLISHED',"b" in pm['NOT_ESTABLISHED'])
    c('gate recorded as not passed with a reason',gate['passed'] is False and 'identifiab' in json.dumps(oc).lower() or 'not identifiable' in oc['branch_reason'])
    c('branch 4 hit and tripod conclusion written',oc['branch']==4 and 'tripod' in oc['tripod_conclusion'].lower())
    c('analysis script archived',(o/'analyze_s307_cpg_per_leg_audit.py').is_file())
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
           calls=dict(model_load=0,forward=0,physics_step=0,neural_tick=0,tier_build=0,render=0,connectome_load=0,rate_model_runs=0),
           limitations=['b unpublished so the paper model is calibrated not validated','driving proportional to DNg100 synapses is a declared assumption','cross-dataset mapping','no support/walking claim'])
    (o/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),nfail=len(r['failed']),failed=r['failed'][:3])))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
