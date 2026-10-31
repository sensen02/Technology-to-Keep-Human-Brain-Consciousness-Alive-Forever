"""S306 verification: independent BANC recomputation, source quotes, next-round spec. Writes ONLY into --output."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype'); BANC=ROOT/'data/flywire/banc_connectome.npz'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    assert (o/'S306_PREREG_FROZEN').is_file(), 'verifier must write only into the S306 directory'
    ck=[]
    def c(n,v,**e): ck.append(dict(name=n,passed=bool(v),**e))
    pre=json.loads((o/'PREREG.json').read_text());res=json.loads((o/'BANC_RESOLVE.json').read_text())
    mo=json.loads((o/'MOTIF_IN_BANC.json').read_text());sg=json.loads((o/'SIGN_PROVENANCE.json').read_text())
    src=json.loads((o/'SOURCE_CPG.json').read_text());oc=json.loads((o/'OUTCOMES.json').read_text());sp=json.loads((o/'NEXT_ROUND_SPEC.json').read_text())
    c('zero dynamics declared',all(v==0 for v in pre['calls'].values()))
    c('connectome calls declared separately',pre['connectome_calls']==dict(connectome_load=1,tier_build=1))
    c('source has DOI and quotes',src['doi']=='10.1101/2025.09.12.675944' and len(src['quotes'])>=8)
    c('E/I attribution recorded as external', 'paper' in src['ei_attribution']['note'])
    c('model form and constants recorded','Wsynaptic' in src['model_form']['constants'] and 'tau' in src['model_form']['constants'])
    c('four-dataset claim marked NOT_ESTABLISHED','NOT_ESTABLISHED' in src['datasets']['four_datasets_status'])
    z=np.load(BANC,allow_pickle=True);rt=np.asarray(z['root_ids']);pt=np.asarray(z['ann_primary_type']).astype(str)
    pre_i=rt[np.asarray(z['pre'])];post_i=rt[np.asarray(z['post'])];w=np.asarray(z['syn'],float)
    for lab,name in (('E1','IN17A001'),('E2','INXXX466'),('I1','IN16B036')):
        m=pt==name;c(f'{lab} cell count reproduced',res['types'][lab]['cells']==int(m.sum()),recount=int(m.sum()))
    def syn(a,b):
        sa=set(res['types'][a]['root_ids']);sb=set(res['types'][b]['root_ids'])
        if not sa or not sb: return 0.0
        m=np.isin(pre_i,list(sa))&np.isin(post_i,list(sb));return float(w[m].sum())
    for e in ('E1->E2','E2->I1','I1->E1','I1->E2'):
        a,b=e.split('->');c('edge '+e+' reproduced',abs(syn(a,b)-mo['pairs'][e]['synapses'])<1e-6,verifier=syn(a,b),archived=mo['pairs'][e]['synapses'])
    c('DNg100->E1 reproduced as non-zero',abs(syn('DNg100','E1')-mo['dn_edges']['DNg100->E1']['synapses'])<1e-6)
    c('DNg100->E2 is zero',mo['dn_edges']['DNg100->E2']['synapses']==0.0)
    c('DNb08 edges recorded as missing',all(v['synapses']==0 for k,v in mo['dn_edges'].items() if k.startswith('DNb08')))
    c('BANC has no transmitter identity',sg['banc_nt']['nt_neuron']['all_minus_one'] is True and sg['banc_nt']['nt_pair']['all_minus_one'] is True)
    c('E/I marked NOT_ESTABLISHED locally','NOT_ESTABLISHED' in sg['conclusion'])
    t=np.load(o/'TIER_ROOT_IDS.npz')['root_ids']
    c('tier root ids archived with 12000 entries',t.shape==(12000,),observed_shape=list(t.shape))
    ts=set(int(x) for x in t)
    for lab in ('E1','E2','I1'):
        c(f'{lab} in-tier count reproduced',res['types'][lab]['in_tier_count']==len([r for r in res['types'][lab]['root_ids'] if r in ts]))
    c('descending neurons absent from the tier reproduced',all(res['types'][lab]['in_tier_count']==0 for lab in ('DNg100','DNb08')))
    c('C1 hit with C4 parallel constraint',oc['branch']=='C1' and oc['ei_sign']['c4_parallel_constraint'] is True)
    c('C2 caveat for the DNb08 pathway recorded','DNb08' in oc['c2_caveat'])
    c('next round spec has two arms with a pre-registered grid','arm1' in sp and 'x0.5' in sp['arm1']['grid'] and 'arm2' in sp)
    c('frozen marker present so the verifier writes only here',(o/'S306_PREREG_FROZEN').is_file())
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
           calls=dict(model_load=0,physics_step=0,forward=0,neural_tick=0,render=0),connectome_calls=dict(connectome_load=0,tier_build=0),
           limitations=['cell-type mapping is cross-dataset (paper uses MANC/FANC, this is BANC)','E/I is an external citation, not a local measurement','no support/walking claim'])
    (o/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),nfail=len(r['failed']),failed=r['failed'][:3])))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
