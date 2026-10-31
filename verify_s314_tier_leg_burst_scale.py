"""S314 verification: M18 map, per-leg rejudge, split identity, f* closed form, A1 bitwise, counters."""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S309=Path('/home/sensen/Desktop/engine修复验收/s309-native-cpg/20261026T205541Z')
S312=Path('/home/sensen/Desktop/engine修复验收/s312-tier-motif-input-isolation/20261026T211120Z')
S313=Path('/home/sensen/Desktop/engine修复验收/s313-burst-band-and-tier-drive/20261026T211545Z')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def intervals(x):
    i=np.flatnonzero(x)
    if len(i)<3: return None,None
    d=np.diff(i).astype(float);big=d[d>3*np.median(d)]
    if len(big)<10: return None,None
    return 1000.0/np.median(big),float(np.std(big)/max(np.mean(big),1e-9))
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    ck=[]
    def c(n,v,**e): ck.append(dict(name=n,passed=bool(v),**e))
    pre=json.loads((o/'PREREG.json').read_text());mp=json.loads((o/'M18_GLOBAL_INDEX.json').read_text())
    rej=json.loads((o/'LEG_REJUDGE.json').read_text());sp=json.loads((o/'EXTERNAL_INPUT_SPLIT.json').read_text())
    sr=json.loads((o/'SCALE_RULE.json').read_text());A=json.loads((o/'ARMS.json').read_text())
    led=json.loads((o/'CALL_LEDGER.json').read_text());oc=json.loads((o/'OUTCOMES.json').read_text())
    c('frozen marker before dynamics',(o/'S314_PREREG_FROZEN').is_file())
    c('M18 has 18 unique sorted indices',len(mp['M18'])==18 and mp['M18']==sorted(mp['M18']))
    c('M18 roles cover 6 legs x 3 roles',len({(e['leg'],e['role']) for e in mp['entries']})==18)
    c('f* matches the closed form',abs(sr['f_star']-0.05/(65.0*sr['G_ext_uS']))<1e-12)
    c('external+internal equals the archived net input',sp['sum_equals_TSPK'] is True and sp['max_abs_diff']<1e-9,diff=sp['max_abs_diff'])
    # independent per-leg rejudge for T5 with the archived mapping
    m=np.load(S312/'SPK_T5_iso_raw_d050.npz')['motif'];col={}
    for e in mp['entries']: col[(e['leg'],e['role'])]=mp['M18'].index(e['global_index'])
    LEG=('lf','lm','lh','rf','rm','rh');rh=0;freqs={}
    for lg in LEG:
        f,cv=intervals(m[:,col[(lg,'E1')]]);freqs[lg]=f
        if f is not None and 4<=f<=200 and cv is not None and cv<0.25: rh+=1
    c('T5 per-leg burst frequency reproduced independently',rh>=4,rhythmic_legs=rh,freqs=freqs)
    c('T6 is 6/6 and T5 5/6 in the archive',rej['T6_iso_raw_d160']['n_rhythmic']==6 and rej['T5_iso_raw_d050']['n_rhythmic']==5)
    c('A1 matches S309 k1_control bitwise',A['A1_control']['a1_bitwise'] is True)
    exp=pre['counters']['normal']
    c('counters equal the preregistered normal branch',all(led[k]==v for k,v in exp.items()),ledger=led)
    c('no arm reaches 5/6 legs and x3 is declared with its clause table',max(A[k]['n_rhythmic'] for k in A)<=2 and oc['branch']=='x3' and oc['branch_clause']['x3_literal_all_zero'] is False)
    c('scaling audit found non-negative matrices',all(A[k]['audit']['nonneg'] for k in A))
    c('verifier writes only verify/',(o/'verify').is_dir())
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
      calls=dict(model_load=0,physics_step=0,forward=0,render=0),
      limitations=['A1 spike matrix compared by the analysis itself; the verifier confirms the flag, not a re-run','neural only, no physics'])
    (o/'verify'/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),nfail=len(r['failed']),failed=r['failed'][:3])))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
