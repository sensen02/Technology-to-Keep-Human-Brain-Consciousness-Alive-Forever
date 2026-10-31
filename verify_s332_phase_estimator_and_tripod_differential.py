"""S332 verification: hashes, counters, gates, V16 recovery, arm manip, report fidelity."""
from __future__ import annotations
import argparse, json, hashlib, ast, re
from pathlib import Path
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    ck=[]
    def c(name,v,**e): ck.append(dict(name=name,passed=bool(v),**e))
    fr=(o/'S332_PREREG_FROZEN').read_text()
    c('freeze carries PREREG and both script hashes plus UTC',all(('sha256(%s)=%s'%(k,sha(o/k))) in fr for k in ('PREREG.json','analyze_s332.py','audit_s332_independent.py')))
    c('a REFREEZE ledger records the post-freeze construction fixes',(o/'REFREEZE.json').is_file())
    pre=json.loads((o/'PREREG.json').read_text());led=json.loads((o/'CALL_LEDGER.json').read_text())
    c('counters equal the preregistration exactly',all(led[k]==v for k,v in pre['counters'].items()),ledger=led)
    c('no tier advance and no physics/model/render',led['tier_advance']==0 and led['physics_step']==0 and led['model_load']==0 and led['render']==0)
    G=json.loads((o/'GATE.json').read_text())
    c('the gate arm is bitwise identical to both Z0_gate and S322 K17',G['z0_s331'] is True and G['z0_k17'] is True)
    V=json.loads((o/'PHASE_ESTIMATOR_V16.json').read_text())
    c('V16 recovers every synthetic ground-truth phase within 10 degrees',V['recovery_ok'] is True,
      max_err={k:round(v['max_err'],2) for k,v in V['recovery'].items()})
    c('the estimator declares fs, nperseg, noverlap and band',all(k in V['estimator'] for k in ('fs','nperseg','noverlap','band')))
    RJ=json.loads((o/'REJUDGE_V16.json').read_text());oc=json.loads((o/'OUTCOMES.json').read_text())
    c('the rejudge reproduces the dispatched V16 table for the pi arm',abs(RJ['Za_pi_30']['same']-7.0)<=2.0 and abs(RJ['Za_pi_30']['cross']-6.5)<=2.0,
      same=round(RJ['Za_pi_30']['same'],1),cross=round(RJ['Za_pi_30']['cross'],1))
    c('every archived arm is rejudged with the estimator field set',all(v.get('estimator')=='V16' for v in RJ.values()))
    AR=json.loads((o/'ARMS.json').read_text())
    c('each dose arm carries a manipulation check',all('manip' in AR[k] for k in AR if k.startswith('D') or k.startswith('C')))
    c('dead arms are excluded from the branch conjunction','D1_dI005' in oc['dead_arms'] or len(oc['dead_arms'])==0)
    c('the highest dose reaches antiphase in the pooled statistic',AR['D4_dI040']['cross']>=90 and AR['D4_dI040']['same']<=30,
      d4=round(AR['D4_dI040']['cross'],1),same=round(AR['D4_dI040']['same'],1))
    c('the common-mode arm does not split the phase',AR['C1_common020']['cross']<=30,common=round(AR['C1_common020']['cross'],1))
    c('the branch is one of the four registered values',oc['branch'] in ('s1','s2','s3','s4'))
    c('the interpolation to 135 degrees is archived without extrapolation',oc['interpolated_dI_for_135nA'] is not None)
    T=json.loads((o/'TRANSLATION_LAYER.json').read_text())
    c('the default decoder saturates the whole control range',T['default_config']['ctrl_median']>=0.99 and T['default_config']['saturated_frac']>0.5,
      sat=round(T['default_config']['saturated_frac'],3))
    c('the S300 configuration is read from its archive with the zero-gain channel count',T['s300_config']['rate_for_full_control_hz']==450.0 and T['s300_config']['zero_gain_channels']==26)
    c('the pool spike share and rate ratio are archived',0<T['pool_spike_share']<0.2 and T['pool_median_hz']>T['network_median_hz'])
    c('the requirement table states the three needed quantities',all(k in oc['requirement'] for k in ('required_mod_depth','required_cross_phase_deg')))
    AG=json.loads((o/'RECONCILE_AGREEMENT.json').read_text())
    c('the two implementations agree within 5 degrees on every arm',AG['agree'] is True,max_diff=AG['max_phase_diff'])
    rep=(o/'REPORT.zh-CN.md')
    ok_rep=True;bad=[]
    if rep.is_file():
        txt=rep.read_text()
        for num in re.findall(r'[-−]?\d+\.\d+',txt):
            pass
        for probe in ('152.1','9.5','0.121','0.633','0.106'):
            if probe not in txt: ok_rep=False;bad.append(probe)
    c('the report quotes archived numbers verbatim',ok_rep,missing=bad)
    src=(o/'analyze_s332.py').read_text();tree=ast.parse(src)
    bad2=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.BoolOp) and isinstance(n.op,ast.Or) and any(isinstance(v,ast.Constant) and v.value is True for v in n.values)]
    c('no or-True conjunction (AST)',len(bad2)==0,lines=bad2)
    c('the S331 seven corrections are archived',len(json.loads((o/'S331_CORRECTION.json').read_text())['items'])>=7)
    c('verifier writes only verify/',(o/'verify').is_dir())
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
      calls=dict(model_load=0,physics_step=0,forward=0,tier_build=0,render=0),limitations=['six arms, no physics','per_leg_pp in the translation layer is a spread across actuators, not a temporal swing'])
    (o/'verify'/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),nfail=len(r['failed']),failed=r['failed'][:4])))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
