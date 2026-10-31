"""Independent saved-NPZ verification; no sample generation and no neural/model imports."""
from pathlib import Path
import json,argparse,hashlib
import numpy as np
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())+'\n')
def main(p):
 pre=json.loads((p/'PREREG.json').read_text());r=json.loads((p/'CERTIFICATION_RESULTS.json').read_text());checks={};failed=[]
 frozen=json.loads((p/'FROZEN_SHA256.json').read_text())['files'];checks['all_frozen_identity_hashes']=all(sha(p/k)==v for k,v in frozen.items())
 counts={};specs=pre['cells']
 for spec in specs:
  key=spec['key'];z=np.load(p/'arrays'/f'{key}.npz',allow_pickle=False);ok={}
  ok['input_count_shape']=z['input_count_A_1ms'].shape==z['input_count_B_1ms'].shape==(100,20000)
  ok['seeds_exact']=np.array_equal(z['seeds'],spec['seeds'])
  for impl,fs in (('main',40.),('audit',50.)):
   est=z[impl+'__phase_unwrapped_deg'];tt=z[impl+'__time_original_s'];truth=z[impl+'__latent_reference_phase_deg'];res=z[impl+'__circular_latent_reference_residual_deg']
   n=int(12*fs);ok[impl+'.shape']=est.shape==truth.shape==res.shape==(100,n) and len(tt)==n
   ok[impl+'.integer_window']=np.array_equal(np.rint(tt*fs).astype(int),np.arange(int(4*fs),int(16*fs)))
   expected=spec['phase']-360*spec['df']*tt
   ok[impl+'.latent_formula']=np.allclose(truth,expected[None,:])
   recompute=(est-truth+180)%360-180;ok[impl+'.point_residual']=np.allclose(recompute,res)
   q=np.mean(np.exp(1j*np.radians(est)),axis=1);R=np.abs(q);means=np.degrees(np.angle(q));target=(spec['phase']-360*spec['df']*10+180)%360-180
   bias=(means-target+180)%360-180;cell=r['implementations'][impl][key]
   ok[impl+'.original_bias_recompute']=np.isclose(float(bias.mean()),cell['bias_deg'],atol=1e-9)
   ok[impl+'.original_R_recompute']=np.isclose(float(R.mean()),cell['R'],atol=1e-12)
   passtest=np.abs(float(bias.mean()))<=5 and float(R.mean())>=.9
   ok[impl+'.pass_predicates_unchanged']=bool(passtest)==cell['passes'] and bool(np.abs(float(bias.mean()))<=5)==cell['bias_predicate_pass'] and bool(float(R.mean())>=.9)==cell['R_predicate_pass']
   amps=z[impl+'__ampA'];ok[impl+'.amp_trace_shape']=amps.shape==est.shape and z[impl+'__ampB'].shape==est.shape
   inc=np.column_stack([np.zeros(100),np.diff(est,axis=1)]);ok[impl+'.increment_from_same_trace']=np.allclose(inc,z[impl+'__unwrap_increment_deg'])
   edge=int(.2*fs);advance=np.median(est[:,-edge:],axis=1)-np.median(est[:,:edge],axis=1);ct=np.floor(np.abs(advance)/180).astype(int)
   ok[impl+'.count_from_same_trace']=np.array_equal(ct,z[impl+'__halfturn_count'])
   tc=tt-tt.mean();slope=est@tc/float(tc@tc);ok[impl+'.slope_from_same_trace']=np.allclose(slope,z[impl+'__drift_deg_s'],atol=1e-9)
   ok[impl+'.per_seed_coverage']=len(cell['samples'])==100
  checks[key]=all(ok.values());counts[key]=ok
  if not checks[key]:failed.append(dict(cell=key,failed=[k for k,v in ok.items() if not v]))
 source=json.loads((p/'S345_INPUT_HASHES.json').read_text());root=Path('/home/sensen/Desktop/engine修复验收/s345-twenty-second-engagement/20261027T054112Z')
 checks['original_S345_unchanged']=all((root/k).is_file() and sha(root/k)==v for k,v in source.items())
 checks['full_grid36']=len(specs)==len(r['implementations']['main'])==len(r['implementations']['audit'])==36
 summary={impl:dict(passed=sum(c['passes'] for c in tab.values()),failed=sum(not c['passes'] for c in tab.values()),exit_code=0 if all(c['passes'] for c in tab.values()) else 1) for impl,tab in r['implementations'].items()}
 result=dict(passed=all(checks.values()),exit_code=0 if all(checks.values()) else 1,checks=checks,cell_checks=counts,failures=failed,certification=summary,verification='saved array formulas only; no Poisson regeneration')
 dump(p/'INDEPENDENT_EVIDENCE_VERIFICATION.json',result);print(json.dumps(dict(evidence_exit=result['exit_code'],failures=failed,certification=summary)))
 return result['exit_code']
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);raise SystemExit(main(ap.parse_args().output))
