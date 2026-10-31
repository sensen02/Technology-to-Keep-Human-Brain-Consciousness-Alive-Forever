"""Saved-array three-R identity audit only; no generation/filter/model calls."""
from pathlib import Path
import argparse,json,hashlib,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
S=Path('/home/sensen/Desktop/engine修复验收/s345-candidate-certification/20261027T125512Z')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())+'\n')
def quant(x):return {str(q):float(np.percentile(x,q)) for q in (0,5,25,50,75,95,100)}
def main(o):
 o.mkdir(parents=True,exist_ok=False);before={str(p.relative_to(S)):sha(p) for p in S.rglob('*') if p.is_file()};dump(o/'INPUT_HASHES.json',before)
 pre=json.loads((S/'PREREG.json').read_text());cert=json.loads((S/'CERTIFICATION_RESULTS.json').read_text());rows=[];details={};checks={};representatives=[]
 for spec in pre['cells']:
  key=spec['key'];z=np.load(S/'arrays'/f'{key}.npz');details[key]={}
  for impl in ('main','audit'):
   est=z[impl+'__phase_unwrapped_deg'];assert est.shape[0]==100
   v=np.exp(1j*np.radians(est));m=v.mean(axis=1);rt=np.abs(m);unit=m/rt
   mt=float(rt.mean());u=unit.mean();pool=m.mean();rb=float(np.abs(u));rp=float(np.abs(pool))
   covariance=((rt-mt)*unit).mean();reconstructed=mt*u+covariance
   assert np.abs(pool-reconstructed)<1e-12
   cell=cert['implementations'][impl][key];checks[key+'.'+impl+'.mean_time_R_formula']=bool(np.isclose(mt,cell['R'],atol=1e-12))
   checks[key+'.'+impl+'.complex_vector_identity']=bool(np.abs(pool-reconstructed)<1e-12)
   checks[key+'.'+impl+'.direct_pool_identity']=bool(np.abs(v.mean()-pool)<1e-12)
   checks[key+'.'+impl+'.hundred_samples']=est.shape[0]==cell['n']==100
   bias=np.array([r['bias_deg'] for r in cell['samples']]);res=z[impl+'__circular_latent_reference_residual_deg'];rres=np.abs(np.exp(1j*np.radians(res)).mean(axis=1))
   drift=z[impl+'__drift_deg_s']+360*spec['df'];wind=z[impl+'__halfturn_count'];truthtime=z[impl+'__time_original_s'];fs=40 if impl=='main' else 50;edge=int(.2*fs)
   truthadvance=-360*spec['df']*(np.median(truthtime[-edge:])-np.median(truthtime[:edge]));truthwind=int(np.floor(np.abs(truthadvance)/180))
   fail=not cell['passes'];classification='FAILED_ORIGINAL_MEAN_WITHIN_SAMPLE_R' if fail else 'PASS_ORIGINAL_PREDICATES'
   dd=dict(original_formula='mean_s(abs(mean_t(exp(i*phase_st))))',classification=classification,
    mean_R_time=mt,R_time_quantiles=quant(rt),samples_R_time_below_point9=int(np.sum(rt<.9)),R_between=rb,R_pooled=rp,
    product_of_scalar_Rs=mt*rb,scalar_product_error=rp-mt*rb,
    complex_identity=dict(pooled_vector=[float(pool.real),float(pool.imag)],mean_R_time_times_mean_unit_vector=[float((mt*u).real),float((mt*u).imag)],
     covariance_vector=[float(covariance.real),float(covariance.imag)],reconstructed_vector=[float(reconstructed.real),float(reconstructed.imag)],error=float(np.abs(pool-reconstructed))),
    sample_R_time=rt.tolist(),sample_mean_phase_deg=np.degrees(np.angle(m)).tolist(),
    signed_sample_mean_bias_quantiles_deg=quant(bias),mean_signed_bias_deg=float(bias.mean()),mean_absolute_sample_bias_deg=float(np.abs(bias).mean()),rms_sample_bias_deg=float(np.sqrt(np.mean(bias*bias))),
    residual_R_quantiles_latent_reference=quant(rres),
    signed_error_quantiles_per_sample_deg=np.percentile(res,[0,5,50,95,100],axis=1).T.tolist(),
    drift_error_quantiles_latent_reference_deg_s=quant(drift),winding_histogram={str(int(k)):int(np.sum(wind==k)) for k in np.unique(wind)},latent_reference_winding=truthwind,
    scope='residual/drift/winding reference is latent oscillator, NOT unique filtered analytic-phase truth',
    original_pass=cell['passes'],bias_predicate_pass=cell['bias_predicate_pass'],R_predicate_pass=cell['R_predicate_pass'])
   details[key][impl]=dd
   rows.append(dict(cell=key,implementation=impl,phase=spec['phase'],duty=spec['duty'],rateA=spec['rates'][0],rateB=spec['rates'][1],df=spec['df'],n=100,
    mean_R_time=mt,time_R_p5=float(np.percentile(rt,5)),time_R_p50=float(np.median(rt)),time_R_p95=float(np.percentile(rt,95)),samples_below_point9=int(np.sum(rt<.9)),
    R_between=rb,R_pooled=rp,scalar_R_product=mt*rb,scalar_product_error=rp-mt*rb,complex_identity_error=float(np.abs(pool-reconstructed)),
    mean_signed_bias=float(bias.mean()),mean_abs_sample_bias=float(np.abs(bias).mean()),rms_bias=float(np.sqrt(np.mean(bias*bias))),original_pass=cell['passes'],classification=classification))
  if not cert['implementations']['main'][key]['passes']:
   group=spec['df']
   if not any(x['df']==group for x in representatives):
    est=z['main__phase_unwrapped_deg'];rt=np.abs(np.exp(1j*np.radians(est)).mean(axis=1));j=int(np.argsort(rt)[50]);representatives.append(dict(df=group,cell=key,sample=j))
 dump(o/'PER_CELL_THREE_R.json',dict(cells=details,identity='m_s = R_s*u_s; mean_s(m_s) = mean(R_s)*mean(u_s) + mean((R_s-mean(R_s))*u_s). R_pooled is magnitude of the vector sum, not product of scalar Rs.'))
 with (o/'PER_CELL_THREE_R.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 summary={}
 for impl in ('main','audit'):
  rr=[r for r in rows if r['implementation']==impl];failed=[r for r in rr if not r['original_pass']]
  summary[impl]=dict(cells=len(rr),n_per_cell=100,failed=len(failed),within_sample_predicate_failures=len(failed),between_seed_pooling_is_original_predicate=False,
   original_failed_cells=[r['cell'] for r in failed],max_complex_identity_error=max(r['complex_identity_error'] for r in rr),max_scalar_product_error=float(max(np.abs(r['scalar_product_error']) for r in rr)),
   failed_df0=sum(r['df']==0 for r in failed),failed_df002=sum(r['df']==.02 for r in failed),
   max_R_between_deficit=max(1-r['R_between'] for r in rr),
   exact_mean_time_failure_cells=[r['cell'] for r in failed if r['mean_R_time']<.9])
 dump(o/'SUMMARY.json',dict(implementations=summary,conclusion='28 failures per implementation arise directly from within-sample temporal concentration. Cross-seed dispersion exists but does not enter original mean(R_time) predicate. Both contribute to pooled R through weighted complex identity; they are not independent multiplicative factors.',causal_source='UNRESOLVED: harmonics/noise/filter/estimator contributions cannot be separated by this observational audit'))
 fig,axs=plt.subplots(2,2,figsize=(17,10));rr=[r for r in rows if r['implementation']=='main'];xx=np.arange(36)
 for field,label in (('mean_R_time','mean sample time R'),('R_between','seed mean phase R_between'),('R_pooled','pooled R')):axs[0,0].plot(xx,[r[field] for r in rr],'.-',label=label)
 axs[0,0].axhline(.9,c='r',ls='--');axs[0,0].legend();axs[0,0].set_title('Three distinct quantities, original gate on mean R_time only')
 axs[0,1].plot(xx,[r['scalar_product_error'] for r in rr],'.-');axs[0,1].axhline(0,c='k',ls=':');axs[0,1].set_title('R_pooled - mean(R_time)*R_between (not generally zero)')
 for ax,rep in zip(axs[1],representatives):
  z=np.load(S/'arrays'/f"{rep['cell']}.npz");j=rep['sample'];tt=z['main__time_original_s'];res=z['main__circular_latent_reference_residual_deg'][j]
  ax.plot(tt,res);ax.set_title(rep['cell']+' sample '+str(j));ax.set_xlabel('original time s');ax.set_ylabel('signed latent-reference residual deg')
 for ax in axs.flat:ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(o/'THREE_R_AND_REPRESENTATIVES.png',dpi=140);plt.close(fig);dump(o/'REPRESENTATIVES.json',representatives)
 unchanged={str(p.relative_to(S)):sha(p) for p in S.rglob('*') if p.is_file()}==before;checks['source_unchanged']=unchanged;checks['coverage72_rows']=len(rows)==72
 assert all(checks.values());dump(o/'VERIFICATION.json',dict(passed=True,exit_code=0,checks=checks))
 lines=['# 候选合成认证三种R只读归因','',
  '原采用 mean_seed(abs(mean_time(exp(i theta))))，即每样本时间R再平均；不是先合并所有种子时间点，也不是种子平均相位R。',
  '', '## 精确复数恒等式',
  '令m_s=mean_t exp(iθ_st)=R_s u_s，则 mean_s m_s = mean_s R_s * mean_s u_s + mean_s[(R_s-mean R_s)u_s]。R_pooled是右侧完整复数和的模，不能把三个标量R直接相乘。逐格保存向量各项及验证误差。',
  '', '## 逐实现分类','```json',json.dumps(summary,ensure_ascii=False,indent=2),'```',
  '', '两实现各28失败均为原mean(R_time)<.9；跨种子相位散度不进入该谓词，因此不是跨种子汇总把本来通过的样本抵消成失败。R_pooled可同时包含样本内和跨种子贡献，但它是另一个量，不能用来替换原门。',
  'mean signed bias、mean absolute bias、RMS、每样本分位分列，小平均偏差不能替代100样本误差分布。相对潜在真值的residual R、漂移和绕转另列；方脉冲潜在相位不是滤后解析相位唯一真值，不能自动分类为估计器误差。',
  '', '## 最小改动建议与限制',
  '不需“修”跨种子聚合来消除28失败，聚合忠实实现原mean time R。未来认证设计需明确区分时间聚集与适用真值回收，并保留逐样本分布和复数向量恒等式检查；不能直接改门/调参把它绿化。低R计算位置已定位，谐波/噪声/滤波/估计偏差的具体因果贡献仍未解析。',
  '本次无新样本/无滤波/无网格重跑，原包保持不变；S345 NOT_JUDGED、LA30绕转未解析和禁止物理保持。']
 (o/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n');(o/Path(__file__).name).write_bytes(Path(__file__).read_bytes());(Path('/home/sensen/Desktop')/'S345_THREE_R_ATTRIBUTION.png').write_bytes((o/'THREE_R_AND_REPRESENTATIVES.png').read_bytes())
 dump(o/'MANIFEST_SHA256.json',{p.name:sha(p) for p in o.iterdir() if p.is_file() and p.name!='MANIFEST_SHA256.json'})
 print(json.dumps(dict(output=str(o),validation_exit=0,summary=summary),ensure_ascii=False))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path);main(ap.parse_args().output)
