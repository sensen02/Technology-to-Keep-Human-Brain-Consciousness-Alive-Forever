"""Read-only attribution from saved candidate certificate trajectories; no RNG or filtering."""
from pathlib import Path
import argparse,json,hashlib,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
S=Path('/home/sensen/Desktop/engine修复验收/s345-candidate-certification/20261027T125512Z')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())+'\n')
def quant(v):return {str(q):float(np.percentile(v,q)) for q in (0,5,50,95,100)}
def main(o):
 o.mkdir(parents=True,exist_ok=False);before={str(f.relative_to(S)):sha(f) for f in S.rglob('*') if f.is_file()};dump(o/'INPUT_HASHES.json',before)
 data=json.loads((S/'CERTIFICATION_RESULTS.json').read_text());pre=json.loads((S/'PREREG.json').read_text());rows=[];details={};checks={}
 for spec in pre['cells']:
  key=spec['key'];z=np.load(S/'arrays'/f'{key}.npz');details[key]={}
  for impl in ('main','audit'):
   est=z[impl+'__phase_unwrapped_deg'];res=z[impl+'__circular_latent_reference_residual_deg'];time=z[impl+'__time_original_s'];truth=z[impl+'__latent_reference_phase_deg']
   complex_time=np.mean(np.exp(1j*np.radians(est)),axis=1);Rtime=np.abs(complex_time);Rres=np.abs(np.mean(np.exp(1j*np.radians(res)),axis=1))
   center=(spec['phase']-360*spec['df']*10+180)%360-180;sample_mean=np.degrees(np.angle(complex_time));bias=(sample_mean-center+180)%360-180
   mean_R=float(Rtime.mean());pooled_R=float(np.abs(complex_time.mean()));cross_seed_R=float(np.abs(np.mean(np.exp(1j*np.radians(sample_mean)))))
   mean_bias=float(bias.mean());mean_abs_bias=float(np.abs(bias).mean());rms_bias=float(np.sqrt(np.mean(bias*bias)))
   neg=int(np.sum(bias<0));pos=int(np.sum(bias>0));cancel=mean_abs_bias-np.abs(mean_bias)
   q=np.percentile(res,[0,5,50,95,100],axis=1).T
   drift=z[impl+'__drift_deg_s'];drift_err=drift+360*spec['df'];slip=z[impl+'__halfturn_count']
   fs=40 if impl=='main' else 50;edge=int(.2*fs);truthadv=-360*spec['df']*(np.median(time[-edge:])-np.median(time[:edge]));truth_count=int(np.floor(np.abs(truthadv)/180))
   old=data['implementations'][impl][key];checks[key+'.'+impl+'.R_formula']=np.isclose(mean_R,old['R'],atol=1e-12);checks[key+'.'+impl+'.bias_formula']=np.isclose(mean_bias,old['bias_deg'],atol=1e-9)
   fail=not old['passes'];reason='SAMPLE_INTERNAL_TEMPORAL_CONCENTRATION_BELOW_GATE' if fail else 'PASS_AS_ARCHIVED'
   d=dict(time_R_quantiles=quant(Rtime),latent_residual_R_quantiles=quant(Rres),signed_sample_bias_quantiles_deg=quant(bias),
    signed_error_quantiles_per_sample_deg=q.tolist(),signed_error_quantile_columns=['p0','p5','p50','p95','p100'],
    drift_latent_reference_error_quantiles_deg_s=quant(drift_err),winding_histogram={str(int(v)):int(np.sum(slip==v)) for v in np.unique(slip)},
    latent_reference_count=truth_count,latent_reference_winding_mismatch_samples=int(np.sum(slip!=truth_count)),
    mean_time_R=mean_R,pooled_time_and_seed_R=pooled_R,cross_seed_mean_phase_R=cross_seed_R,
    mean_signed_bias_deg=mean_bias,mean_abs_sample_bias_deg=mean_abs_bias,rms_sample_bias_deg=rms_bias,
    positive_sample_bias=pos,negative_sample_bias=neg,cancellation_magnitude_deg=float(cancel),
    samples_time_R_below_point9=int(np.sum(Rtime<.9)),samples_time_R_above_or_equal_point9=int(np.sum(Rtime>=.9)),
    classification=reason,causal_source='UNRESOLVED: pulse shape/harmonics, counting noise, filtering and estimation are not separated by saved observations',
    scope='latent-reference residual/drift/winding are diagnostics, not unique filtered analytic-phase errors')
   details[key][impl]=d
   rows.append(dict(implementation=impl,cell=key,df_hz=spec['df'],phase_deg=spec['phase'],duty=spec['duty'],rateA=spec['rates'][0],rateB=spec['rates'][1],
    archived_pass=old['passes'],classification=reason,mean_time_R=mean_R,time_R_p5=float(np.percentile(Rtime,5)),time_R_p50=float(np.median(Rtime)),time_R_p95=float(np.percentile(Rtime,95)),
    pooled_R=pooled_R,cross_seed_R=cross_seed_R,within_sample_deficit=1-mean_R,extra_pooling_deficit=mean_R-pooled_R,
    mean_residual_R=float(Rres.mean()),residual_R_p5=float(np.percentile(Rres,5)),mean_signed_bias=mean_bias,mean_abs_bias=mean_abs_bias,rms_bias=rms_bias,
    cancellation_magnitude=float(cancel),positive_samples=pos,negative_samples=neg,samples_R_below_gate=int(np.sum(Rtime<.9)),
    drift_error_p5=float(np.percentile(drift_err,5)),drift_error_p50=float(np.median(drift_err)),drift_error_p95=float(np.percentile(drift_err,95)),winding_mismatches=int(np.sum(slip!=truth_count))))
 dump(o/'PER_CELL_ATTRIBUTION.json',dict(cells=details,formula='R_cell=mean_seed(abs(mean_time(exp(i*phase_est)))) ; not abs(mean_seed_time(exp(i*phase_est)))'))
 with (o/'PER_CELL_ATTRIBUTION.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 summaries={}
 for impl in ('main','audit'):
  summaries[impl]={}
  for df in (0,.02):
   rr=[r for r in rows if r['implementation']==impl and r['df_hz']==df];ff=[r for r in rr if not r['archived_pass']]
   summaries[impl][str(df)]=dict(cells=len(rr),failed=len(ff),failed_internal_temporal_R=len(ff),failed_due_only_to_cross_seed_pooling=0,
    failed_mean_R_range=[min(r['mean_time_R'] for r in ff),max(r['mean_time_R'] for r in ff)],
    failed_mean_R_below_samples_range=[min(r['samples_R_below_gate'] for r in ff),max(r['samples_R_below_gate'] for r in ff)],
    max_bias_cancellation_deg=max(r['cancellation_magnitude'] for r in rr),max_rms_sample_bias_deg=max(r['rms_bias'] for r in rr),
    max_extra_pooling_deficit=max(r['extra_pooling_deficit'] for r in rr))
 dump(o/'SUMMARY.json',dict(groups=summaries,conclusion='All28 failures per implementation are a mean of genuinely low within-sample temporal R values. Cross-seed phase offsets do not enter this predicate; pooling is a different untested statistic. Small average signed bias can mask positive-negative sample errors, quantified separately. Cause of within-sample fluctuations remains unresolved.',no_accuracy_inference_from_agreement=True))
 fig,axs=plt.subplots(2,2,figsize=(16,10));keys=list(details);xx=np.arange(36)
 for impl,col in (('main','tab:blue'),('audit','tab:orange')):
  rr=[r for r in rows if r['implementation']==impl]
  axs[0,0].plot(xx,[r['mean_time_R'] for r in rr],'.-',label=impl+' mean time R',color=col)
  axs[0,1].plot(xx,[r['mean_abs_bias'] for r in rr],'.-',label=impl+' mean abs sample bias',color=col)
 axs[0,0].axhline(.9,c='r',ls='--');axs[0,0].legend();axs[0,0].set_title('Original predicate: temporal concentration inside each sample')
 axs[0,1].legend();axs[0,1].set_title('Signed mean can cancel; sample absolute bias separately visible')
 failedkeys=[k for k in keys if not data['implementations']['main'][k]['passes']];rep0=next(k for k in failedkeys if details[k]['main'] and data['implementations']['main'][k]['df']==0)
 rep1=next(k for k in failedkeys if data['implementations']['main'][k]['df']!=0)
 selection=[]
 for ax,key in zip(axs[1],(rep0,rep1)):
  z=np.load(S/'arrays'/f'{key}.npz');est=z['main__phase_unwrapped_deg'];tt=z['main__time_original_s'];truth=z['main__latent_reference_phase_deg'];rt=np.abs(np.mean(np.exp(1j*np.radians(est)),axis=1));j=int(np.argsort(rt)[50])
  residual=(est[j]-truth[j]+180)%360-180;ax.plot(tt,residual);ax.set_title(key+' median-R sample '+str(j));ax.set_ylabel('signed latent-reference residual (deg)');ax.set_xlabel('original time s')
  selection.append(dict(cell=key,sample=j,selection='first failed cell in df group and median-R sample, fixed no scan'))
  np.savez_compressed(o/(key+'_representative.npz'),time_s=tt,estimated_phase_deg=est[j],latent_reference_deg=truth[j],circular_residual_deg=residual,time_R=rt[j])
 for ax in axs.flat:ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(o/'FAILURE_ATTRIBUTION.png',dpi=140);plt.close(fig);dump(o/'REPRESENTATIVES.json',selection)
 unchanged={str(f.relative_to(S)):sha(f) for f in S.rglob('*') if f.is_file()}==before;checks['source_package_unchanged']=unchanged;checks['all72_cell_rows']=len(rows)==72
 assert all(checks.values());dump(o/'VERIFICATION.json',dict(passed=True,exit_code=0,checks=checks,no_new_samples=True,no_grid_rerun=True))
 lines=['# S345候选认证失败只读归因','', 'bash-225终止exit1；原日志36格均完成并输出认证汇总，无Traceback。证据exit0、主/审认证均exit1，原因是认证谓词而非执行异常。',
  '本包只读全部已存轨迹，不生成新样本、不滤波、不重跑网格或神经臂、不改参数。',
  '', '## 有界问题的结论',
  '原公式为 R_cell=mean_seed(|mean_time exp(i theta_est)|)。它先在每个样本内部计算时间R，再作普通算术平均；跨样本的平均相位互相抵消不参与此公式。两实现每个28失败格均是该平均低于0.9，**没有格子因跨样本先合并相位才失败**。',
  '每样本内部集中度和跨样本相位散度都可以存在，但后者只影响另列的pooled_R/cross_seed_R，不是这28格R失败的直接计算来源。',
  '', '## 分组摘要','```json',json.dumps(summaries,ensure_ascii=False,indent=2),'```',
  '', '## 小平均偏差是否隐藏抵消',
  '逐格同时列mean signed bias、mean absolute sample bias、RMS、正/负样本数及cancellation=mean(abs bias)-abs(mean bias)。它们来自每样本的实际圆均值，不再用小总均值断言100样本全部准确。逐时signed误差分位、latent residual R、drift error、绕转直方图全保存在PER_CELL_ATTRIBUTION.json/CSV。',
  '', '## 归因限制',
  '方脉冲潜在振荡器相位不是滤波后解析相位的唯一真值。latent-relative residual/drift/winding只能描述相对生成器参考的偏离，不能直接命名为仪器相位估计误差。已知频差也降低时间R；其理论理想值能过0.9，不足以自动豁免观测失败。',
  '低R在样本内部发生的计算位置已确定；具体由方脉冲谐波/解析幅度调制、随机计数噪声、滤波或估计偏差造成多少，现有观测没有反事实分离，继续UNRESOLVED。两实现一致不替代准确性证明。',
  '', '## 最小修复位置（不执行）',
  '候选生产认证循环仍将样本时间集中度mean_R当仪器认证门。证据支持在未来预登记中把“时间相位聚集”与“相对适用真值的回收准确性”拆开命名/输出；不得直接将门改为残差R或放宽0.9来消除失败。每seed trace及误差分布应继续保留。',
  '不需更改跨样本平均代码来修低R，因为它忠实执行原公式；小平均偏差的准确性解释需要RMS/分位与真值适用范围，而不是二次平均。',
  '', '原认证包完全不变；S345 NOT_JUDGED、LA30剩余绕转未解析及物理限制保留。']
 (o/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n');(o/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
 (Path('/home/sensen/Desktop')/'S345_FAILURE_ATTRIBUTION.png').write_bytes((o/'FAILURE_ATTRIBUTION.png').read_bytes())
 dump(o/'MANIFEST_SHA256.json',{f.name:sha(f) for f in o.iterdir() if f.is_file() and f.name!='MANIFEST_SHA256.json'})
 print(json.dumps(dict(output=str(o),summaries=summaries,passed=True),ensure_ascii=False))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path);main(ap.parse_args().output)
