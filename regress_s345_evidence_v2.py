"""Deterministic three-case diagnostic. Separate integrity and recovery exits; never modifies S345."""
from pathlib import Path
import argparse,json,hashlib,datetime
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SOURCE=Path('/home/sensen/Desktop/engine修复验收/s345-twenty-second-engagement/20261027T054112Z')
LIMITS=dict(circular_bias_abs_deg=5.,p95_abs_deg=10.,max_abs_deg=10.,drift_error_abs_deg_s=1.,halfturn_count_error=0)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())+'\n')
def wrap(x):return (x+180)%360-180
def R(x):return float(np.abs(np.mean(np.exp(1j*np.radians(x)))))
def winding(x,fs):
 edge=int(.2*fs);advance=float(np.median(x[-edge:])-np.median(x[:edge]));return dict(net_advance_deg=advance,net_halfturn_count=int(np.floor(np.abs(advance)/180)))
def snapshot():return {str(p.relative_to(SOURCE)):sha(p) for p in SOURCE.rglob('*') if p.is_file()}
def verify(o):
 pre=json.loads((o/'PREREG.json').read_text());results=json.loads((o/'RESULTS.json').read_text());checks={};recovery={}
 for name,r in results['cases'].items():
  z=np.load(o/(name+'.npz'));defined=z['truth_defined_retained'].astype(bool)
  residual=wrap(z['estimated_delta_unwrapped_deg']-z['generator_delta_retained_deg'])
  checks[name+'.full_length']=len(z['full_time_s'])==800 and len(z['retained_time_s'])==480
  checks[name+'.complete_all_sample_residual']=np.allclose(residual,z['reference_residual_all_samples_deg'])
  checks[name+'.undefined_not_scored']=r['accuracy']['n_scored']==int(defined.sum()) and r['accuracy']['n_undefined']==int((~defined).sum())
  checks[name+'.definition_exact_zero_only']=np.array_equal(z['truth_defined_full'],z['generator_amplitude_B']!=0)
  checks[name+'.all_finite']=all(np.isfinite(z[k]).all() for k in z.files)
  reference_advance=-360.0*r['input_formulas']['df_hz']*(np.median(z['retained_time_s'][-8:])-np.median(z['retained_time_s'][:8]))
  reference_count=int(np.floor(np.abs(reference_advance)/180.0))
  if defined.all():
   checks[name+'.independent_truth_advance']=np.isclose(reference_advance,r['winding_recovery']['truth']['net_advance_deg'])
   checks[name+'.independent_truth_count']=reference_count==r['winding_recovery']['truth']['net_halfturn_count']
  checks[name+'.time_R_recompute']=np.isclose(R(z['estimated_delta_unwrapped_deg']),r['signal_time_R'])
  checks[name+'.residual_R_recompute']=np.isclose(R(residual[defined]),r['accuracy']['residual_R'])
  checks[name+'.p95_recompute']=np.isclose(np.percentile(np.abs(residual[defined]),95),r['accuracy']['abs_error_quantiles_deg']['95'])
  checks[name+'.max_recompute']=np.isclose(np.max(np.abs(residual[defined])),r['accuracy']['abs_error_quantiles_deg']['100'])
  accuracy=r['accuracy'];limits=pre['limits']
  recovered=dict(bias=bool(np.abs(accuracy['signed_circular_bias_deg'])<=limits['circular_bias_abs_deg']),
   p95=bool(accuracy['abs_error_quantiles_deg']['95']<=limits['p95_abs_deg']),maximum=bool(accuracy['abs_error_quantiles_deg']['100']<=limits['max_abs_deg']))
  if accuracy['n_undefined']==0:
   recovered['drift']=bool(np.abs(r['drift_recovery']['error_deg_s'])<=limits['drift_error_abs_deg_s'])
   recovered['halfturn']=r['winding_recovery']['count_error']==limits['halfturn_count_error']
  recovery[name]=dict(pass_on_defined_phase_samples=all(recovered.values()),checks=recovered,
   drift_and_winding_status=('UNDEFINED' if accuracy['n_undefined'] else 'TESTED'),
   no_verdict_at_undefined_samples=accuracy['n_undefined'])
 checks['S345_unchanged']=snapshot()==json.loads((o/'INPUT_HASHES.json').read_text())
 checks['thresholds_identical_to_frozen_prereg']=pre['limits']==LIMITS
 evidence=all(checks.values());recovered=all(r['pass_on_defined_phase_samples'] for r in recovery.values())
 dump(o/'EVIDENCE_TEST.json',dict(passed=evidence,exit_code=0 if evidence else 1,checks=checks))
 dump(o/'RECOVERY_TEST.json',dict(passed=recovered,exit_code=0 if recovered else 1,cases=recovery,
  note='Undefined phase points and drift/winding across them are neither correct nor incorrect; full arrays retained. Evidence pass cannot override recovery failure.'))
 print(json.dumps(dict(evidence_pass=evidence,recovery_pass=recovered,recovery=recovery),ensure_ascii=False))
 return 0 if evidence and recovered else 1

def run(o):
 o.mkdir(parents=True,exist_ok=False)
 dump(o/'PREREG.json',dict(utc=datetime.datetime.now(datetime.UTC).isoformat(),limits=LIMITS,fs_hz=40.,input_rows=800,duration_s=20.,retained_window_s=[4,16],
  filter=dict(order=4,band_hz=[.5,4.],form='SOS',operation='sosfiltfilt full signal; trim4s/edge; Hilbert on retained signal'),
  cases=['static_165','detuned_002','single_amplitude_collapse'],
  truth_defined='generator amplitude exactly nonzero; descriptive truth-domain mask only, not a production amplitude gate',
  zero_truth_handling='all sample reference residual retained but undefined sample never scored; global drift and winding accuracy across an undefined point UNTESTED',
  distinction='signal temporal R; truth-relative residual R and signed error quantiles; drift/winding recovery are independent metrics',
  calls=dict(neural_tick=0,decoder_eval=0,physics_step=0,Poisson_draws=0)))
 dump(o/'INPUT_HASHES.json',snapshot())
 dump(o/'PREREG_FROZEN.json',dict(prereg_sha256=sha(o/'PREREG.json'),script_sha256=sha(Path(__file__))))
 fs=40.;t=np.arange(800)/fs;k=160;tt=t[k:-k];sos=signal.butter(4,[.5,4],fs=fs,btype='bandpass',output='sos')
 specs=[('static_165',0.,False),('detuned_002',.02,False),('single_amplitude_collapse',0.,True)];results={};plot={}
 for name,df,collapse in specs:
  pa=2*np.pi*2*t;pb=2*np.pi*(2+df)*t-np.radians(165.)
  amp=1-np.exp(-.5*((t-10.)/.15)**2) if collapse else np.ones_like(t)
  defined=amp!=0;retdefined=defined[k:-k]
  a=20+10*np.cos(pa);b=20+10*amp*np.cos(pb);truth=np.degrees(pa-pb)[k:-k]
  a1=signal.sosfiltfilt(sos,a-a.mean())[k:-k];b1=signal.sosfiltfilt(sos,b-b.mean())[k:-k]
  ha=signal.hilbert(a1);hb=signal.hilbert(b1);ua=np.unwrap(np.angle(ha));ub=np.unwrap(np.angle(hb))
  estimated=np.degrees(np.unwrap(ua-ub));res=wrap(estimated-truth);raw=np.degrees(np.angle(ha)-np.angle(hb))
  valid=res[retdefined]
  acc=dict(n_scored=int(retdefined.sum()),n_undefined=int((~retdefined).sum()),residual_R=R(valid),
   signed_circular_bias_deg=float(np.degrees(np.angle(np.mean(np.exp(1j*np.radians(valid)))))),
   signed_error_quantiles_deg={str(q):float(np.percentile(valid,q)) for q in (0,1,5,50,95,99,100)},
   abs_error_quantiles_deg={str(q):float(np.percentile(np.abs(valid),q)) for q in (0,50,90,95,99,100)})
  ests=float(np.polyfit(tt-tt[0],estimated,1)[0]);ts=-360.0*df;ew=winding(estimated,fs)
  # Truth is computed analytically from the generator, never via the tested unwrap/count routine.
  truth_advance=-360.0*df*(float(np.median(tt[-8:]))-float(np.median(tt[:8])))
  tw=dict(net_advance_deg=truth_advance,net_halfturn_count=int(np.floor(np.abs(truth_advance)/180.0)),source='generator analytic df times elapsed median-endpoint time; no unwrap')
  recovered_drift=dict(status='TESTED',truth_deg_s=ts,estimate_deg_s=ests,error_deg_s=ests-ts,truth_source='generator analytic -360*df')
  recovered_winding=dict(status='TESTED',estimate=ew,truth=tw,count_error=ew['net_halfturn_count']-tw['net_halfturn_count'])
  if not retdefined.all():
   recovered_drift=dict(status='UNDEFINED',reason='global latent-carrier slope saved as reference only')
   recovered_winding=dict(status='UNDEFINED',reason='no accuracy verdict for continuity across exactly zero amplitude')
  results[name]=dict(input_formulas=dict(A='20+10*cos(2*pi*2*t)',B='20+10*amplitude(t)*cos(2*pi*(2+df)*t-radians(165))',
    amplitude=('1-exp(-0.5*((t-10)/0.15)^2)' if collapse else '1'),df_hz=df),
   metadata=dict(fs_hz=fs,full_window_s=[0,20],last_input_sample_s=float(t[-1]),retained_window_s=[4,16],last_retained_sample_s=float(tt[-1]),filter_order=4,filter_band_hz=[.5,4],construction='full SOS -> trim -> Hilbert'),
   signal_time_R=R(estimated),truth_reference_time_R=R(truth[retdefined]),accuracy=acc,drift_recovery=recovered_drift,winding_recovery=recovered_winding,
   descriptive_only=dict(estimated_ols_slope_deg_s=ests,latent_reference_slope_deg_s=ts,estimated_winding=ew,latent_reference_winding=tw,
    generator_amplitude_min=float(amp.min()),analytic_amplitude_B_min=float(np.abs(hb).min())),
   truth_caveat='known sinusoidal carrier; at exact generator zero phase is undefined; modulation/filtered analytic phase can differ from latent phase',masked_observations=0)
  data=dict(full_time_s=t,retained_time_s=tt,input_A=a,input_B=b,generator_amplitude_B=amp,truth_defined_full=defined,truth_defined_retained=retdefined,
   generator_phaseA_rad=pa,generator_phaseB_rad=pb,generator_delta_retained_deg=truth,estimated_phaseA_rad=ua,estimated_phaseB_rad=ub,
   estimated_delta_unwrapped_deg=estimated,estimated_delta_wrapped_deg=wrap(estimated),
   reference_residual_all_samples_deg=res,residual_score_defined_mask=retdefined,
   analytic_amplitude_A=np.abs(ha),analytic_amplitude_B=np.abs(hb),filtered_A=a1,filtered_B=b1,
   preunwrap_increment_deg=np.r_[0,np.diff(raw)],postunwrap_increment_deg=np.r_[0,np.diff(estimated)],
   cumulative_net_advance_deg=estimated-np.median(estimated[:8]),filter_sos=sos)
  np.savez_compressed(o/(name+'.npz'),**data);plot[name]=data
 dump(o/'RESULTS.json',dict(cases=results,thresholds_source='PREREG.json',not_original_certificate=True))
 fig,axs=plt.subplots(3,3,figsize=(18,11))
 for i,(name,df,collapse) in enumerate(specs):
  z=plot[name];r=results[name];a=r['accuracy']
  axs[i,0].plot(tt,z['analytic_amplitude_A'],label='analytic A');axs[i,0].plot(tt,z['analytic_amplitude_B'],label='analytic B');axs[i,0].set_title(name);axs[i,0].legend(fontsize=7)
  axs[i,1].plot(tt,z['reference_residual_all_samples_deg']);axs[i,1].axhline(10,c='r',ls='--');axs[i,1].axhline(-10,c='r',ls='--')
  if collapse:
   bad=~z['truth_defined_retained'];axs[i,1].scatter(tt[bad],z['reference_residual_all_samples_deg'][bad],c='r',marker='x',label='undefined: not scored');axs[i,1].legend(fontsize=7)
  axs[i,1].set_title('time R %.6f; residual R %.6f'%(r['signal_time_R'],a['residual_R']))
  axs[i,2].plot(tt,z['cumulative_net_advance_deg'],label='estimated median baseline')
  truth=z['generator_delta_retained_deg'];axs[i,2].plot(tt,truth-np.median(truth[:8]),'--',label='carrier reference median baseline')
  axs[i,2].set_title('drift/winding '+r['winding_recovery']['status']);axs[i,2].legend(fontsize=7)
  for ax in axs[i]:ax.grid(alpha=.2);ax.set_xlabel('original time (s)')
 fig.suptitle('Separate integrity and recovery verdicts; all samples retained; zero-amplitude truth undefined');fig.tight_layout();fig.savefig(o/'RESIDUAL_AND_WINDING.png',dpi=140);plt.close(fig)
 (o/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
 (Path('/home/sensen/Desktop')/'S345_CERT_EVIDENCE_V2_RESIDUAL_AND_WINDING.png').write_bytes((o/'RESIDUAL_AND_WINDING.png').read_bytes())
 rc=verify(o)
 recovery=json.loads((o/'RECOVERY_TEST.json').read_text());evidence=json.loads((o/'EVIDENCE_TEST.json').read_text())
 lines=['# 确定性证据回归v2：完整性与回收分离','',
  '执行前已写入PREREG和脚本/配置哈希。阈值固定：|圆偏差|≤5°、p95≤10°、最大绝对误差≤10°、漂移误差≤1°/s、净半圈计数一致。没有运行后调门。',
  '证据完整性exit=%d；任务回收exit=%d；整体程序exit=%d。'%(evidence['exit_code'],recovery['exit_code'],rc),
  '', '|案例|时间R|残差R（仅真值有定义点）|signed误差p5/p50/p95°|abs误差p95/max°|漂移回收|绕转回收|','|---|---:|---:|---|---|---|---|']
 for name,r in results.items():
  a=r['accuracy'];q=a['signed_error_quantiles_deg'];e=a['abs_error_quantiles_deg']
  lines.append('|%s|%.6f|%.6f|%.4f / %.4f / %.4f|%.4f / %.4f|%s|%s|'%(name,r['signal_time_R'],a['residual_R'],q['5'],q['50'],q['95'],e['95'],e['100'],json.dumps(r['drift_recovery'],ensure_ascii=False),json.dumps(r['winding_recovery'],ensure_ascii=False)))
 lines+=['','精确零幅值出现在t=10s：生成器潜在相位仍存，但truth_defined=false。此点的reference残差、幅值、相位跳变、绕转全保存，只是不判对错；跨零点的全局漂移/绕转回收UNDEFINED，不给它盖PASS或FAIL。未在生产数组引入幅值剔除门。',
  '频差例即使均值偏差很小、p95约1°，最大端点误差仍约76°，按执行前最大误差门必须失败。文件生成成功不会将回收失败改绿。',
  '', '## 最小生产改动位置（仅建议）',
  '冻结analyze_s345.py第199–203行synth/train_rate：旁存原时间轴、潜在载波相位、幅值/占空、df和种子；精确零幅值加描述性truth_defined。',
  '第68–93行phase_core：提供可选trace返回（不改变当前顺序），保存解析幅值、wrapped/unwrapped相位与增量。',
  '第214–226行认证循环：每seed先写npz，再分别聚合时间R、truth-relative残差R/带符号误差分位、漂移及绕转回收；审计器检查覆盖与可复算，同时分开完整性和回收的exit。',
  '', '本三案例诊断不是原36格认证，不修复其结论；S345封存脚本不改、NOT_JUDGED/严格exit1/28格独立阻断/禁止接物理全部保持。']
 (o/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n')
 dump(o/'MANIFEST_SHA256.json',{p.name:sha(p) for p in o.iterdir() if p.is_file() and p.name!='MANIFEST_SHA256.json'})
 return rc
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--verify',action='store_true');args=ap.parse_args()
 raise SystemExit(verify(args.output) if args.verify else run(args.output))
