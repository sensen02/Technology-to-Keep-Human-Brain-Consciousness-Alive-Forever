"""Three deterministic evidence regressions only; explicitly reproduces S345 filter/trim/Hilbert order."""
from pathlib import Path
import argparse,json,hashlib,datetime
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SOURCE=Path('/home/sensen/Desktop/engine修复验收/s345-twenty-second-engagement/20261027T054112Z')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def wrap(x):return (x+180)%360-180
def resultant(x):return float(np.abs(np.mean(np.exp(1j*np.radians(x)))))
def winding(x,fs):
 edge=int(.2*fs);advance=float(np.median(x[-edge:])-np.median(x[:edge]));return dict(net_advance_deg=advance,net_halfturn_count=int(np.floor(np.abs(advance)/180)))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output;o.mkdir(parents=True,exist_ok=False)
 hashes={str(p.relative_to(SOURCE)):sha(p) for p in SOURCE.rglob('*') if p.is_file()};dump(o/'S345_INPUT_HASHES.json',hashes)
 fs=40.;t=np.arange(800)/fs;k=160;tt=t[k:-k];sos=signal.butter(4,[.5,4],fs=fs,btype='bandpass',output='sos')
 cases=[dict(name='static_165',phase=165.,df=0.,collapse=False),dict(name='detuned_002',phase=165.,df=.02,collapse=False),dict(name='single_amplitude_collapse',phase=165.,df=0.,collapse=True)]
 arrays={};results={};checks={}
 for spec in cases:
  name=spec['name'];phaseA=2*np.pi*2*t;phaseB=2*np.pi*(2+spec['df'])*t-np.radians(spec['phase'])
  amp=np.ones_like(t)
  if spec['collapse']:
   # Exactly one zero-amplitude point and a smooth narrow deterministic trough; never masked.
   amp=1-np.exp(-.5*((t-10.)/.15)**2)
  a=20+10*np.cos(phaseA);b=20+10*amp*np.cos(phaseB)
  truth=np.degrees(phaseA-phaseB)[k:-k]
  ya=signal.sosfiltfilt(sos,a-a.mean())[k:-k];yb=signal.sosfiltfilt(sos,b-b.mean())[k:-k]
  ha=signal.hilbert(ya);hb=signal.hilbert(yb)
  ua=np.unwrap(np.angle(ha));ub=np.unwrap(np.angle(hb));estimated=np.degrees(np.unwrap(ua-ub))
  residual=wrap(estimated-truth);raw=np.degrees(np.angle(ha)-np.angle(hb))
  tw=np.polyfit(tt-tt[0],truth,1)[0];ew=np.polyfit(tt-tt[0],estimated,1)[0]
  wc=winding(estimated,fs);wt=winding(truth,fs)
  r=dict(signal_time_R=resultant(estimated),truth_latent_time_R=resultant(truth),residual_R=resultant(residual),
   residual_signed_circular_bias_deg=float(np.degrees(np.angle(np.mean(np.exp(1j*np.radians(residual)))))),
   abs_residual_quantiles_deg={str(q):float(np.percentile(np.abs(residual),q)) for q in (0,50,90,95,99,100)},
   truth_drift_deg_s=float(tw),estimated_drift_deg_s=float(ew),drift_error_deg_s=float(ew-tw),
   estimated_winding=wc,truth_winding=wt,halfturn_count_error=wc['net_halfturn_count']-wt['net_halfturn_count'],
   net_advance_error_deg=wc['net_advance_deg']-wt['net_advance_deg'],
   generator_envelope_min=float(amp.min()),estimated_ampA_min=float(np.abs(ha).min()),estimated_ampB_min=float(np.abs(hb).min()),
   truth_scope=('latent carrier phase only near amplitude zero; not a claim that filtered analytic phase equals latent phase' if spec['collapse'] else 'explicit sinusoidal carrier phase, known deterministically'),
   masked_samples=0,n=len(tt),fs_hz=fs,retained_window_s=[4.,16.],construction='mean remove full20s -> SOS4 band0.5-4 -> trim4s per edge -> Hilbert on retained12s; same as frozen S345')
  r['recovery_diagnostics']=dict(circular_bias_le5=bool(np.abs(r['residual_signed_circular_bias_deg'])<=5),p95_error_le10=bool(r['abs_residual_quantiles_deg']['95']<=10),drift_error_le1=bool(np.abs(ew-tw)<=1),halfturn_count_exact=wc['net_halfturn_count']==wt['net_halfturn_count'])
  results[name]=r
  payload=dict(full_time_s=t,retained_time_s=tt,input_A=a,input_B=b,generator_amplitude_B=amp,
   generator_phase_A_rad=phaseA,generator_phase_B_rad=phaseB,truth_delta_unwrapped_deg=truth,truth_delta_wrapped_deg=wrap(truth),
   filtered_A=ya,filtered_B=yb,analytic_ampA=np.abs(ha),analytic_ampB=np.abs(hb),
   estimated_phaseA_unwrapped_deg=np.degrees(ua),estimated_phaseB_unwrapped_deg=np.degrees(ub),estimated_delta_unwrapped_deg=estimated,
   estimated_delta_wrapped_deg=wrap(estimated),circular_residual_deg=residual,
   preunwrap_increment_deg=np.r_[0,np.diff(raw)],postunwrap_increment_deg=np.r_[0,np.diff(estimated)],
   truth_increment_deg=np.r_[0,np.diff(truth)],cumulative_net_advance_deg=estimated-np.median(estimated[:int(.2*fs)]),filter_sos=sos)
  for key,val in payload.items():arrays[name+'__'+key]=val
  np.savez_compressed(o/(name+'.npz'),**payload)
  checks[name+'.all_fields_finite']=all(np.isfinite(v).all() for v in payload.values())
  checks[name+'.full_and_trim_lengths']=len(t)==800 and len(tt)==480
  checks[name+'.residual_recomputes']=np.allclose(residual,wrap(estimated-truth))
  checks[name+'.no_mask']=len(residual)==480 and r['masked_samples']==0
  checks[name+'.truth_drift_matches_formula']=bool(np.abs(tw+360*spec['df'])<1e-9)
  checks[name+'.metrics_saved_recomputable']=bool(np.abs(resultant(payload['estimated_delta_unwrapped_deg'])-r['signal_time_R'])<1e-12)
  if not spec['collapse']:
   # Report actual known-task recovery separately; no tuning on a failed test.
   checks[name+'.known_sinusoid_mean_bias_le5']=r['recovery_diagnostics']['circular_bias_le5']
   checks[name+'.known_sinusoid_drift_error_le1']=r['recovery_diagnostics']['drift_error_le1']
   checks[name+'.known_sinusoid_halfturn_exact']=r['recovery_diagnostics']['halfturn_count_exact']
 fig,axs=plt.subplots(3,3,figsize=(18,11),sharex='col')
 for i,spec in enumerate(cases):
  name=spec['name'];r=results[name]
  get=lambda f:arrays[name+'__'+f]
  axs[i,0].plot(tt,get('analytic_ampA'),label='analytic amp A');axs[i,0].plot(tt,get('analytic_ampB'),label='analytic amp B');axs[i,0].set_title(name)
  axs[i,1].plot(tt,get('circular_residual_deg'));axs[i,1].set_title('circular residual; time R %.3f / residual R %.3f'%(r['signal_time_R'],r['residual_R']))
  axs[i,2].plot(tt,get('estimated_delta_unwrapped_deg')-get('estimated_delta_unwrapped_deg')[0],label='estimated')
  axs[i,2].plot(tt,get('truth_delta_unwrapped_deg')-get('truth_delta_unwrapped_deg')[0],ls='--',label='known carrier truth')
  axs[i,2].set_title('net half-turn count estimate/truth %d/%d'%(r['estimated_winding']['net_halfturn_count'],r['truth_winding']['net_halfturn_count']))
  for ax in axs[i]:ax.grid(alpha=.2)
  axs[i,0].legend(fontsize=7);axs[i,2].legend(fontsize=7)
 axs[-1,0].set_xlabel('original time (s)');axs[-1,1].set_xlabel('original time (s)');axs[-1,2].set_xlabel('original time (s)')
 axs[1,1].set_ylabel('signed residual (deg)');axs[1,2].set_ylabel('phase advance from first retained sample (deg)')
 fig.suptitle('Deterministic evidence regression, NOT a replacement for S345 certification');fig.tight_layout();fig.savefig(o/'RESIDUAL_AND_WINDING.png',dpi=140);plt.close(fig)
 unchanged={str(p.relative_to(SOURCE)):sha(p) for p in SOURCE.rglob('*') if p.is_file()}==hashes
 checks['S345_unchanged']=unchanged;checks['deterministic_three_cases_only']=len(results)==3
 dump(o/'REGRESSION_RESULTS.json',dict(cases=results,scope='No Poisson draws, no neural simulation, no full certification grid',truth_warning='Latent phase is not automatically filtered analytic phase near amplitude zero or non-sinusoidal modulation'))
 dump(o/'TEST_RESULTS.json',dict(passed=all(checks.values()),checks=checks,test_scope='evidence schema and explicit sinusoidal known-task recovery; collapse case observational only',original_certificate_replaced=False))
 lines=['# S345认证逐样本证据确定性回归','', '仅三条确定性信号；不重跑神经臂、不重跑原合成网格。明确复现冻结S345顺序：全20s去均值/4阶SOS带通0.5–4Hz → 裁4s/端 → 在12s片段上Hilbert。','',
  '|任务|信号时间R|真值时间R|环形残差R|p95绝对残差°|漂移估计/真值°每秒|净半圈估计/真值|','|---|---:|---:|---:|---:|---|---|']
 for name,r in results.items():lines.append('|%s|%.6f|%.6f|%.6f|%.5f|%.6f / %.6f|%d / %d|'%(name,r['signal_time_R'],r['truth_latent_time_R'],r['residual_R'],r['abs_residual_quantiles_deg']['95'],r['estimated_drift_deg_s'],r['truth_drift_deg_s'],r['estimated_winding']['net_halfturn_count'],r['truth_winding']['net_halfturn_count']))
 lines+=['','幅值塌陷案例中真值仅是明确构造的潜在载波相位；幅值为零时观测瞬时相位不可识别，滤波/解析信号不必保持该潜在相位。因此该例误差不能直接命名为估计器失败，也不能通过新幅值门删除。所有480保留点均保存。',
  '逐案例npz保存全时间轴、完整输入、潜在真值、估计相位、环形残差、解析幅值、unwrap前后增量、累计推进和SOS系数；REGRESSION_RESULTS.json分别报告时间R、残差R与漂移/绕转回收，绝不将三者合并为一个认证门。',
  '', '## 最小生产脚本改动位置（建议，未修改封存文件）',
  '冻结生产脚本 `/home/sensen/Desktop/engine修复验收/s345-twenty-second-engagement/20261027T054112Z/analyze_s345.py` 的 synth/train_rate返回值附近（第199–203行）：返回或旁存原始时间、预设相位轨迹、幅值/占空和随机种子身份。',
  '同脚本认证循环第214–226行：在phase_core调用后、只剩均值前，保存每seed估计轨迹/真值/环形残差/解析幅值/unwrap增量；将时间R与残差R、误差分位、斜率及绕转回收分别汇总。',
  'phase_core第68–93行：仅补可选trace输出接口，不在这项证据回归里改变滤波、Hilbert顺序、幅值门或容差；审计器验证npz可复算和每seed覆盖，不以均值偏差代替逐样本准确度。',
  '', '## 阻断保持',
  '本回归不是原36格认证。S345 NOT_JUDGED、严格验证exit1、28格认证独立阻断和物理交接限制全部保留。']
 (o/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n')
 (o/'SCOPE.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.UTC).isoformat(),neural_ticks=0,physics_steps=0,decoder_calls=0,Poisson_draws=0,parameter_tuning=False),indent=2)+'\n')
 (o/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
 (Path('/home/sensen/Desktop')/'S345_CERT_RECOVERY_RESIDUAL_AND_WINDING.png').write_bytes((o/'RESIDUAL_AND_WINDING.png').read_bytes())
 dump(o/'MANIFEST_SHA256.json',{p.name:sha(p) for p in o.iterdir() if p.is_file() and p.name!='MANIFEST_SHA256.json'})
 print(json.dumps(dict(output=str(o),passed=all(checks.values()),failed=[k for k,v in checks.items() if not v],cases=results),ensure_ascii=False))
 return 0 if all(checks.values()) else 1
if __name__=='__main__':raise SystemExit(main())
