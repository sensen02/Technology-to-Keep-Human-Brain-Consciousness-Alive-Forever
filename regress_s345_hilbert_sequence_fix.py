"""Single-factor deterministic repair regression, reuses immutable NPZ inputs, no generation."""
from pathlib import Path
import argparse,json,hashlib,datetime
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SRC=Path('/home/sensen/Desktop/engine修复验收/s345-certification-recovery-regression/20261027T102654Z')
S345=Path('/home/sensen/Desktop/engine修复验收/s345-twenty-second-engagement/20261027T054112Z')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def snapshot(root):return {str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())+'\n')
def wrap(v):return (v+180)%360-180
def resultant(v):return float(np.abs(np.mean(np.exp(1j*np.radians(v)))))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output;o.mkdir(parents=True,exist_ok=False)
 pre=json.loads((SRC/'PREREG.json').read_text());limits=pre['limits'];before={str(p):snapshot(p) for p in (SRC,S345)}
 dump(o/'INPUT_HASHES.json',before)
 dump(o/'PREREG.json',dict(utc=datetime.datetime.now(datetime.UTC).isoformat(),source=str(SRC),limits=limits,
  factor='only Hilbert placement before versus after same trim',reuse_inputs=True,new_input_generation=False,filter=pre['filter'],fs_hz=40.,retained_window_s=[4,16],
  truth_domain='same supplied exact-zero flag, no expansion; global recovery across zero UNDEFINED',no_neural_or_decoder_or_physics=True))
 dump(o/'PREREG_FROZEN.json',dict(script_sha256=sha(Path(__file__)),prereg_sha256=sha(o/'PREREG.json')))
 source_results=json.loads((SRC/'RESULTS.json').read_text())['cases'];summary={};allchecks={};plot={}
 for name in pre['cases']:
  z=np.load(SRC/(name+'.npz'));t=z['full_time_s'];tt=z['retained_time_s'];a=z['input_A'];b=z['input_B'];sos=z['filter_sos'];defined=z['truth_defined_retained'].astype(bool)
  assert len(t)==800 and len(tt)==480 and np.array_equal(tt,t[160:640])
  fa=signal.sosfiltfilt(sos,a-a.mean());fb=signal.sosfiltfilt(sos,b-b.mean());truth=z['generator_delta_retained_deg']
  modes={};payload={key:z[key] for key in z.files if key.startswith('generator') or key.startswith('truth_defined') or key in ('full_time_s','retained_time_s','input_A','input_B','filter_sos')}
  for mode in ('original','full_hilbert_then_trim'):
   if mode=='original':ha=signal.hilbert(fa[160:640]);hb=signal.hilbert(fb[160:640])
   if mode=='full_hilbert_then_trim':ha=signal.hilbert(fa)[160:640];hb=signal.hilbert(fb)[160:640]
   u=np.degrees(np.unwrap(np.unwrap(np.angle(ha))-np.unwrap(np.angle(hb))))
   res=wrap(u-truth);valid=res[defined];bias=float(np.degrees(np.angle(np.mean(np.exp(1j*np.radians(valid))))))
   slope=float(np.polyfit(tt-tt[0],u,1)[0]);advance=float(np.median(u[-8:])-np.median(u[:8]));count=int(np.floor(np.abs(advance)/180))
   df=source_results[name]['input_formulas']['df_hz'];truth_slope=-360*df
   truth_advance=-360*df*(float(np.median(tt[-8:]))-float(np.median(tt[:8])));truth_count=int(np.floor(np.abs(truth_advance)/180))
   aq={str(q):float(np.percentile(np.abs(valid),q)) for q in (0,50,90,95,99,100)}
   sq={str(q):float(np.percentile(valid,q)) for q in (0,5,50,95,100)}
   test=dict(bias=bool(np.abs(bias)<=limits['circular_bias_abs_deg']),p95=bool(aq['95']<=limits['p95_abs_deg']),maximum=bool(aq['100']<=limits['max_abs_deg']))
   if defined.all():test.update(drift=bool(np.abs(slope-truth_slope)<=limits['drift_error_abs_deg_s']),halfturn=count==truth_count)
   max_idx=int(np.flatnonzero(defined)[int(np.argmax(np.abs(valid)))])
   modes[mode]=dict(time_R=resultant(u),residual_R=resultant(valid),signed_bias_deg=bias,signed_quantiles_deg=sq,absolute_quantiles_deg=aq,
    max_error_time_s=float(tt[max_idx]),max_error_signed_deg=float(res[max_idx]),max_error_index=max_idx,
    estimated_slope_deg_s=slope,truth_slope_deg_s=truth_slope,drift_error_deg_s=(slope-truth_slope if defined.all() else None),
    advance_deg=advance,estimated_halfturn_count=count,truth_advance_deg=truth_advance,truth_halfturn_count=(truth_count if defined.all() else None),
    crossing_zero_recovery_status=('TESTED' if defined.all() else 'UNDEFINED'),scored_samples=int(defined.sum()),undefined_samples=int((~defined).sum()),
    checks=test,recovery_pass_on_defined_samples=all(test.values()))
   prefix=mode+'__';payload.update({prefix+'estimated_unwrapped_deg':u,prefix+'estimated_wrapped_deg':wrap(u),prefix+'circular_residual_deg':res,
    prefix+'analytic_ampA':np.abs(ha),prefix+'analytic_ampB':np.abs(hb),prefix+'preunwrap_delta_deg':np.degrees(np.angle(ha)-np.angle(hb)),
    prefix+'postunwrap_increment_deg':np.r_[0,np.diff(u)],prefix+'net_advance_deg':u-np.median(u[:8])})
   plot[name+'_'+mode]=dict(time=tt,u=u,ampA=np.abs(ha),ampB=np.abs(hb),res=res)
  assert np.allclose(payload['original__estimated_unwrapped_deg'],z['estimated_delta_unwrapped_deg'],atol=1e-10)
  assert np.array_equal(fa[160:640],signal.sosfiltfilt(sos,a-a.mean())[160:640])
  np.savez_compressed(o/(name+'.npz'),**payload)
  for mode in modes:
   zz=np.load(o/(name+'.npz'));allchecks[name+'.'+mode+'.residual_recompute']=bool(np.allclose(wrap(zz[mode+'__estimated_unwrapped_deg']-zz['generator_delta_retained_deg']),zz[mode+'__circular_residual_deg']))
  allchecks[name+'.original_reproduced']=bool(np.allclose(payload['original__estimated_unwrapped_deg'],z['estimated_delta_unwrapped_deg']))
  allchecks[name+'.input_identical']=bool(np.array_equal(payload['input_A'],a) and np.array_equal(payload['input_B'],b))
  allchecks[name+'.truth_mask_unchanged']=bool(np.array_equal(payload['truth_defined_retained'],z['truth_defined_retained']))
  summary[name]=modes
 fig,axs=plt.subplots(3,3,figsize=(18,11))
 for i,name in enumerate(pre['cases']):
  for mode,color in (('original','tab:orange'),('full_hilbert_then_trim','tab:blue')):
   p=plot[name+'_'+mode]
   axs[i,0].plot(p['time'],p['ampB'],label=mode,color=color)
   axs[i,1].plot(p['time'],p['res'],label=mode,color=color)
   axs[i,2].plot(p['time'],p['u']-np.median(p['u'][:8]),label=mode,color=color)
  z=np.load(SRC/(name+'.npz'));true=z['generator_delta_retained_deg'];axs[i,2].plot(tt,true-np.median(true[:8]),'k--',label='generator analytic reference')
  axs[i,0].set_title(name+' analytic amplitude B');axs[i,1].set_title('signed per-sample residual');axs[i,2].set_title('net advance; undefined across zero stays unscored')
  axs[i,1].axhline(10,c='r',ls='--');axs[i,1].axhline(-10,c='r',ls='--')
  for ax in axs[i]:ax.legend(fontsize=6);ax.grid(alpha=.2);ax.set_xlabel('original time (s)')
 fig.suptitle('Single-factor deterministic repair regression: inputs and all thresholds unchanged');fig.tight_layout();fig.savefig(o/'HILBERT_SEQUENCE_COMPARISON.png',dpi=140);plt.close(fig)
 checks_immutable=all(snapshot(root)==before[str(root)] for root in (SRC,S345));allchecks['source_archives_unchanged']=checks_immutable
 integrity=all(allchecks.values());oldpass=all(v['original']['recovery_pass_on_defined_samples'] for v in summary.values());newpass=all(v['full_hilbert_then_trim']['recovery_pass_on_defined_samples'] for v in summary.values())
 dump(o/'RESULTS.json',dict(cases=summary))
 dump(o/'EVIDENCE_TEST.json',dict(passed=integrity,exit_code=0 if integrity else 1,checks=allchecks))
 dump(o/'RECOVERY_TEST.json',dict(limits=limits,original=dict(passed=oldpass,exit_code=0 if oldpass else 1),repair=dict(passed=newpass,exit_code=0 if newpass else 1),
  undefined_scope='one exact zero phase sample and any across-zero global drift/winding; not passed or failed',original_36_cell_certificate_replaced=False))
 lines=['# Hilbert顺序确定性单因素修复回归','',
  '只复用上一目录三个npz的input_A/input_B、真值、滤波系数和定义域；不重新生成输入、不改变[4,16)窗口、不扩大裁剪、不加幅值门；原阈值完全沿用。',
  '证据完整性exit=%d；原顺序任务回收exit=%d；修复顺序任务回收exit=%d。'%(0 if integrity else 1,0 if oldpass else 1,0 if newpass else 1),
  '', '|案例/顺序|时间R|残差R|p95/max绝对残差°|最大误差时间s|漂移误差°/s|回收|','|---|---:|---:|---|---:|---|---|']
 for name,modes in summary.items():
  for mode,r in modes.items():lines.append('|%s / %s|%.6f|%.6f|%.6f / %.6f|%.3f|%s|%s|'%(name,mode,r['time_R'],r['residual_R'],r['absolute_quantiles_deg']['95'],r['absolute_quantiles_deg']['100'],r['max_error_time_s'],str(r['drift_error_deg_s']),str(r['recovery_pass_on_defined_samples'])))
 lines+=['','## 判读边界','修复是否通过只代表这三个确定性已知任务的证据与回收，不代表Poisson方脉冲的原36格认证通过，也不解释完LA30剩余绕转。幅值精确零点和跨零连续性/绕转保持UNDEFINED，未扩大排除范围。',
  '最小生产修复候选仍是冻结生产脚本phase_core中“trimmed返回裁后实信号→hilbert”的衔接，以及审计脚本band_trim/metrics的对应顺序。只应在独立新版本中调整解析信号构造时机并保留逐样本trace；本次没有修改任何S345封存代码。',
  'S345 NOT_JUDGED、严格验证exit1、28格认证独立阻断及物理交接限制全保留。']
 (o/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n')
 (o/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
 (Path('/home/sensen/Desktop')/'S345_DETERMINISTIC_HILBERT_SEQUENCE_COMPARISON.png').write_bytes((o/'HILBERT_SEQUENCE_COMPARISON.png').read_bytes())
 dump(o/'MANIFEST_SHA256.json',{p.name:sha(p) for p in o.iterdir() if p.is_file() and p.name!='MANIFEST_SHA256.json'})
 print(json.dumps(dict(output=str(o),evidence_exit=0 if integrity else 1,original_recovery_exit=0 if oldpass else 1,repair_recovery_exit=0 if newpass else 1,cases=summary),ensure_ascii=False))
 return 0 if integrity and newpass else 1
if __name__=='__main__':raise SystemExit(main())
