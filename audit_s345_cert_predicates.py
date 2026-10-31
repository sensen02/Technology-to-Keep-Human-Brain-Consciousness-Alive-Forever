"""Read-only analytic audit of existing S345 synthetic certificate; no random draws/grid reruns."""
from pathlib import Path
import json,hashlib,datetime,csv,argparse
import numpy as np
S=Path('/home/sensen/Desktop/engine修复验收/s345-twenty-second-engagement/20261027T054112Z')
DIAGS=[Path('/home/sensen/Desktop/engine修复验收/s345-la30-slip-diagnosis/20261027T080727Z'),Path('/home/sensen/Desktop/engine修复验收/s345-la30-slip-diagnosis/20261027T084034Z')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def snap(p):return {str(f.relative_to(p)):sha(f) for f in p.rglob('*') if f.is_file()}
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def wrap(x):return (x+180.)%360.-180.
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path);o=ap.parse_args().output;o.mkdir(parents=True,exist_ok=False)
 roots=[S]+DIAGS;before={str(p):snap(p) for p in roots};dump(o/'INPUT_HASHES.json',before)
 cert=json.loads((S/'CERT_V22.json').read_text());pre=json.loads((S/'PREREG.json').read_text());rows=[];arrays={}
 dt=.025;start=4.;stop=16.;t=np.arange(start,stop,dt);center=float(t.mean());duration=stop-start
 for key,c in cert['cells'].items():
  df=c['df'];truth=c['phase']-360.*df*t
  z=np.mean(np.exp(1j*np.radians(truth)));idealR=float(np.abs(z));truthmean=float(np.degrees(np.angle(z)))
  codetarget=float(wrap(c['phase']-360.*df*10.));sampletarget=float(wrap(c['phase']-360.*df*center))
  continuousR=float(np.abs(np.sinc(df*duration)))
  biasok=bool(np.abs(c['bias_deg'])<=5.);Rok=bool(c['R']>=.9);nok=bool(c['n']==100 and c['rows']==20000)
  predicates=[]
  if not biasok:predicates.append('ABS_SIGNED_MEAN_BIAS_GT_5')
  if not Rok:predicates.append('MEAN_TEMPORAL_R_LT_0P9')
  if not nok:predicates.append('CERT_COUNT_OR_ROWS_INVALID')
  if not predicates:category='PASS_AS_ARCHIVED'
  if predicates and idealR<.9:category='TASK_PREDICATE_INCOMPATIBLE_FOR_IDEAL_LINEAR_TRUTH'
  if predicates and idealR>=.9:category='INSUFFICIENT_EVIDENCE_TO_ATTRIBUTE_TO_ESTIMATOR_ERROR'
  record=dict(cell=key,phase_initial_deg=c['phase'],duty=c['duty'],rateA_hz=c['rates'][0],rateB_hz=c['rates'][1],df_hz=df,
   input_rows=c['rows'],input_duration_s=20.,trim_s=4.,retained_start_s=start,retained_end_exclusive_s=stop,retained_duration_s=duration,
   phase_samples=len(t),last_sample_s=float(t[-1]),sample_center_s=center,
   true_drift_deg_s=-360*df,true_advance_over12s_deg=-360*df*duration,
   target_used_by_code_deg=codetarget,true_sampled_circular_mean_deg=truthmean,true_sample_center_deg=sampletarget,
   code_target_minus_sample_truth_deg=float(wrap(codetarget-sampletarget)),
   measured_signed_mean_bias_deg=c['bias_deg'],measured_mean_temporal_R=c['R'],n=c['n'],
   theoretical_continuous_R=continuousR,theoretical_sampled_R=idealR,theoretical_R_margin=idealR-.9,
   observed_R_minus_truth_R=c['R']-idealR,noise_R_budget=idealR-.9,
   failure_predicates=','.join(predicates),classification=category,passes_archived=c['passes'],
   task_possible_for_ideal_truth=idealR>=.9,
   what_is_missing='No per-seed waveform/phase trajectory/error distribution or drift-recovery array archived; cannot distinguish filtering, pulse duty geometry, Poisson noise and estimator error from aggregate mean R alone')
  assert c['passes']==bool(biasok and Rok and nok)
  rows.append(record);arrays[key+'_true_phase_deg']=truth
 with (o/'CELL_TABLE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 dump(o/'CELL_TABLE.json',dict(cells=rows,formulas=dict(generator='cycle_A=frac(2*t), cycle_B=frac((2+df)*t-phase_initial/360); rate=rate_value/1000/duty inside pulse',
  ideal_fundamental_delta='phase_initial - 360*df*t',
  R_continuous='abs(sin(pi*df*T)/(pi*df*T)) = abs(sinc(df*T)), T=12s',
  R_discrete='abs(mean(exp(1j*radians(phase_initial-360*df*t)))) with480 samples at25ms',
  mean_bias_code='mean_seed(wrap(measured_circular_mean - wrap(initial -360*df*10s)))',
  R_code='mean_seed(abs(mean_time(exp(1j*measured_delta_phase))))')))
 np.savez_compressed(o/'ANALYTIC_TRUTH.npz',t_original_s=t,**arrays)
 failed=[r for r in rows if not r['passes_archived']];detuned=[r for r in failed if r['df_hz']!=0];static=[r for r in failed if r['df_hz']==0]
 counts={cat:sum(r['classification']==cat for r in rows) for cat in sorted(set(r['classification'] for r in rows))}
 summary=dict(n_cells=len(rows),failed=len(failed),failed_detuned=len(detuned),failed_zero_df=len(static),classification_counts=counts,
  all_bias_pass=all(np.abs(r['measured_signed_mean_bias_deg'])<=5 for r in rows),all_failure_due_to_R=all(r['failure_predicates']=='MEAN_TEMPORAL_R_LT_0P9' for r in failed),
  max_abs_mean_bias=float(max(np.abs(r['measured_signed_mean_bias_deg']) for r in rows)),
  theoretical_detuned_R=next(r['theoretical_sampled_R'] for r in rows if r['df_hz']==.02),
  conclusion='Known detuning does lower temporal R, but ideal df0.02 over12s still yields R>0.9. Thus the18 detuned failed cells cannot all be dismissed as logically impossible tasks. Their small theoretical margin is consumed by waveform/noise/filter effects, unresolved from archived aggregates. All28 failures remain unwaived.',
  signed_bias_limit='Mean signed bias below5deg does not imply per-seed accuracy, RMS error, drift recovery or slip recovery; those were not archived.')
 dump(o/'SUMMARY.json',summary)
 ver=json.loads((S/'verify/VERIFICATION.json').read_text());dump(o/'BLOCKERS_UNCHANGED.json',dict(branch='NOT_JUDGED',strict_verification_exit_code=1,failed_checks=[c for c in ver['checks'] if not c['passed']],certification_failed_cells=len(failed),physics_authorized=False))
 checks=dict(no_new_poisson_draws=True,no_grid_rerun=True,no_new_simulation=True,all36_cells_exported=len(rows)==36,
  archived_failure_predicates_reproduced=len(failed)==28,all_samples480=len(t)==480,
  midpoint_target_equals_circular_truth_with_sampling_offset=all(np.abs(r['code_target_minus_sample_truth_deg'])<=.091 for r in rows),
  continuous_discrete_R_consistent=all(np.abs(r['theoretical_sampled_R']-r['theoretical_continuous_R'])<1e-5 for r in rows),
  all_archives_unchanged=all(snap(p)==before[str(p)] for p in roots))
 assert all(checks.values());dump(o/'VERIFICATION.json',dict(passed=True,checks=checks))
 lines=['# S345 合成认证判据只读审计','', '本次没有生成新Poisson序列、没有重跑3600条合成样本或任何神经/译码仿真；只读原生成器公式与已归档聚合结果。',
  '', '## 判定结果',
  '36格中8格原PASS、28格原FAIL；28格全是R<0.9，带符号均值偏差门全部通过，最大幅值%.6f°。'%summary['max_abs_mean_bias'],
  '18个预设df=0.02Hz格全部失败，10个df=0格失败。没有一格可仅凭现存聚合量确认为“估计器相位偏差>5°”。',
  '', '## 漂移真值与R门是否相容',
  '保留窗[4,16)s，12s；25ms采样共480点。理想相位差 theta(t)=theta0-360*df*t。df=0.02Hz时走相86.4°，理论连续R=abs(sinc(0.24))，离散R=%.9f。'%summary['theoretical_detuned_R'],
  '**理论R仍大于0.9。故“只因真值漂移，R门按构造不可能过”不成立。** 但理想R距离门槛仅约0.008，随机计数、方脉冲谐波/幅值调制、滤波误差均可能使观测降到门下；不能自动豁免，也不能反过来仅因R不足就认定相位测量错误。',
  'df=0时理论参考R=1；10个静态格也R不足。归档没有每种子相位轨迹、误差分布、漂移回收数组，无法把方脉冲任务几何、Poisson噪声、滤波与估计器误差分开。',
  '', '## 相位偏差到底比什么',
  '原代码第214行使用窗口中心10s真值 wrap(theta0-360*df*10)，不是初始真值。第220行每样本先做环形误差，再第224行取100种子的算术均值。对于匀速线性真值且同占空，连续窗的理想圆均值正是窗口中点。',
  '实际采样时间4..15.975s中心为9.9875s，故df0.02时相对代码10s目标差+0.09°，已逐格列出；不足以解释R失败。',
  '这里认证的是**时间圆均值相对中心真值的均值偏差**，没有认证随时间变化真值的逐点恢复、斜率或滑差。正负误差可相互抵消；偏差小不保证每种子误差小。',
  '', '## 失败分类',
  '估计误差已确认：0格（不是证明估计器无误，而是缺逐样本证据）；理想线性真值与门逻辑不相容：0格；证据不足以归因：28格。',
  '逐格完整表见CELL_TABLE.csv/JSON：初相位、占空、两速率、df、窗口、实际带符号偏差、R、真值R、余量、失败谓词与分类均独立列出。',
  '', '## 最小修复位置（未执行）',
  '原冻结生产脚本 synth/train_rate（第199–203行）、target_mid与errs/rs聚合（第214–226行）及审计器certification判定应在下一次预登记中明确分开：相位估计回收 vs 时间相位差聚集。',
  '保留固定门结果，不在本次改0.9。未来若认证仪器，应保存每种子已知真值轨迹与观测相位的带符号环形残差、斜率/滑差误差和不确定性，再判断是估计误差还是被测任务确有漂移/谐波影响。缺的证据不能由目前mean_bias和mean_R反推。',
  '生成器中同一占空比对两信号基波带来相同常相移，理想基波相差仍是theta0-360*df*t；但窄带化的方脉冲解析相位不一定等于此理想基波，因此不能把基波R当作带谐波解析信号的严格上界。',
  '', '## 阻断不变',
  'S345 NOT_JUDGED、严格验证exit1和LA30四项失败均不变；28格认证失败保持独立阻断；不接身体，不进物理。']
 (o/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n')
 dump(o/'SCOPE.json',dict(utc=datetime.datetime.now(datetime.UTC).isoformat(),source=str(S),method='analytic formula evaluation on truth trajectory only; no stochastic rerun',dynamic_calls=0,parameter_changes=0))
 (o/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
 dump(o/'MANIFEST_SHA256.json',{p.name:sha(p) for p in o.iterdir() if p.is_file() and p.name!='MANIFEST_SHA256.json'})
 print(json.dumps(dict(output=str(o),summary=summary,checks=checks),ensure_ascii=False))
if __name__=='__main__':main()
