"""Four deterministic expected-count responses; compare saved random samples only."""
from pathlib import Path
import argparse,json,hashlib,importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
S=Path('/home/sensen/Desktop/engine修复验收/s345-candidate-certification/20261027T125512Z')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())+'\n')
def load(p,n):
 spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def wrap(v):return (v+180)%360-180
def quant(v):return {str(q):float(np.percentile(v,q)) for q in (0,5,50,95,100)}
def main(o):
 o.mkdir(parents=True,exist_ok=False);before={str(f.relative_to(S)):sha(f) for f in S.rglob('*') if f.is_file()};dump(o/'SOURCE_HASHES.json',before)
 m=load(S/'analyze_s345.py','reference_candidate_main');a=load(S/'audit_s345_independent.py','reference_candidate_audit');specs=[(.3,0.),(.5,0.),(.7,0.),(.3,.02)]
 dump(o/'SCOPE_AND_FORMULA.json',dict(formula='lambda_A(t)=105.4/1000/duty if frac(2*t)<duty else0; lambda_B(t)=59.2/1000/duty if frac((2+df)*t-150/360)<duty else0',rows=20000,dt_s=.001,window_s=[4,16],rates_hz=[105.4,59.2],latent_truth='150-360*df*t',
  comparison_limitation='Saved certification samples use105/59Hz, not105.4/59.2. Both exact requested-rate reference and source-rate105/59 reference are calculated with identical fixed waveform/pipeline; stochastic-versus-reference matched comparison uses105/59 only, avoiding attributing rate mismatch to noise.',
  new_Poisson_samples=0,neural_calls=0,scope='deterministic expectation pipeline response, not unique true analytic phase or replacement certification',functions=dict(main=str(S/'analyze_s345.py')+':phase_core',audit=str(S/'audit_s345_independent.py')+':metrics')))
 results={};checks={};plot={};time=np.arange(20000)/1000.
 for duty,df in specs:
  key=f'p150_d{int(duty*10)}_r105_59_df{int(df*100)}';z=np.load(S/'arrays'/f'{key}.npz');case={}
  arrays={'input_time_s':time}
  for role,rates in [('requested',(105.4,59.2)),('source_matched',(105.,59.))]:
   ca=np.where(np.mod(time*2,1)<duty,rates[0]/1000/duty,0.)
   cb=np.where(np.mod(time*(2+df)-150/360,1)<duty,rates[1]/1000/duty,0.)
   arrays[role+'__expected_A_1ms']=ca;arrays[role+'__expected_B_1ms']=cb
   for impl in ('main','audit'):
    bm=25 if impl=='main' else 20;fs=1000/bm;n=20000//bm
    ya=ca.reshape(n,bm).sum(1);yb=cb.reshape(n,bm).sum(1)
    r=m.phase_core(ya,yb,fs,4.,with_bootstrap=False,return_trace=True) if impl=='main' else a.metrics(ya,yb,trim_s=4.,bootstrap=False,return_trace=True)
    tr=r.pop('trace');tt=tr['time_s'];det=tr['phase_unwrapped_deg'];latent=150-360*df*tt;res=wrap(det-latent);latentR=float(np.abs(np.mean(np.exp(1j*np.radians(latent)))))
    result=dict(rates=list(rates),deterministic_time_R=r['R'],latent_time_R=latentR,deterministic_original_R_pass=bool(r['R']>=.9),
     deterministic_vs_latent_signed_error_quantiles_deg=quant(res),deterministic_vs_latent_residual_R=float(np.abs(np.mean(np.exp(1j*np.radians(res))))),
     role='pipeline response reference, not analytic truth')
    pref=role+'__'+impl+'__';arrays.update({pref+k:v for k,v in tr.items()});arrays[pref+'latent_phase_deg']=latent;arrays[pref+'deterministic_minus_latent_deg']=res
    if role=='source_matched':
     random=z[impl+'__phase_unwrapped_deg'];diff=wrap(random-det[None,:]);rR=np.abs(np.mean(np.exp(1j*np.radians(diff)),axis=1));signedq=np.percentile(diff,[0,5,50,95,100],axis=1).T;absq=np.percentile(np.abs(diff),[50,95,99,100],axis=1).T
     result.update(random_vs_matched_reference_residual_R_quantiles=quant(rR),random_vs_matched_reference_signed_quantiles_per_sample_deg=signedq.tolist(),
      random_vs_matched_reference_abs_quantiles_per_sample_deg=absq.tolist(),sample_median_abs_error_deg_quantiles=quant(absq[:,0]),sample_p95_abs_error_deg_quantiles=quant(absq[:,1]),sample_max_abs_error_deg_quantiles=quant(absq[:,3]),n=100)
     arrays[pref+'saved_random_phase_deg']=random;arrays[pref+'random_minus_deterministic_deg']=diff;arrays[pref+'random_residual_R']=rR;arrays[pref+'random_signed_error_quantiles']=signedq
     assert random.shape==diff.shape and random.shape[0]==100
     checks[key+'.'+impl+'.same_time_as_archived_random']=bool(np.array_equal(tt,z[impl+'__time_original_s']))
     plot[key+'_'+impl]=(tt,res,diff,int(np.argsort(rR)[50]),result)
    case[role+'_'+impl]=result
    checks[key+'.'+role+'.'+impl+'.trace_matches_actual_R']=bool(np.isclose(np.abs(np.mean(np.exp(1j*np.radians(det)))),r['R']))
  np.savez_compressed(o/(key+'.npz'),**arrays);results[key]=case
 dump(o/'RESULTS.json',dict(cases=results,interpretation='Do not linearly subtract nonlinear circular resultant lengths or grant exemptions. Deterministic R failure demonstrates waveform+pipeline sufficiency for low temporal concentration, not correctness of every estimated phase.'))
 fig,axs=plt.subplots(4,3,figsize=(18,14))
 for i,(key,case) in enumerate(results.items()):
  for impl,col in [('main','tab:blue'),('audit','tab:orange')]:
   tt,res,diff,j,r=plot[key+'_'+impl]
   axs[i,0].plot(tt,res,label=impl,color=col)
   axs[i,1].plot(tt,diff[j],label=impl+' median residual-R sample%d'%j,color=col)
   rr=np.abs(np.mean(np.exp(1j*np.radians(diff)),axis=1));axs[i,2].hist(rr,bins=15,alpha=.5,label=impl,color=col)
  axs[i,0].set_title(key+' expectation response vs latent')
  axs[i,1].set_title('saved random vs source-rate matched response')
  axs[i,2].set_title('sample residual R distribution')
  for ax in axs[i]:ax.grid(alpha=.2);ax.legend(fontsize=7)
 fig.tight_layout();fig.savefig(o/'WAVEFORM_REFERENCE_RESPONSE.png',dpi=140);plt.close(fig)
 (Path('/home/sensen/Desktop')/'S345_WAVEFORM_REFERENCE_RESPONSE.png').write_bytes((o/'WAVEFORM_REFERENCE_RESPONSE.png').read_bytes())
 checks['source_unchanged']={str(f.relative_to(S)):sha(f) for f in S.rglob('*') if f.is_file()}==before;checks['four_cases']=len(results)==4
 assert all(checks.values());dump(o/'VERIFICATION.json',dict(passed=True,exit_code=0,checks=checks,no_new_random_samples=True))
 lines=['# 四例确定性期望输入管线响应参考','', '原生成器公式仅计算期望计数，不抽取新Poisson样本、不重跑36格/神经臂。固定150°；占空.3/.5/.7无频差与.3/.02Hz，共四例。',
  '**速率口径限制**：用户固定105.4/59.2，但原认证对应100样本为105/59。主表给105.4/59.2，同时提供源速率105/59的严格匹配参考用于随机样本残差；不把0.4/0.2Hz差异归因于噪声。不扫描参数。',
  '', '|格|实现|要求速率确定性R|源速率匹配R|潜在真值R|原R门在要求速率确定性输入上|','|---|---|---:|---:|---:|---|']
 for key,case in results.items():
  for impl in ('main','audit'):
   x=case['requested_'+impl];y=case['source_matched_'+impl];lines.append('|%s|%s|%.6f|%.6f|%.6f|%s|'%(key,impl,x['deterministic_time_R'],y['deterministic_time_R'],x['latent_time_R'],str(x['deterministic_original_R_pass'])))
 lines+=['','每个npz保存期望输入/实际候选函数同次trace/潜在相位差/逐时环形差/对应100条随机相位及相对确定性响应残差。随机残差R和signed/absolute分位逐样本保留。',
  '期望响应是该方脉冲经当前管线的确定性参考，不是非正弦信号唯一真实相位。若其R已低于.9，只能证明无随机计数时原波形+管线已足够产生低R；不能自动豁免随机样本，不能把R线性相减解释剩余噪声。',
  '工程最小位置仍在认证循环的时间R与回收证据命名/适用性分离；不支持据此改0.9门或选新滤波参数。原28格认证阻断/LA30未解析/NOT_JUDGED/严格exit1/物理限制保持。']
 (o/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n');(o/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
 dump(o/'MANIFEST_SHA256.json',{f.name:sha(f) for f in o.iterdir() if f.is_file() and f.name!='MANIFEST_SHA256.json'})
 print(json.dumps(dict(output=str(o),exit_code=0,R={k:{impl:case['requested_'+impl]['deterministic_time_R'] for impl in ('main','audit')} for k,case in results.items()}),ensure_ascii=False))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);main(ap.parse_args().output)
