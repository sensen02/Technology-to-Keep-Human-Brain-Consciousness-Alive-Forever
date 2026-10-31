"""Bounded candidate-copy engineering regression; no simulation, no certification grid."""
from pathlib import Path
import argparse,json,hashlib,difflib,importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
S=Path('/home/sensen/Desktop/engine修复验收/s345-twenty-second-engagement/20261027T054112Z')
I=Path('/home/sensen/Desktop/engine修复验收/s345-certification-recovery-regression/20261027T102654Z')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())+'\n')
def snap(p):return {str(f.relative_to(p)):sha(f) for f in p.rglob('*') if f.is_file()}
def replace(s,a,b):
 assert s.count(a)==1,a
 return s.replace(a,b)
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def main(o):
 o.mkdir(parents=True,exist_ok=False);before={str(p):snap(p) for p in (S,I)};dump(o/'INPUT_HASHES.json',before)
 source={n:(S/n).read_text() for n in ('analyze_s345.py','audit_s345_independent.py')};changed={}
 s=source['analyze_s345.py']
 s=replace(s,'def trimmed(x,fs,trim_s,order=4,band=BAND):','def trimmed(x,fs,trim_s,order=4,band=BAND,analytic=False):')
 s=replace(s,'    z=y[k:len(y)-k]\n    return z,len(z)/max(len(x),1)','    full=sgn.hilbert(y) if analytic else y\n    z=full[k:len(y)-k]\n    return z,len(z)/max(len(x),1)')
 s=replace(s,'def phase_core(a,b,fs,trim_s,with_bootstrap=True):','def phase_core(a,b,fs,trim_s,with_bootstrap=True,return_trace=False):')
 s=replace(s,'    ya,ret=trimmed(a,fs,trim_s);yb,_=trimmed(b,fs,trim_s)','    ha,ret=trimmed(a,fs,trim_s,analytic=True);hb,_=trimmed(b,fs,trim_s,analytic=True)\n    ya=None if ha is None else ha.real;yb=None if hb is None else hb.real')
 s=replace(s,'    ua=np.unwrap(np.angle(sgn.hilbert(ya)));ub=np.unwrap(np.angle(sgn.hilbert(yb)))','    ua=np.unwrap(np.angle(ha));ub=np.unwrap(np.angle(hb))')
 s=replace(s,'    return dict(phase_deg=float(np.degrees(mu)),R=R,slope_deg_s=float(np.degrees(slope)),slope_ci_deg_s=ci,','    result=dict(phase_deg=float(np.degrees(mu)),R=R,slope_deg_s=float(np.degrees(slope)),slope_ci_deg_s=ci,')
 s=replace(s,'      block_span=circular_span_deg(blocks),retained=ret,cycles=float(min(fa,fb)*n/fs),fa=fa,fb=fb)\ndef bootstrap_slope','      block_span=circular_span_deg(blocks),retained=ret,cycles=float(min(fa,fb)*n/fs),fa=fa,fb=fb)\n    if return_trace:\n        result[\'trace\']=dict(time_s=trim_s+np.arange(n)/fs,ampA=np.abs(ha),ampB=np.abs(hb),phase_unwrapped_deg=np.degrees(unw),phase_wrapped_deg=wrap(np.degrees(unw)),unwrap_increment_deg=np.r_[0,np.degrees(np.diff(unw))],retained_window_s=np.array([trim_s,len(a)/fs-trim_s]))\n    return result\ndef bootstrap_slope')
 changed['analyze_s345.py']=s
 s=source['audit_s345_independent.py']
 s=replace(s,'def band_trim(series, trim_s=TRIM_S):','def band_trim(series, trim_s=TRIM_S, analytic=False):')
 s=replace(s,'    return y[k:-k]','    full = signal.hilbert(y) if analytic else y\n    return full[k:-k]')
 s=replace(s,'def metrics(a, b, trim_s=TRIM_S, bootstrap=True):','def metrics(a, b, trim_s=TRIM_S, bootstrap=True, return_trace=False):')
 s=replace(s,'    ya, yb = band_trim(a, trim_s), band_trim(b, trim_s)','    ha, hb = band_trim(a, trim_s, analytic=True), band_trim(b, trim_s, analytic=True)\n    ya, yb = ha.real, hb.real')
 s=replace(s,'    ha, hb = signal.hilbert(ya), signal.hilbert(yb)\n','')
 s=replace(s,'    return dict(phase_deg=phase, R=resultant, rho=float(np.corrcoef(ya, yb)[0, 1]),','    result = dict(phase_deg=phase, R=resultant, rho=float(np.corrcoef(ya, yb)[0, 1]),')
 s=replace(s,"                drift_status=status)\n","                drift_status=status)\n    if return_trace:\n        result['trace'] = dict(time_s=trim_s+np.arange(len(delta))/FS,ampA=np.abs(ha),ampB=np.abs(hb),phase_unwrapped_deg=np.degrees(delta),phase_wrapped_deg=wrap(np.degrees(delta)),unwrap_increment_deg=np.r_[0,np.degrees(np.diff(delta))],retained_window_s=np.array([trim_s,len(a)/FS-trim_s]))\n    return result\n")
 changed['audit_s345_independent.py']=s
 lineage={}
 for name,text in changed.items():
  (o/name).write_text(text);(o/(name+'.diff')).write_text(''.join(difflib.unified_diff(source[name].splitlines(True),text.splitlines(True),fromfile=str(S/name),tofile=str(o/name))))
  lineage[name]=dict(source=str(S/name),candidate=str(o/name),old_sha256=sha(S/name),new_sha256=sha(o/name))
 dump(o/'LINEAGE_SHA256.json',lineage)
 pre=json.loads((I/'PREREG.json').read_text());dump(o/'REGRESSION_PREREG.json',dict(limits=pre['limits'],inputs=str(I),adapter='For saved40Hz input only: audit.FS=40 and BIN_MS=25 in memory; no interpolation or regeneration. Native audit50Hz settings remain unchanged in candidate source and are NOT tested here.',scope='candidate actual phase_core/metrics only, not candidate main entry or whole certification'))
 m=module(o/'analyze_s345.py','candidate_main');a=module(o/'audit_s345_independent.py','candidate_audit');a.FS=40.;a.BIN_MS=25
 checks={};results={};plot={}
 for name in pre['cases']:
  z=np.load(I/(name+'.npz'));truth=z['generator_delta_retained_deg'];mask=z['truth_defined_retained'].astype(bool);tt=z['retained_time_s'];lim=pre['limits'];cases={}
  for impl,fn in [('main',lambda:m.phase_core(z['input_A'],z['input_B'],40.,4.,with_bootstrap=False,return_trace=True)),('audit',lambda:a.metrics(z['input_A'],z['input_B'],trim_s=4.,bootstrap=False,return_trace=True))]:
   r=fn();tr=r.pop('trace');est=tr['phase_unwrapped_deg'];res=(est-truth+180)%360-180;valid=res[mask]
   bias=float(np.degrees(np.angle(np.mean(np.exp(1j*np.radians(valid))))));p95=float(np.percentile(np.abs(valid),95));maximum=float(np.max(np.abs(valid)))
   df=0.02 if name=='detuned_002' else 0.;truthslope=-360*df;slope=float(r['slope_deg_s']);truthadvance=-360*df*(np.median(tt[-8:])-np.median(tt[:8]));truthcount=int(np.floor(np.abs(truthadvance)/180))
   rc=dict(bias=bool(np.abs(bias)<=lim['circular_bias_abs_deg']),p95=bool(p95<=lim['p95_abs_deg']),maximum=bool(maximum<=lim['max_abs_deg']))
   if mask.all():rc.update(drift=bool(np.abs(slope-truthslope)<=lim['drift_error_abs_deg_s']),halfturn=r['slip_count']==truthcount)
   cases[impl]=dict(bias_deg=bias,p95_abs_deg=p95,max_abs_deg=maximum,time_R=r['R'],residual_R=float(np.abs(np.mean(np.exp(1j*np.radians(valid))))),slope_deg_s=slope,drift_error_deg_s=(slope-truthslope if mask.all() else None),estimated_halfturn_count=r['slip_count'],truth_halfturn_count=(truthcount if mask.all() else None),recovery_checks=rc,passed_defined_samples=all(rc.values()),cross_zero_status=('TESTED' if mask.all() else 'UNDEFINED'),undefined_samples=int((~mask).sum()))
   np.savez_compressed(o/(name+'_'+impl+'.npz'),**tr,truth_unwrapped_deg=truth,truth_defined=mask,circular_residual_deg=res,input_A=z['input_A'],input_B=z['input_B'],full_time_s=z['full_time_s'])
   plot[name+'_'+impl]=(tt,est,res,tr['ampB'])
   checks[name+'.'+impl+'.evidence_fields_finite']=all(np.isfinite(v).all() for v in tr.values())
   checks[name+'.'+impl+'.same_input_window']=bool(np.array_equal(tr['time_s'],tt) and len(est)==480)
   checks[name+'.'+impl+'.residual_recomputable']=bool(np.allclose(res,(tr['phase_unwrapped_deg']-truth+180)%360-180))
  results[name]=cases
 # AST structural confirmation, not a substitute for actual-call regression above.
 for name in changed:
  checks[name+'.Hilbert_before_slice']='full=' in changed[name] or 'full = ' in changed[name]
  checks[name+'.exact_diff_saved']=(o/(name+'.diff')).is_file()
 checks['sealed_inputs_unchanged']=all(snap(p)==before[str(p)] for p in (S,I))
 project=Path('/run/media/sensen/Data2/cell_wound_prototype')
 checks['project_not_overwritten']=all(sha(project/name)==sha(S/name) for name in source)
 epass=all(checks.values());rpass=all(v['passed_defined_samples'] for c in results.values() for v in c.values())
 dump(o/'EVIDENCE_TEST.json',dict(passed=epass,exit_code=0 if epass else 1,checks=checks));dump(o/'RECOVERY_TEST.json',dict(passed_on_defined_scope=rpass,exit_code=0 if rpass else 1,cases=results,undefined_scope='collapse zero sample and across-zero drift/winding not certified',original_certificate_replaced=False))
 fig,axs=plt.subplots(3,2,figsize=(15,11))
 for i,name in enumerate(pre['cases']):
  for impl in ('main','audit'):
   tt,est,res,amp=plot[name+'_'+impl];axs[i,0].plot(tt,res,label=impl);axs[i,1].plot(tt,est-np.median(est[:8]),label=impl)
  axs[i,0].axhline(10,c='r',ls='--');axs[i,0].axhline(-10,c='r',ls='--');axs[i,0].set_title(name+' signed residual');axs[i,1].set_title('actual candidate functions, net phase advance')
  for ax in axs[i]:ax.legend();ax.grid(alpha=.2);ax.set_xlabel('original time s')
 fig.tight_layout();fig.savefig(o/'CANDIDATE_FUNCTION_REGRESSION.png',dpi=140);plt.close(fig)
 (Path('/home/sensen/Desktop')/'S345_CANDIDATE_FUNCTION_REGRESSION.png').write_bytes((o/'CANDIDATE_FUNCTION_REGRESSION.png').read_bytes())
 lines=['# S345 Hilbert顺序候选版本工程回归','', '仅复制封存脚本到本目录；未覆盖封存或项目脚本。精确差异和来源完整路径/旧新SHA256见LINEAGE_SHA256.json及两个.diff。',
  '主候选trimmed仅加可选analytic路径，phase_core调用该路径；审计候选band_trim/metrics对应修改。默认非相位过滤输出不变；计数公式/滤波阶数/频带/trim/阈值/门未改。新增return_trace仅输出证据，不改统计。',
  '直接import候选模块调用实际phase_core/metrics，不调用神经main；没有使用独立诊断复现函数代替。原npz采样率40Hz，因此审计模块仅在测试适配器内FS=40、BIN_MS=25，无重采样；候选审计原生50Hz配置保留、未在本组三输入上认证。',
  '证据完整性exit%d；任务回收exit%d（限真值有定义域）；跨零连续性/绕转UNDEFINED。'%(0 if epass else 1,0 if rpass else 1),
  '', '|任务|实现|p95/max残差°|漂移误差°/s|回收|','|---|---|---|---|---|']
 for name,rr in results.items():
  for impl,r in rr.items():lines.append('|%s|%s|%.6f / %.6f|%s|%s|'%(name,impl,r['p95_abs_deg'],r['max_abs_deg'],str(r['drift_error_deg_s']),str(r['passed_defined_samples'])))
 lines+=['','最小变更位置在两候选脚本的滤波结果与Hilbert衔接；详见.diff。该回归不授权调用候选main重跑认证或神经臂，也不使原S345通过。原28格认证阻断、LA30剩余绕转未解析、NOT_JUDGED/严格exit1/禁止物理均保持。']
 (o/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n');(o/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
 dump(o/'MANIFEST_SHA256.json',{p.name:sha(p) for p in o.iterdir() if p.is_file() and p.name!='MANIFEST_SHA256.json'})
 print(json.dumps(dict(output=str(o),evidence_exit=0 if epass else 1,recovery_exit=0 if rpass else 1,results=results),ensure_ascii=False))
 return 0 if epass and rpass else 1
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();raise SystemExit(main(args.output))
