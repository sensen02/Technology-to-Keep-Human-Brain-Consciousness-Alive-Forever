"""Native candidate-function tests with explicit fixed sampling adapter; no generated signals or parameter sweep."""
from pathlib import Path
import json,hashlib,difflib,importlib.util,argparse
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
S=Path('/home/sensen/Desktop/engine修复验收/s345-twenty-second-engagement/20261027T054112Z')
C=Path('/home/sensen/Desktop/engine修复验收/s345-hilbert-order-candidate/20261027T110758Z')
I=Path('/home/sensen/Desktop/engine修复验收/s345-certification-recovery-regression/20261027T102654Z')
P=Path('/run/media/sensen/Data2/cell_wound_prototype')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def snap(p):return {str(f.relative_to(p)):sha(f) for f in p.rglob('*') if f.is_file()}
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())+'\n')
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def wrap(v):return (v+180)%360-180
def main(o):
 o.mkdir(parents=True,exist_ok=False);names=['analyze_s345.py','audit_s345_independent.py']
 before={str(x):sha(x) for name in names for x in (S/name,P/name)}
 inputhash={str(I/(n+'.npz')):sha(I/(n+'.npz')) for n in ('static_165','detuned_002','single_amplitude_collapse')}
 dump(o/'SOURCE_HASHES_BEFORE.json',before);dump(o/'INPUT_NPZ_HASHES.json',inputhash)
 for name in names:
  (o/name).write_bytes((C/name).read_bytes())
  (o/(name+'.diff')).write_text(''.join(difflib.unified_diff((S/name).read_text().splitlines(True),(o/name).read_text().splitlines(True),fromfile=str(S/name),tofile=str(o/name))))
 dump(o/'LINEAGE.json',{n:dict(source=str(S/n),candidate=str(o/n),source_sha256=sha(S/n),candidate_sha256=sha(o/n),copied_candidate=str(C/n),candidate_copy_identical=sha(C/n)==sha(o/n)) for n in names})
 pre=json.loads((I/'PREREG.json').read_text());lim=pre['limits'];dump(o/'TEST_PREREG.json',dict(limits=lim,retained_window_s=[4,16],main_native=dict(fs=40,bin_ms=25,order=4),audit_native=dict(fs=50,bin_ms=20,order=3),
  adapter=dict(algorithm='scipy.signal.resample_poly',up=5,down=4,window=['kaiser',5.0],padtype='constant',cval=0.0,antialias='polyphase FIR default design; not a candidate IIR parameter change',output_time='integer index / 50Hz',scope='40Hz saved input adapted into native50Hz function, NOT native50Hz input certification'),
  truth='evaluate archived formula delta_deg=165-360*df*t at physical time; no wrapped-angle interpolation or random regeneration',
  zero='exact t10s generator amplitude zero is UNDEFINED; no expanded exclusion; cross-zero global recovery UNDEFINED',scope='actual candidate functions only; no native main simulation or36-cell grid',window_assertion='exact integer indices and length; floating4ULP representation diagnostic fixed before run; no phase/drift/recovery tolerance changed'))
 dump(o/'TEST_PREREG_FROZEN.json',dict(test_prereg_sha256=sha(o/'TEST_PREREG.json'),driver_sha256=sha(Path(__file__))))
 m=load(o/names[0],'native_candidate_main');a=load(o/names[1],'native_candidate_audit');assert m.BIN==25 and a.BIN_MS==20 and a.FS==50
 evidence={};results={};plot={};window_audit={}
 for name in pre['cases']:
  z=np.load(I/(name+'.npz'));rr={}
  for impl in ('main','audit'):
   fs=40. if impl=='main' else 50.
   if impl=='main':ia=z['input_A'].copy();ib=z['input_B'].copy();t=z['full_time_s'].copy()
   if impl=='audit':
    ia=signal.resample_poly(z['input_A'],5,4,window=('kaiser',5.0),padtype='constant',cval=0.)
    ib=signal.resample_poly(z['input_B'],5,4,window=('kaiser',5.0),padtype='constant',cval=0.);t=np.arange(len(ia))/fs
   df_formula=.02 if name=='detuned_002' else 0.
   truth=165.0-360.0*df_formula*t
   k=int(4*fs);tt=t[k:-k];truth=truth[k:-k]
   defined=np.ones(len(tt),bool)
   if name=='single_amplitude_collapse':defined[tt==10.]=False
   r=(m.phase_core(ia,ib,fs,4.,with_bootstrap=False,return_trace=True) if impl=='main' else a.metrics(ia,ib,trim_s=4.,bootstrap=False,return_trace=True))
   tr=r.pop('trace');est=tr['phase_unwrapped_deg'];res=wrap(est-truth);valid=res[defined]
   q={str(q):float(np.percentile(np.abs(valid),q)) for q in (50,95,99,100)};bias=float(np.degrees(np.angle(np.mean(np.exp(1j*np.radians(valid))))))
   df=.02 if name=='detuned_002' else 0.;truth_slope=-360*df;truth_advance=-360*df*(float(np.median(tt[-int(.2*fs):]))-float(np.median(tt[:int(.2*fs)])));truth_count=int(np.floor(np.abs(truth_advance)/180))
   recovery=dict(bias=bool(np.abs(bias)<=lim['circular_bias_abs_deg']),p95=bool(q['95']<=lim['p95_abs_deg']),maximum=bool(q['100']<=lim['max_abs_deg']))
   if defined.all():recovery.update(drift=bool(np.abs(r['slope_deg_s']-truth_slope)<=lim['drift_error_abs_deg_s']),halfturn=r['slip_count']==truth_count)
   rr[impl]=dict(native_fs_hz=fs,phase_deg=r['phase_deg'],signal_time_R=r['R'],residual_R=float(np.abs(np.mean(np.exp(1j*np.radians(valid))))),signed_bias_deg=bias,
    signed_quantiles_deg={str(q):float(np.percentile(valid,q)) for q in (0,5,50,95,100)},absolute_quantiles_deg=q,
    drift_error_deg_s=(r['slope_deg_s']-truth_slope if defined.all() else None),slope_deg_s=r['slope_deg_s'],net_advance_deg=r['total_advance_deg'],halfturn_count=r['slip_count'],
    cross_zero_recovery=('TESTED' if defined.all() else 'UNDEFINED'),n_scored=int(defined.sum()),n_undefined=int((~defined).sum()),recovery_checks=recovery,recovery_pass=all(recovery.values()))
   payload=dict(full_time_s=t,retained_time_s=tt,input_A=ia,input_B=ib,truth_latent_phase_deg=truth,truth_defined=defined,reference_residual_all_samples_deg=res,**{key:val for key,val in tr.items() if key!='time_s'})
   np.savez_compressed(o/(name+'_'+impl+'.npz'),**payload);plot[name+'_'+impl]=payload
   # Trace must be EXACTLY the arrays used by the actual production function.
   phase=np.angle(np.mean(np.exp(1j*np.radians(est))));R=np.abs(np.mean(np.exp(1j*np.radians(est))))
   edge=int(.2*fs);advance=float(np.median(est[-edge:])-np.median(est[:edge]));count=int(np.floor(np.abs(advance)/180))
   evidence[name+'.'+impl+'.trace_phase_equals_function']=np.abs(wrap(np.degrees(phase)-r['phase_deg']))<1e-9
   evidence[name+'.'+impl+'.trace_R_equals_function']=np.abs(R-r['R'])<1e-12
   evidence[name+'.'+impl+'.trace_count_equals_function']=count==r['slip_count'] and np.isclose(advance,r['total_advance_deg'])
   evidence[name+'.'+impl+'.trace_slope_equals_function']=np.isclose(np.polyfit(tt-tt[0],est,1)[0],r['slope_deg_s'])
   expected_index=np.arange(k,len(ia)-k,dtype=np.int64)
   candidate_index=np.rint(tr['time_s']*fs).astype(np.int64)
   reference_index=np.rint(tt*fs).astype(np.int64)
   difference=tr['time_s']-tt
   integer_match=bool(len(tr['time_s'])==len(expected_index) and np.array_equal(candidate_index,expected_index) and np.array_equal(reference_index,expected_index))
   # Floating representation diagnostic, not a phase/recovery tolerance.
   ulp=np.spacing(np.maximum(np.abs(tt),np.abs(tr['time_s'])))
   rounding_ok=bool(np.all(np.abs(difference)<=4*ulp))
   evidence[name+'.'+impl+'.exact_integer_window']=integer_match and np.array_equal(tr['retained_window_s'],[4,16])
   evidence[name+'.'+impl+'.time_representation_within_machine_rounding']=rounding_ok
   window_audit[name+'.'+impl]=dict(full_n=len(ia),retained_n=len(tt),index_start=int(expected_index[0]),index_end_exclusive=int(expected_index[-1]+1),
    reference_first_s=float(tt[0]),reference_last_s=float(tt[-1]),candidate_first_s=float(tr['time_s'][0]),candidate_last_s=float(tr['time_s'][-1]),
    actual_retained_window_s=tr['retained_window_s'].tolist(),different_float_elements=int(np.count_nonzero(difference)),max_abs_float_difference_s=float(np.max(np.abs(difference))),
    max_difference_ulp=float(np.max(np.abs(difference)/ulp)),exact_integer_indices=integer_match,machine_rounding_ok=rounding_ok)
   np.savez_compressed(o/(name+'_'+impl+'_window_indices.npz'),expected_index=expected_index,candidate_index=candidate_index,reference_index=reference_index,candidate_time_s=tr['time_s'],reference_time_s=tt,difference_s=difference)
   evidence[name+'.'+impl+'.native_parameters_unmodified']=a.FS==50 and a.BIN_MS==20 and m.BIN==25
  results[name]=rr
 after={str(x):sha(x) for n in names for x in (S/n,P/n)};dump(o/'SOURCE_HASHES_AFTER.json',after);evidence['source_scripts_unchanged']=before==after
 evidence['input_npz_unchanged']=all(sha(Path(k))==v for k,v in inputhash.items())
 per_impl={}
 for impl in ('main','audit'):
  ei=all(v for key,v in evidence.items() if ('.'+impl+'.') in key or key in ('source_scripts_unchanged','input_npz_unchanged'))
  ri=all(row[impl]['recovery_pass'] for row in results.values())
  per_impl[impl]=dict(evidence_exit=0 if ei else 1,recovery_exit=0 if ri else 1)
  dump(o/(impl.upper()+'_EVIDENCE_TEST.json'),dict(passed=ei,exit_code=0 if ei else 1,checks={k:v for k,v in evidence.items() if ('.'+impl+'.') in k or k in ('source_scripts_unchanged','input_npz_unchanged')}))
  dump(o/(impl.upper()+'_RECOVERY_TEST.json'),dict(passed_on_defined_scope=ri,exit_code=0 if ri else 1,cases={k:row[impl] for k,row in results.items()}))
 epass=all(evidence.values());rpass=all(v['recovery_pass'] for row in results.values() for v in row.values())
 dump(o/'PER_IMPLEMENTATION_EXITS.json',per_impl)
 dump(o/'WINDOW_AXIS_AUDIT.json',dict(rows=window_audit,assertion='integer sample indices and lengths exact; floating representation separately bounded by4ULP from fixed arithmetic operations, no scientific threshold changed'))
 dump(o/'EVIDENCE_TEST.json',dict(passed=epass,exit_code=0 if epass else 1,checks=evidence));dump(o/'RECOVERY_TEST.json',dict(passed_on_defined_scope=rpass,exit_code=0 if rpass else 1,cases=results,full_certificate_replaced=False))
 fig,axs=plt.subplots(3,2,figsize=(15,11))
 for i,name in enumerate(pre['cases']):
  for impl in ('main','audit'):
   q=plot[name+'_'+impl];tt=q['retained_time_s'];est=q['phase_unwrapped_deg'];axs[i,0].plot(tt,q['reference_residual_all_samples_deg'],label=impl+' native')
   axs[i,1].plot(tt,est-np.median(est[:int(.2*results[name][impl]['native_fs_hz'])]),label=impl+' native')
  for ax in axs[i]:ax.grid(alpha=.2);ax.legend(fontsize=7);ax.set_xlabel('original time s')
  axs[i,0].axhline(10,c='r',ls='--');axs[i,0].axhline(-10,c='r',ls='--');axs[i,0].set_title(name+' actual function residuals');axs[i,1].set_title('same function trace net advance')
 fig.suptitle('Native settings retained; audit input has explicit5/4 fixture adapter');fig.tight_layout();fig.savefig(o/'NATIVE_CANDIDATE_REGRESSION.png',dpi=140);plt.close(fig)
 (Path('/home/sensen/Desktop')/'S345_NATIVE_CANDIDATE_REGRESSION.png').write_bytes((o/'NATIVE_CANDIDATE_REGRESSION.png').read_bytes())
 lines=['# 候选原生参数实际函数回归','', '候选脚本逐字复制上一候选副本；相对S345仅Hilbert-before-trim和可选trace返回，默认调用兼容。精确.diff与来源/候选SHA256已保存。S345及项目两脚本执行前后SHA256一致。',
  '主实现保留40Hz/4阶；审计保留50Hz/3阶，未更改模块全局。已有npz只有40Hz，审计入口使用明确固定5/4多相重采样作为fixture适配，输入源哈希和适配后输入均保存；没有信号再生成。该适配属于测试入口的范围扩展，不是生产候选改动，不得冒充原始50Hz采集证据。',
  '回归确实调用候选phase_core/metrics。trace来源于它们本次使用的相同ha/hb；相位/R/OLS/净推进/计数均从trace复算并与函数返回值断言相等，不另算一条正确轨迹附会。',
  '证据完整性exit%d；任务定义域回收exit%d。'%(0 if epass else 1,0 if rpass else 1),
  '', '|案例|实现|p95/max误差°|漂移误差°/s|回收|','|---|---|---|---|---|']
 for name,row in results.items():
  for impl,r in row.items():lines.append('|%s|%s|%.6f / %.6f|%s|%s|'%(name,impl,r['absolute_quantiles_deg']['95'],r['absolute_quantiles_deg']['100'],str(r['drift_error_deg_s']),str(r['recovery_pass'])))
 lines+=['','幅值零点精确t10s不计对错；跨该点连续性/漂移/净绕转UNDEFINED。所有估计值保留，没有新增生产幅值门。',
  '工程候选回归仅支持下一步决定是否认证；原28格认证阻断、LA30剩余绕转未解析、S345 NOT_JUDGED/严格exit1/禁止物理全部保持。']
 (o/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n');(o/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
 dump(o/'MANIFEST_SHA256.json',{f.name:sha(f) for f in o.iterdir() if f.is_file() and f.name!='MANIFEST_SHA256.json'})
 print(json.dumps(dict(output=str(o),evidence_exit=0 if epass else 1,recovery_exit=0 if rpass else 1,results=results),ensure_ascii=False))
 return 0 if epass and rpass else 1
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path);raise SystemExit(main(ap.parse_args().output))
