"""Original36x100 synthetic candidate certification only; no neural arms. Frozen per-run."""
from pathlib import Path
import argparse,json,hashlib,difflib,importlib.util,datetime,csv
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
S=Path('/home/sensen/Desktop/engine修复验收/s345-twenty-second-engagement/20261027T054112Z')
C=Path('/home/sensen/Desktop/engine修复验收/s345-hilbert-order-candidate/20261027T123302Z')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())+'\n')
def loadmodule(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def snap(p):return {str(f.relative_to(p)):sha(f) for f in p.rglob('*') if f.is_file()}
def wrap(x):return (x+180)%360-180
def main(o):
 o.mkdir(parents=True,exist_ok=False)
 source=snap(S);dump(o/'S345_INPUT_HASHES.json',source)
 old=json.loads((S/'CERT_V22.json').read_text())['cells'];specs=[]
 for key,c in old.items():specs.append(dict(key=key,phase=c['phase'],duty=c['duty'],rates=c['rates'],df=c['df'],seeds=[7000+i+int(c['phase'])+int(c['duty']*10)+int(c['df']*100) for i in range(100)]))
 pre=dict(utc=datetime.datetime.now(datetime.UTC).isoformat(),cells=specs,rows=20000,input_dt_s=.001,retained_window_s=[4,16],
  thresholds=dict(abs_mean_signed_bias_deg=5.,mean_temporal_R=.9,n=100),
  reference='original target_mid=wrap(phase-360*df*10s); original bias predicate and R predicate unchanged',
  main=dict(bin_ms=25,fs=40,order=4,band=[.5,4]),audit=dict(bin_ms=20,fs=50,order=3,band=[.5,4]),
  candidate_change='only full filtered-window Hilbert before trim plus same-call trace return',
  truth_scope='latent fundamental phase from pulse generator; NOT necessarily filtered analytic phase. Residual/drift metrics are latent-reference diagnostics, not replacement gates.',
  runs=dict(neural_ticks=0,decoder_calls=0,physics_steps=0,certification_samples=3600),save='per-cell compressed NPZ with all100 inputs/traces; no missing trajectories allowed')
 dump(o/'PREREG.json',pre)
 for name in ('analyze_s345.py','audit_s345_independent.py'):(o/name).write_bytes((C/name).read_bytes())
 # Exact executed original generator without stochastic changes or adapter.
 generator='''import numpy as np\ndef synth(rate,phase_deg,duty,df,seed,n=20000):\n    rng=np.random.default_rng(seed);t=np.arange(n)/1000.0;f=2.0\n    def train_rate(rate_value,ph,fr):\n        cyc=np.mod(t*fr-ph/360.0,1.0)\n        lam=np.where(cyc<duty,rate_value/1000.0/duty,0.0)\n        return rng.poisson(lam).astype(float)\n    return train_rate(rate[0],0.0,f),train_rate(rate[1],phase_deg,f+df)\n'''
 (o/'generator.py').write_text(generator);(o/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
 hashes={name:sha(o/name) for name in ('PREREG.json','analyze_s345.py','audit_s345_independent.py','generator.py',Path(__file__).name)}
 dump(o/'FROZEN_SHA256.json',dict(utc=datetime.datetime.now(datetime.UTC).isoformat(),files=hashes))
 m=loadmodule(o/'analyze_s345.py','candidate_main');au=loadmodule(o/'audit_s345_independent.py','candidate_audit');gen=loadmodule(o/'generator.py','original_generator')
 assert m.BIN==25 and au.BIN_MS==20 and au.FS==50
 tables={'main':{},'audit':{}};oldtruth={};integrity={};records=[]
 for spec in specs:
  key=spec['key'];phase,duty,rates,df=spec['phase'],spec['duty'],spec['rates'],spec['df']
  arrays={};summ={'main':[],'audit':[]};countsA=[];countsB=[]
  for sample,seed in enumerate(spec['seeds']):
   ca,cb=gen.synth(tuple(rates),phase,duty,df,seed);assert len(ca)==len(cb)==20000
   countsA.append(ca.astype(np.uint16));countsB.append(cb.astype(np.uint16))
   for impl in ('main','audit'):
    bm=25 if impl=='main' else 20;fs=1000/bm;n=20000//bm
    a=ca.reshape(n,bm).sum(1);b=cb.reshape(n,bm).sum(1)
    r=m.phase_core(a,b,fs,4.,with_bootstrap=False,return_trace=True) if impl=='main' else au.metrics(a,b,trim_s=4.,bootstrap=False,return_trace=True)
    tr=r.pop('trace');tt=tr['time_s'];latent=phase-360*df*tt;estimate=tr['phase_unwrapped_deg'];res=wrap(estimate-latent)
    # Original predicates: one circular mean per sample, center reference; R is temporal concentration.
    original_target=wrap(phase-360*df*10.);bias=wrap(r['phase_deg']-original_target)
    truth_advance=-360*df*(np.median(tt[-int(.2*fs):])-np.median(tt[:int(.2*fs)]));truth_count=int(np.floor(np.abs(truth_advance)/180))
    residual_R=float(np.abs(np.mean(np.exp(1j*np.radians(res)))));drift_error=r['slope_deg_s']+360*df
    row=dict(sample_index=sample,seed=seed,bias_deg=bias,time_R=r['R'],residual_R=residual_R,
      p95_abs_latent_residual_deg=float(np.percentile(np.abs(res),95)),max_abs_latent_residual_deg=float(np.max(np.abs(res))),
      signed_p5_latent_residual_deg=float(np.percentile(res,5)),signed_p50_latent_residual_deg=float(np.percentile(res,50)),signed_p95_latent_residual_deg=float(np.percentile(res,95)),
      drift_deg_s=r['slope_deg_s'],latent_drift_deg_s=-360*df,drift_latent_reference_error_deg_s=drift_error,
      net_advance_deg=r['total_advance_deg'],halfturn_count=r['slip_count'],latent_reference_halfturn_count=truth_count,
      halfturn_latent_reference_error=r['slip_count']-truth_count)
    summ[impl].append(row)
    for field,value in tr.items():
     if field not in ('time_s','retained_window_s'):arrays.setdefault(impl+'__'+field,[]).append(value)
    arrays.setdefault(impl+'__latent_reference_phase_deg',[]).append(latent)
    arrays.setdefault(impl+'__circular_latent_reference_residual_deg',[]).append(res)
    arrays.setdefault(impl+'__drift_deg_s',[]).append(r['slope_deg_s'])
    arrays.setdefault(impl+'__halfturn_count',[]).append(r['slip_count'])
    arrays[impl+'__time_original_s']=tt;arrays[impl+'__retained_window_s']=tr['retained_window_s']
    assert np.abs(wrap(float(np.degrees(np.angle(np.mean(np.exp(1j*np.radians(estimate))))))-r['phase_deg']))<1e-9
  arrays={k:(np.stack(v) if isinstance(v,list) else v) for k,v in arrays.items()}
  arrays.update(input_count_A_1ms=np.stack(countsA),input_count_B_1ms=np.stack(countsB),input_time_s=np.arange(20000)/1000.,seeds=np.asarray(spec['seeds'],np.int64))
  (o/'arrays').mkdir(exist_ok=True);np.savez_compressed(o/'arrays'/f'{key}.npz',**arrays)
  for impl,samples in summ.items():
   bias=float(np.mean([r['bias_deg'] for r in samples]));rr=float(np.mean([r['time_R'] for r in samples]));bp=bool(np.abs(bias)<=5.);rp=bool(rr>=.9)
   cell=dict(**spec,n=len(samples),rows=20000,bias_deg=bias,R=rr,bias_predicate_pass=bp,R_predicate_pass=rp,passes=bool(len(samples)==100 and bp and rp),
    temporal_R_range=[min(r['time_R'] for r in samples),max(r['time_R'] for r in samples)],
    residual_R_mean=float(np.mean([r['residual_R'] for r in samples])),p95_latent_error_mean_deg=float(np.mean([r['p95_abs_latent_residual_deg'] for r in samples])),
    worst_latent_error_deg=float(max(r['max_abs_latent_residual_deg'] for r in samples)),
    mean_drift_latent_error_deg_s=float(np.mean([r['drift_latent_reference_error_deg_s'] for r in samples])),
    latent_error_scope='diagnostic only: latent oscillator reference is not an oracle for pulse-wave filtered analytic phase',
    original_pass=old[key]['passes'],change=('UNCHANGED' if bool(len(samples)==100 and bp and rp)==old[key]['passes'] else ('RECOVERED' if bp and rp else 'REGRESSED')),samples=samples)
   tables[impl][key]=cell
   records.append(dict(implementation=impl,cell=key,phase_deg=phase,duty=duty,rateA_hz=rates[0],rateB_hz=rates[1],df_hz=df,n=len(samples),bias_deg=bias,R=rr,bias_pass=bp,R_pass=rp,pass_cell=cell['passes'],original_pass=old[key]['passes'],change=cell['change'],residual_R_mean=cell['residual_R_mean'],mean_drift_latent_error_deg_s=cell['mean_drift_latent_error_deg_s']))
  integrity[key]=dict(samples=100,both_implementations=True,all_input_rows20000=True)
  print(json.dumps(dict(completed=key,main=tables['main'][key]['passes'],audit=tables['audit'][key]['passes'])),flush=True)
 dump(o/'CERTIFICATION_RESULTS.json',dict(implementations=tables,thresholds=pre['thresholds'],original_failed_cells=[k for k,v in old.items() if not v['passes']],scope=pre['truth_scope']))
 with (o/'CELL_COMPARISON.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 checks=dict(complete36_main=len(tables['main'])==36,complete36_audit=len(tables['audit'])==36,all_n100=all(c['n']==100 for tab in tables.values() for c in tab.values()),source_unchanged=snap(S)==source,
  frozen_hashes=all(sha(o/n)==h for n,h in hashes.items()),no_neural_simulation=True)
 summary={impl:dict(passed_cells=sum(c['passes'] for c in tab.values()),failed_cells=[k for k,c in tab.items() if not c['passes']],
  recovered=[k for k,c in tab.items() if c['change']=='RECOVERED'],regressed=[k for k,c in tab.items() if c['change']=='REGRESSED'],
  bias_failed=[k for k,c in tab.items() if not c['bias_predicate_pass']],R_failed=[k for k,c in tab.items() if not c['R_predicate_pass']],
  exit_code=0 if all(c['passes'] for c in tab.values()) else 1) for impl,tab in tables.items()}
 dump(o/'EVIDENCE_TEST.json',dict(passed=all(checks.values()),exit_code=0 if all(checks.values()) else 1,checks=checks))
 dump(o/'CERTIFICATION_EXITS.json',summary)
 fig,axs=plt.subplots(2,2,figsize=(17,10));keys=list(old);xx=np.arange(36)
 for impl,col in (('main','tab:blue'),('audit','tab:orange')):
  axs[0,0].plot(xx,[tables[impl][k]['R'] for k in keys],'.-',label=impl,color=col)
  axs[0,1].plot(xx,[tables[impl][k]['bias_deg'] for k in keys],'.-',label=impl,color=col)
 axs[0,0].axhline(.9,c='r',ls='--');axs[0,0].set_title('Original R predicate unchanged');axs[0,1].axhline(5,c='r',ls='--');axs[0,1].axhline(-5,c='r',ls='--');axs[0,1].set_title('Original signed mean-bias predicate unchanged')
 # Deterministic representative: first failing cell main, sample median-R ranked, chosen without tuning.
 failing=[k for k,c in tables['main'].items() if not c['passes']];rep=failing[0] if failing else keys[0]
 ranked=np.argsort([s['time_R'] for s in tables['main'][rep]['samples']]);j=int(ranked[len(ranked)//2]);z=np.load(o/'arrays'/f'{rep}.npz')
 tt=z['main__time_original_s'];res=z['main__circular_latent_reference_residual_deg'][j];est=z['main__phase_unwrapped_deg'][j];latent=z['main__latent_reference_phase_deg'][j]
 axs[1,0].plot(tt,res);axs[1,0].set_title(rep+' sample%d latent-reference residual (not analytic truth)'%j);axs[1,0].set_ylabel('deg')
 axs[1,1].plot(tt,est-est[0],label='estimated');axs[1,1].plot(tt,latent-latent[0],'--',label='latent reference');axs[1,1].set_title('Representative time advance; time R and residual R differ')
 for ax in axs.flat:ax.grid(alpha=.2)
 axs[0,0].legend();axs[0,1].legend();axs[1,1].legend();fig.tight_layout();fig.savefig(o/'CANDIDATE_CERTIFICATION.png',dpi=140);plt.close(fig)
 dump(o/'REPRESENTATIVE_SELECTION.json',dict(cell=rep,sample_index=j,seed=specs[keys.index(rep)]['seeds'][j],selection='first main failing cell, median temporal R sample; no parameter selection'))
 (Path('/home/sensen/Desktop')/'S345_CANDIDATE_CERTIFICATION.png').write_bytes((o/'CANDIDATE_CERTIFICATION.png').read_bytes())
 lines=['# S345候选原规模合成认证','', '仅候选合成认证，原36格×100样本×20s；每个样本两实现各自原生预处理，不重采样输入、不跑十条神经臂。唯一算法改动是全窗Hilbert后裁剪及逐样本trace输出。',
  '执行前冻结候选脚本、精确原生成器、入口、36格/种子列表/判据哈希。偏差≤5°、R≥0.9、n=100均原样。',
  '证据exit%d；主认证exit%d；审计认证exit%d。'%(0 if all(checks.values()) else 1,summary['main']['exit_code'],summary['audit']['exit_code']),
  '', '|实现|通过格|仍失败格|原失败恢复格|原通过退化格|','|---|---:|---:|---:|---:|']
 for impl,s in summary.items():lines.append('|%s|%d|%d|%d|%d|'%(impl,s['passed_cells'],len(s['failed_cells']),len(s['recovered']),len(s['regressed'])))
 lines+=['','逐格比较见CELL_COMPARISON.csv；每格arrays/*.npz保存100条原1ms计数、种子、两实现实际时间轴/同次解析幅值/签名相位/展开增量、潜在相位参考及逐时环形残差。CERTIFICATION_RESULTS.json保存每样本漂移/绕转和分位描述，不以新描述量替代原门。',
  '方脉冲生成器的潜在载波相位不自动等于经滤波的解析相位；残差/斜率误差相对latent reference仅诊断，不能自行把它当作唯一正确答案。原36格的均值偏差和时间R门分别保留。',
  '本包不解决LA30低幅值剩余绕转，也不改原S345 NOT_JUDGED/严格exit1/历史28格阻断，不授权物理交接。失败不调参、不追加扫描。']
 (o/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n')
 dump(o/'MANIFEST_SHA256.json',{str(p.relative_to(o)):sha(p) for p in o.rglob('*') if p.is_file() and p.name!='MANIFEST_SHA256.json'})
 print(json.dumps(dict(output=str(o),evidence_exit=0 if all(checks.values()) else 1,certification_exits=summary),ensure_ascii=False))
 return 0 if all(checks.values()) and all(s['exit_code']==0 for s in summary.values()) else 1
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);raise SystemExit(main(ap.parse_args().output))
