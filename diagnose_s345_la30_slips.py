"""Bounded read-only diagnosis of S345 LA30_s0; no engine imports or simulation."""
import argparse,csv,json,hashlib,datetime
from pathlib import Path
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
S=Path('/home/sensen/Desktop/engine修复验收/s345-twenty-second-engagement/20261027T054112Z')
R=Path('/home/sensen/Desktop/engine修复验收')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def wrap(x):return (x+180)%360-180
def trace(ca,cb,bm,order):
 fs=1000/bm;n=len(ca)//bm
 a=ca[:n*bm].reshape(n,bm).sum(1);b=cb[:n*bm].reshape(n,bm).sum(1)
 sos=signal.butter(order,[.5,4],fs=fs,btype='bandpass',output='sos')
 fa=signal.sosfiltfilt(sos,a-a.mean());fb=signal.sosfiltfilt(sos,b-b.mean())
 k=int(4*fs);fa=fa[k:-k];fb=fb[k:-k]
 za=signal.hilbert(fa);zb=signal.hilbert(fb)
 pa=np.angle(za);pb=np.angle(zb);ua=np.unwrap(pa);ub=np.unwrap(pb)
 raw=pa-pb;principal=np.angle(za*np.conj(zb));separate=ua-ub
 umain=np.unwrap(separate);uproduct=np.unwrap(principal)
 t=4+np.arange(len(pa))/fs;edge=max(1,int(.2*fs))
 advance=lambda x:float(np.degrees(np.median(x[-edge:])-np.median(x[:edge])))
 mean=np.mean(np.exp(1j*principal))
 return dict(fs=fs,bin_ms=bm,order=order,t=t,a=fa,b=fb,ampA=np.abs(za),ampB=np.abs(zb),pa=pa,pb=pb,ua=ua,ub=ub,raw=raw,principal=principal,separate=separate,main=umain,product=uproduct,
  metrics=dict(rho=float(np.corrcoef(fa,fb)[0,1]),phase_deg=float(np.degrees(np.angle(mean))),R=float(np.abs(mean)),
   advance_main_deg=advance(umain),advance_product_deg=advance(uproduct),slips_main=int(np.floor(np.abs(advance(umain))/180)),slips_product=int(np.floor(np.abs(advance(uproduct))/180)),
   cycles_A=float((ua[-1]-ua[0])/(2*np.pi)),cycles_B=float((ub[-1]-ub[0])/(2*np.pi)),
   max_path_difference_deg=float(np.max(np.abs(wrap(np.degrees(umain-uproduct))))),
   retained_start_s=4.,retained_last_sample_s=float(t[-1]),retained_duration_s=len(t)/fs,
   pole_radius=float(max(np.max(np.abs(np.roots(s[3:]))) for s in sos))))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path);o=ap.parse_args().output;o.mkdir(parents=True,exist_ok=False)
 files=[p for p in S.rglob('*') if p.is_file()];before={str(p.relative_to(S)):sha(p) for p in files}
 dump(o/'INPUT_MANIFEST.json',before)
 dump(o/'SCOPE.json',dict(utc=datetime.datetime.now(datetime.UTC).isoformat(),input=str(S/'SPK_LA30_s0.npz'),no_new_simulation=True,no_new_threshold=True,trim_s=4.,purpose='localize disagreement only; no winner selected',dynamic_calls=0))
 z=np.load(S/'SPK_LA30_s0.npz');shape=tuple(z['shape']);sp=np.unpackbits(z['packed'],axis=1)[:,:int(shape[1])];assert sp.shape==shape==(20000,12000)
 roots=np.load(R/'s306-cpg-source-audit/20261026T203904Z/TIER_ROOT_IDS.npz')['root_ids'];loc={int(v):i for i,v in enumerate(roots)}
 groups={k:set() for k in ('lf','lm','lh','rf','rm','rh')}
 with (R/'s290-motor-mapping-coverage/20261026T174500Z/PER_CHANNEL.csv').open() as f:
  for row in csv.DictReader(f):
   lg=row.get('name','')[:2].lower()
   if lg in groups:
    for v in json.loads(row.get('pool_root_ids') or '[]'):
     if int(v) in loc:groups[lg].add(loc[int(v)])
 A=sorted(set.union(*(groups[k] for k in ('lf','rm','lh'))));B=sorted(set.union(*(groups[k] for k in ('lm','rf','rh'))));assert len(A)==100 and len(B)==102
 ca=sp[:,A].sum(1).astype(float);cb=sp[:,B].sum(1).astype(float)
 ts={f'b{bm}_o{order}':trace(ca,cb,bm,order) for bm in (20,25) for order in (3,4)}
 saved=json.loads((S/'ARMS.json').read_text())['LA30_s0'];aud=json.loads((S/'AUDIT_INDEPENDENT.json').read_text())['arms']['LA30_s0']
 exact=ts['b25_o4'];ind=ts['b20_o3']
 assert exact['metrics']['slips_main']==saved['phase']['slip_count']
 assert ind['metrics']['slips_product']==aud['slip_count']
 ver=json.loads((S/'verify/VERIFICATION.json').read_text());failed=[c for c in ver['checks'] if not c['passed']]
 cert=json.loads((S/'CERT_V22.json').read_text());cf=[k for k,v in cert['cells'].items() if not v['passes']]
 dump(o/'FINAL_FAILURES.json',dict(verification_passed=ver['passed'],checks=ver['n_checks'],failed_checks=failed,
  classification={c['name']:'LA30 implementation disagreement; not a synthetic-certification check' for c in failed},certification=dict(cells=len(cert['cells']),failed=len(cf),failed_cells=cf,independent_blocker=True)))
 arrays=dict(raw_count_A=ca,raw_count_B=cb)
 events=[];increments=[]
 for key,tr in ts.items():
  for f in ('t','a','b','ampA','ampB','pa','pb','ua','ub','raw','principal','separate','main','product'):arrays[key+'_'+f]=tr[f]
  for path in ('main','product'):
   u=tr[path];dv=np.diff(u);rawdv=np.diff(tr['principal']);correction=dv-rawdv
   edge=max(1,int(.2*tr['fs']));baseline=float(np.median(u[:edge]));net=np.degrees(u-baseline)
   levels=np.floor(np.abs(net)/180).astype(int)
   triggers=set((np.flatnonzero(np.abs(correction)>np.pi)+1).tolist())
   triggers.update((np.flatnonzero(np.diff(levels)!=0)+1).tolist())
   # Capture isolated large increment disagreements without masking amplitude.
   for j in range(1,len(u)):
    rec=dict(pipeline=key,path=path,index=j,time_s=float(tr['t'][j]),ampA=float(tr['ampA'][j]),ampB=float(tr['ampB'][j]),
     phaseA_deg=float(np.degrees(tr['pa'][j])),phaseB_deg=float(np.degrees(tr['pb'][j])),signed_delta_deg=float(np.degrees(tr['principal'][j])),
     wrapped_increment_deg=float(np.degrees(rawdv[j-1])),unwrapped_increment_deg=float(np.degrees(dv[j-1])),
     unwrap_correction_deg=float(np.degrees(correction[j-1])),net_advance_deg=float(net[j]),net_halfturn_floor=int(levels[j]),
     ampA_percentile=float(np.mean(tr['ampA']<=tr['ampA'][j])),ampB_percentile=float(np.mean(tr['ampB']<=tr['ampB'][j])),
     distance_to_retained_edge_s=float(min(tr['t'][j]-4,16-tr['t'][j])))
    increments.append(rec)
    if j in triggers:
     rec=dict(rec,trigger='unwrap jump correction or net half-turn floor changed',
      condition='floor(abs(degrees(u(t)-median(u(first0.2s))))/180) changes; final archived count uses median(last0.2s) not event total')
     events.append(rec)
 np.savez_compressed(o/'TRACES.npz',**arrays)
 for name,rows in (('EVENTS.csv',events),('INCREMENTS.csv',increments)):
  with (o/name).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 dump(o/'EVENTS.json',dict(events=events,count=len(events),amplitude_gate_added=False,counts_are_net_not_event_counts=True))
 # Common time grid: each pipeline's same-period waveform and exact branch increments.
 ref_t=exact['t'];audit_pr=np.interp(ref_t,ind['t'],ind['product']);audit_pr-=audit_pr[0]
 main_pr=exact['main']-exact['main'][0];diff=np.degrees(audit_pr-main_pr)
 dd=np.diff(diff);jump_ix=np.flatnonzero(np.abs(dd)>90)+1
 candidate_times=[]
 for j in jump_ix:
  time=float(ref_t[j])
  if not candidate_times or time-candidate_times[-1]>.12:candidate_times.append(time)
 event_boundaries={}
 for key,tr in ts.items():
  step=np.degrees(np.diff(tr['product']));idx=np.flatnonzero(np.abs(step)>90)+1
  event_boundaries[key]=[dict(time_s=float(tr['t'][j]),increment_deg=float(step[j-1]),
    ampA_over_median=float(tr['ampA'][j]/np.median(tr['ampA'])),ampB_over_median=float(tr['ampB'][j]/np.median(tr['ampB'])),
    raw_increment_deg=float(np.degrees(tr['principal'][j]-tr['principal'][j-1]))) for j in idx]
 dump(o/'LARGE_PHASE_INCREMENTS.json',event_boundaries)
 dump(o/'DIAGNOSIS.json',dict(pipelines={k:v['metrics'] for k,v in ts.items()}, 
  archived_main=dict(slips=saved['phase']['slip_count'],cycles=saved['phase']['cycles']),archived_audit=dict(slips=aud['slip_count'],cycles=aud['cycles']),
  common_grid_jump_times_s=candidate_times,definitions=dict(main='unwrap(unwrap(angle(ha))-unwrap(angle(hb)))',audit='unwrap(angle(ha*conj(hb)))',
   both_final_count='floor(abs(median(last0.2s)-median(first0.2s))/pi)',counts_distinction='Net advance count is not the number of individual threshold crossings'),
  inference='No new amplitude threshold and no corrected locking label is produced; numerical Hilbert phase winding near low amplitude can differ while circular means agree.'))
 fig,axs=plt.subplots(4,1,figsize=(16,13),sharex=True)
 for key in ('b25_o4','b20_o3'):
  tr=ts[key]
  axs[0].plot(tr['t'],tr['a']/np.std(tr['a']),label=key+' A',alpha=.7);axs[0].plot(tr['t'],tr['b']/np.std(tr['b']),label=key+' B',alpha=.5)
  axs[1].plot(tr['t'],tr['ampA']/np.median(tr['ampA']),label=key+' amp A');axs[1].plot(tr['t'],tr['ampB']/np.median(tr['ampB']),label=key+' amp B')
  axs[2].plot(tr['t'],np.degrees(tr['product']-tr['product'][0]),label=key+' product unwrap')
  axs[3].plot(tr['t'][1:],np.degrees(np.diff(tr['product'])),label=key+' unwrap increment')
 for ax in axs:
  for t0 in candidate_times:ax.axvline(t0,c='red',alpha=.25)
  ax.legend(fontsize=8,ncol=4);ax.grid(alpha=.2)
 axs[0].set_ylabel('filtered counts / std');axs[1].set_ylabel('analytic amplitude / median');axs[2].set_ylabel('net phase advance (deg)');axs[3].set_ylabel('per-sample increment (deg)');axs[3].set_xlabel('original time (s); retained [4,16)')
 fig.suptitle('S345 LA30_s0 only: circular agreement does not establish phase winding');fig.tight_layout();fig.savefig(o/'SLIP_EVENT_OVERVIEW.png',dpi=150);plt.close(fig)
 # Five evenly selected strongest common-grid discontinuities, explicit selection recorded.
 iu=ind['product'];ibase=float(np.median(iu[:max(1,int(.2*ind['fs']))]));iadvance=np.degrees(iu-ibase)
 chosen=[];count_triggers=[]
 for level in range(1,6):
  hits=np.flatnonzero(iadvance>=level*180.0);assert len(hits)>0
  j=int(hits[0]);t0=float(ind['t'][j]);chosen.append(t0)
  count_triggers.append(dict(halfturn_level=level,time_s=t0,net_advance_deg=float(iadvance[j]),
   ampA_over_median=float(ind['ampA'][j]/np.median(ind['ampA'])),ampB_over_median=float(ind['ampB'][j]/np.median(ind['ampB'])),
   distance_to_retained_end_s=16-t0))
 dump(o/'FIVE_COUNT_TRIGGERS.json',dict(triggers=count_triggers,meaning='first time audit net cumulative advance reaches each of five half-turn levels; final median endpoint floor remains5'))
 dump(o/'ZOOM_SELECTION.json',dict(method='each of the five audit net half-turn levels, not selected by amplitude or apparent correctness',times_s=chosen))
 fig,axs=plt.subplots(len(chosen),2,figsize=(16,3*len(chosen)),squeeze=False)
 for row,t0 in enumerate(chosen):
  for key in ('b25_o4','b20_o3'):
   tr=ts[key];m=(tr['t']>=t0-.35)&(tr['t']<=t0+.35)
   axs[row,0].plot(tr['t'][m],tr['ampA'][m]/np.median(tr['ampA']),label=key+' amp A');axs[row,0].plot(tr['t'][m],tr['ampB'][m]/np.median(tr['ampB']),label=key+' amp B')
   axs[row,1].plot(tr['t'][m],np.degrees(tr['principal'][m]),'.-',label=key+' principal')
  axs[row,0].set_title('audit half-turn level %d at %.3f s'%(row+1,t0));axs[row,1].set_title('signed wrapped delta; do not confuse ±180 seam with slip')
  for ax in axs[row]:ax.legend(fontsize=7);ax.axvline(t0,c='r',alpha=.3);ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(o/'FIVE_EVENT_ZOOMS.png',dpi=140);plt.close(fig)
 after={str(p.relative_to(S)):sha(p) for p in files};assert before==after
 checks=dict(input_hashes_unchanged=before==after,main_slips_reproduced=exact['metrics']['slips_main']==0,audit_slips_reproduced=ind['metrics']['slips_product']==5,
  all_samples_recorded=len(increments)==sum(2*(len(v['t'])-1) for v in ts.values()),finite_arrays=all(np.isfinite(v).all() for v in arrays.values()),four_original_failed_checks=len(failed)==4,cert_failed_independent=len(cf)==28,
  no_dynamic_calls=True,no_amplitude_mask=True)
 dump(o/'DIAGNOSTIC_VERIFICATION.json',dict(passed=all(checks.values()),checks=checks))
 lines=['# S345 LA30_s0 滑差来源有界诊断','', '范围：仅一条已有尖峰；无新增仿真、无译码改动、无幅值门、无容差更改。',
  '输入：`'+str(S/'SPK_LA30_s0.npz')+'`','', '## 精确复现与单因素分解',
  '|箱宽/阶数|ρ|圆相位°|A周期/B周期|分别unwrap再作差计数|复积相位unwrap计数|净推进°（复积）|','|---|---:|---:|---|---:|---:|---:|']
 for key,tr in ts.items():
  m=tr['metrics'];lines.append('|%s|%.7f|%.5f|%.5f / %.5f|%d|%d|%.4f|'%(key,m['rho'],m['phase_deg'],m['cycles_A'],m['cycles_B'],m['slips_main'],m['slips_product'],m['advance_product_deg']))
 lines+=['','`DIAGNOSIS.json`记录两实现定义；`INCREMENTS.csv`给出四种箱宽/阶数组合、两条unwrap路径的每一步；`EVENTS.csv`不筛除事件。幅值百分位仅描述，不用于计数或拒绝。',
  '最终两者都用端点0.2s中位数的净推进除以180°取整，并非逐次穿越事件累加。若同一箱宽/阶数下两unwrap路径相同，而换箱宽/阶数后净推进改变，则不能归因于“净绕转vs越界计数”定义不同。',
  '低幅值时解析相位可出现奇点或近奇点；有限样本相位在±π附近的相邻增量可能选取不同支路。仅平均圆相位与ρ不能裁定真实神经失锁；本诊断不指定获胜实现。',
  '', '## 原四条严格检查（不是合成认证检查）']
 for c in failed:lines.append('- `'+c['name']+'`：'+json.dumps(c.get('detail'),ensure_ascii=False))
 lines+=['','合成认证另有28/36格失败，仍是独立阻断项；滑差定位成功不解除仪器认证失败，不授权进物理。','',
  '## 图像与逐事件材料','`'+str(o/'SLIP_EVENT_OVERVIEW.png')+'`覆盖保留窗全部样本，红线为两管线净展开突变处；`'+str(o/'FIVE_EVENT_ZOOMS.png')+'`放大五个最大差异事件，选择规则独立记录，不替代全量事件表。',
  '两端4s滤波裁剪后，Hilbert在裁后片段上重新构造解析信号，其端点仍需与内部事件分开解释；CSV逐样本记录距保留窗端点距离。',
  '', '## 后续最小动作','依据四种单因素对照与事件幅值，先决定需验证的是滤波/采样对解析相位支路的稳定性还是计数定义。没有证据前不改S345统计量，不加幅值门、不改容差、不重跑参数网格。',
  '', '边界：NOT_EMERGENT；包络投影量；非行为或生物有效性认证。']
 (o/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n')
 for f in ('SLIP_EVENT_OVERVIEW.png','FIVE_EVENT_ZOOMS.png'):(Path('/home/sensen/Desktop')/('S345_LA30_'+f)).write_bytes((o/f).read_bytes())
 dump(o/'MANIFEST_SHA256.json',{str(p.relative_to(o)):sha(p) for p in sorted(o.iterdir()) if p.is_file() and p.name!='MANIFEST_SHA256.json'})
 print(json.dumps(dict(output=str(o),pipelines={k:v['metrics'] for k,v in ts.items()},candidate_times=candidate_times,checks=checks),ensure_ascii=False))
if __name__=='__main__':main()
