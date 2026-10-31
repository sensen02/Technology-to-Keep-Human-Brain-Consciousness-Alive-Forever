"""Single-factor, read-only LA30_s0 test: move Hilbert before trim; no new simulation."""
from pathlib import Path
import argparse,json,csv,hashlib,datetime
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
S=Path('/home/sensen/Desktop/engine修复验收/s345-twenty-second-engagement/20261027T054112Z')
D=Path('/home/sensen/Desktop/engine修复验收/s345-la30-slip-diagnosis/20261027T080727Z')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def snapshot(root):return {str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}
def analyze(fa,fb,fs,k,mode):
 a=fa[k:-k];b=fb[k:-k]
 if mode=='trim_then_hilbert':za=signal.hilbert(a);zb=signal.hilbert(b)
 if mode=='hilbert_then_trim':za=signal.hilbert(fa)[k:-k];zb=signal.hilbert(fb)[k:-k]
 pa=np.angle(za);pb=np.angle(zb);ua=np.unwrap(pa);ub=np.unwrap(pb)
 principal=np.angle(za*np.conj(zb));u=np.unwrap(principal);separate=np.unwrap(ua-ub)
 edge=int(.2*fs);baseline=float(np.median(u[:edge]));net=np.degrees(u-baseline)
 advance=float(np.degrees(np.median(u[-edge:])-np.median(u[:edge])))
 mu=np.mean(np.exp(1j*principal));t=4+np.arange(len(a))/fs
 return dict(t=t,a=a,b=b,ampA=np.abs(za),ampB=np.abs(zb),principal=principal,u=u,separate=separate,ua=ua,ub=ub,pa=pa,pb=pb,net=net,
  metrics=dict(slips=int(np.floor(np.abs(advance)/180)),advance_deg=advance,cycles_A=float((ua[-1]-ua[0])/(2*np.pi)),cycles_B=float((ub[-1]-ub[0])/(2*np.pi)),
   phase_deg=float(np.degrees(np.angle(mu))),R=float(np.abs(mu)),rho=float(np.corrcoef(a,b)[0,1]),n=len(a),fs=fs,retained_window_s=[4,16],
   unwrap_path_net_advance_difference_deg=float(np.degrees((np.median(separate[-edge:])-np.median(separate[:edge]))-(np.median(u[-edge:])-np.median(u[:edge]))))))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output;o.mkdir(parents=True,exist_ok=False)
 hashes={'S345':snapshot(S),'prior_diagnosis':snapshot(D)};dump(o/'INPUT_HASHES.json',hashes)
 z=np.load(D/'TRACES.npz');ca=z['raw_count_A'];cb=z['raw_count_B'];assert len(ca)==len(cb)==20000
 trigger=json.loads((D/'FIVE_COUNT_TRIGGERS.json').read_text())['triggers']
 traces={};rows=[];summary={};eventrows=[]
 for bm in (20,25):
  for order in (3,4):
   key=f'b{bm}_o{order}';fs=1000/bm;n=len(ca)//bm
   a=ca.reshape(n,bm).sum(1);b=cb.reshape(n,bm).sum(1)
   sos=signal.butter(order,[.5,4],fs=fs,btype='bandpass',output='sos')
   fa=signal.sosfiltfilt(sos,a-a.mean());fb=signal.sosfiltfilt(sos,b-b.mean());k=int(4*fs)
   both={mode:analyze(fa,fb,fs,k,mode) for mode in ('trim_then_hilbert','hilbert_then_trim')}
   old=both['trim_then_hilbert'];new=both['hilbert_then_trim']
   assert np.array_equal(old['a'],new['a']) and np.array_equal(old['b'],new['b'])
   assert np.allclose(old['u'],z[key+'_product'],atol=1e-10)
   summary[key]={m:r['metrics'] for m,r in both.items()}
   for mode,tr in both.items():
    for field,value in tr.items():
     if isinstance(value,np.ndarray):traces[key+'_'+mode+'_'+field]=value
    for j in range(1,len(tr['t'])):
     rows.append(dict(pipeline=key,mode=mode,time_s=float(tr['t'][j]),ampA=float(tr['ampA'][j]),ampB=float(tr['ampB'][j]),
      signed_phase_deg=float(np.degrees(tr['principal'][j])),raw_increment_deg=float(np.degrees(tr['principal'][j]-tr['principal'][j-1])),
      unwrapped_increment_deg=float(np.degrees(tr['u'][j]-tr['u'][j-1])),net_advance_deg=float(tr['net'][j]),
      running_net_halfturn=int(np.floor(np.abs(tr['net'][j])/180))))
    for evt in trigger:
     j=int(np.argmin(np.abs(tr['t']-evt['time_s'])))
     eventrows.append(dict(pipeline=key,mode=mode,audit_trigger_level=evt['halfturn_level'],reference_time_s=evt['time_s'],sample_time_s=float(tr['t'][j]),
      ampA=float(tr['ampA'][j]),ampB=float(tr['ampB'][j]),ampA_over_median=float(tr['ampA'][j]/np.median(tr['ampA'])),ampB_over_median=float(tr['ampB'][j]/np.median(tr['ampB'])),
      signed_phase_deg=float(np.degrees(tr['principal'][j])),raw_increment_deg=float(np.degrees(tr['principal'][j]-tr['principal'][j-1])),
      unwrap_increment_deg=float(np.degrees(tr['u'][j]-tr['u'][j-1])),net_advance_deg=float(tr['net'][j]),
      running_halfturn=int(np.floor(np.abs(tr['net'][j])/180)),final_slips=tr['metrics']['slips']))
   traces[key+'_full_filtered_A']=fa;traces[key+'_full_filtered_B']=fb
   traces[key+'_full_hilbert_ampA']=np.abs(signal.hilbert(fa));traces[key+'_full_hilbert_ampB']=np.abs(signal.hilbert(fb))
   traces[key+'_full_time_s']=np.arange(n)/fs
   traces[key+'_full_unwrapped_delta']=np.unwrap(np.angle(signal.hilbert(fa)*np.conj(signal.hilbert(fb))))
   for mode,tr in both.items():traces[key+'_'+mode+'_metrics_values']=np.array([tr['metrics']['slips'],tr['metrics']['advance_deg'],tr['metrics']['cycles_A'],tr['metrics']['cycles_B']])
   traces[key+'_filter_sos']=sos
   traces[key+'_old_array_reference_max_error']=np.array(np.max(np.abs(old['u']-z[key+'_product'])))
   traces[key+'_mode_retained_equal']=np.array(True)
   # Keep local plot material without changing diagnostic input.
   traces[key+'_fs']=np.array(fs)
   for mode,tr in both.items():traces[key+'_'+mode+'_event_levels']=np.array([r['running_halfturn'] for r in eventrows if r['pipeline']==key and r['mode']==mode])
   if key=='b20_o3': audit_both=both
 np.savez_compressed(o/'ORDER_COMPARISON_TRACES.npz',**traces)
 for name,data in (('ALL_INCREMENTS.csv',rows),('FIVE_EVENTS_COMPARISON.csv',eventrows)):
  with (o/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 dump(o/'RESULTS.json',dict(pipelines=summary,factor_changed='Hilbert call before versus after identical [4,16) trim',threshold_changed=False,amplitude_mask_added=False))
 dump(o/'FIVE_EVENTS_COMPARISON.json',dict(events=eventrows))
 fig,axs=plt.subplots(4,2,figsize=(17,13),sharex=True)
 for row,(key,res) in enumerate(summary.items()):
  for mode,color in (('trim_then_hilbert','tab:orange'),('hilbert_then_trim','tab:blue')):
   pref=key+'_'+mode+'_';t=traces[pref+'t']
   axs[row,0].plot(t,traces[pref+'net'],label=mode,color=color)
   axs[row,1].plot(t,traces[pref+'ampA']/np.median(traces[pref+'ampA']),label=mode+' A',color=color)
   axs[row,1].plot(t,traces[pref+'ampB']/np.median(traces[pref+'ampB']),label=mode+' B',color=color,ls='--')
  for ax in axs[row]:
   for evt in trigger:ax.axvline(evt['time_s'],c='r',alpha=.15)
   ax.grid(alpha=.2);ax.legend(fontsize=7)
  axs[row,0].set_ylabel('net advance deg');axs[row,0].set_title(key+' | slips old/new %d/%d'%(res['trim_then_hilbert']['slips'],res['hilbert_then_trim']['slips']))
  axs[row,1].set_title('analytic amplitude / own median; filtered real signals identical')
 axs[-1,0].set_xlabel('original time (s)');axs[-1,1].set_xlabel('original time (s)');fig.suptitle('LA30_s0: single-factor Hilbert-before-trim comparison');fig.tight_layout();fig.savefig(o/'HILBERT_ORDER_OVERVIEW.png',dpi=140);plt.close(fig)
 fig,axs=plt.subplots(5,2,figsize=(17,16))
 for row,evt in enumerate(trigger):
  t0=evt['time_s']
  for mode,tr in audit_both.items():
   m=(tr['t']>=t0-.25)&(tr['t']<=t0+.25)
   axs[row,0].plot(tr['t'][m],tr['ampA'][m]/np.median(tr['ampA']),label=mode+' A');axs[row,0].plot(tr['t'][m],tr['ampB'][m]/np.median(tr['ampB']),ls='--',label=mode+' B')
   axs[row,1].plot(tr['t'][m],tr['net'][m],'.-',label=mode)
  for ax in axs[row]:ax.axvline(t0,c='r',alpha=.4);ax.legend(fontsize=7);ax.grid(alpha=.2)
  axs[row,0].set_title('original audit trigger #%d @ %.3fs'%(evt['halfturn_level'],t0));axs[row,1].set_title('net winding, same count definition')
 fig.tight_layout();fig.savefig(o/'FIVE_EVENTS_HILBERT_ORDER.png',dpi=140);plt.close(fig)
 checks=dict(old_traces_reproduced=True,real_filtered_arrays_unchanged=True,same_windows=True,no_new_simulation=True,no_threshold_change=True,
  five_events_all_four_pipelines_and_two_orders=len(eventrows)==40,s345_unchanged=snapshot(S)==hashes['S345'],prior_diagnosis_unchanged=snapshot(D)==hashes['prior_diagnosis'])
 assert all(checks.values());dump(o/'VERIFICATION.json',dict(passed=True,checks=checks))
 lines=['# LA30_s0 Hilbert顺序单因素诊断','', '仅改解析信号构造顺序；原尖峰、箱宽、SOS滤波、阈值、保留窗全部不变。无新增仿真、无幅值门。',
  '|管线|裁后Hilbert计数|全窗Hilbert后裁剪计数|旧净推进°|新净推进°|新A/B周期|','|---|---:|---:|---:|---:|---|']
 for key,v in summary.items():
  a=v['trim_then_hilbert'];b=v['hilbert_then_trim'];lines.append('|%s|%d|%d|%.6f|%.6f|%.6f / %.6f|'%(key,a['slips'],b['slips'],a['advance_deg'],b['advance_deg'],b['cycles_A'],b['cycles_B']))
 lines+=['','逐事件全部数值见 `FIVE_EVENTS_COMPARISON.csv`；逐样本增量、幅值和净绕转见 `ALL_INCREMENTS.csv` 与 `ORDER_COMPARISON_TRACES.npz`。所有幅值只作描述，没有剔除。',
  '即使分歧因顺序改变消失，也仅说明裁后Hilbert的有限窗端点构造影响了解析轨迹绕转；不能认定真实神经锁相已认证。若仍有分歧，记低幅值相位绕转未解析。',
  'S345 NOT_JUDGED、严格验证exit1及28/36格认证失败均保持不变；本诊断不解禁身体或物理。']
 (o/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n')
 for name in ('HILBERT_ORDER_OVERVIEW.png','FIVE_EVENTS_HILBERT_ORDER.png'):(Path('/home/sensen/Desktop')/('S345_LA30_'+name)).write_bytes((o/name).read_bytes())
 dump(o/'SCOPE.json',dict(utc=datetime.datetime.now(datetime.UTC).isoformat(),source_diagnosis=str(D),source_spikes=str(S/'SPK_LA30_s0.npz'),neural_tick=0,decoder_calls=0,physics_step=0))
 (o/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
 dump(o/'MANIFEST_SHA256.json',{p.name:sha(p) for p in o.iterdir() if p.is_file() and p.name!='MANIFEST_SHA256.json'})
 print(json.dumps(dict(output=str(o),summary=summary,checks=checks),ensure_ascii=False))
if __name__=='__main__':main()
