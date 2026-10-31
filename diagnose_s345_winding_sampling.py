"""P3/S1: fixed Fourier interpolant winding audit; no new observations or simulation."""
from pathlib import Path
import argparse, json, hashlib, csv
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SOURCE=Path('/home/sensen/Desktop/engine修复验收/s345-la30-slip-diagnosis/20261027T084034Z')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x): p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def snapshot(): return {str(p):sha(p) for p in SOURCE.iterdir() if p.is_file()}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output;o.mkdir(parents=True,exist_ok=False)
 before=snapshot();dump(o/'INPUT_HASHES.json',before)
 dump(o/'PREREGISTRATION.json',dict(scope='LA30_s0 saved full filtered signals only',factors=[1,2,4,8,16],interpolation='scipy.signal.resample real full 20s periodic Fourier interpolant, then Hilbert before trim',window='integer samples [4,16)',endpoint='median first/last 0.2s',count='floor(abs(advance_deg)/180)',threshold_changed=False,amplitude_mask=False,neural_ticks=0,physics_steps=0,limitations='Interpolation assumes a periodic bandlimited continuation; it supplies no new observations and cannot establish physical phase truth. No winner selected.'))
 z=np.load(SOURCE/'ORDER_COMPARISON_TRACES.npz');arrays={};rows=[]
 fig,axs=plt.subplots(4,2,figsize=(15,13))
 for row,key in enumerate(('b20_o3','b20_o4','b25_o3','b25_o4')):
  fs=float(z[key+'_fs']);fa=z[key+'_full_filtered_A'];fb=z[key+'_full_filtered_B'];base={}
  for factor in (1,2,4,8,16):
   hz=fs*factor;n=len(fa)*factor
   fulla=signal.hilbert(signal.resample(fa,n));fullb=signal.hilbert(signal.resample(fb,n))
   # Original knots must be retained by the interpolant, not refiltered.
   knot_error=max(float(np.max(np.abs(fulla[::factor]-signal.hilbert(fa)))),float(np.max(np.abs(fullb[::factor]-signal.hilbert(fb)))))
   assert knot_error<1e-9
   lo=int(4*hz);hi=int(16*hz);a=fulla[lo:hi];b=fullb[lo:hi];t=np.arange(lo,hi)/hz
   principal=np.angle(a*np.conj(b));u=np.unwrap(principal);ua=np.unwrap(np.angle(a));ub=np.unwrap(np.angle(b));separate=ua-ub
   edge=int(.2*hz);advance=lambda v:float(np.degrees(np.median(v[-edge:])-np.median(v[:edge])))
   adv=advance(u);sadv=advance(separate)
   rec=dict(pipeline=key,factor=factor,fs=hz,n=len(t),advance_deg=adv,halfturn_count=int(np.floor(abs(adv)/180)),separate_advance_deg=sadv,separate_halfturn_count=int(np.floor(abs(sadv)/180)),cycles_A=float((ua[-1]-ua[0])/(2*np.pi)),cycles_B=float((ub[-1]-ub[0])/(2*np.pi)),max_product_increment_deg=float(np.max(np.abs(np.degrees(np.diff(u))))),min_amp_A=float(np.min(np.abs(a))),min_amp_B=float(np.min(np.abs(b))),knot_max_error=knot_error)
   if factor==1:
    assert np.allclose(u,z[key+'_hilbert_then_trim_u'],atol=1e-10);base['u']=u
   else:
    # branch disagreement at shared native knots, after aligning the initial branch
    du=u[::factor]-base['u'];du-=du[0];rec['native_knot_branch_difference_turns']=float(np.max(np.abs(du))/(2*np.pi))
   prefix=key+f'_x{factor}_'
   for name,value in dict(t=t,analytic_A=a,analytic_B=b,principal=principal,u=u,ua=ua,ub=ub,separate=separate).items():arrays[prefix+name]=value
   rows.append(rec);axs[row,0].plot(t,np.degrees(u-u[0]),label=f'x{factor}: {rec["halfturn_count"]}')
  p=key+'_x16_';a=arrays[p+'analytic_A'];b=arrays[p+'analytic_B'];t=arrays[p+'t']
  axs[row,1].plot(t,np.abs(a)/np.median(np.abs(a)),label='A');axs[row,1].plot(t,np.abs(b)/np.median(np.abs(b)),label='B')
  axs[row,0].set_title(key+' relative phase advance (deg)');axs[row,1].set_title('x16 amplitude / median; no masking')
  for ax in axs[row]:ax.grid(alpha=.25);ax.legend(fontsize=8);ax.set_xlabel('time (s)')
 np.savez_compressed(o/'TRACES.npz',**arrays)
 with (o/'SAMPLING_COMPARISON.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
 dump(o/'RESULTS.json',dict(rows=rows,classification='NUMERICAL_INTERPOLANT_DIAGNOSIS_ONLY',historical_certification='UNCHANGED_FAIL',physical_authorization=False))
 fig.suptitle('LA30: fixed full-signal Fourier interpolant, not new observations');fig.tight_layout();fig.savefig(o/'WINDING_SAMPLING.png',dpi=140);plt.close(fig)
 assert snapshot()==before
 dump(o/'VERIFICATION.json',dict(passed=True,input_hashes_unchanged=True,native_traces_reproduced=True,original_analytic_knots_retained=True,all_20_cases_saved=len(rows)==20))
 lines=['# LA30 相位绕转采样诊断（P3/S1）','','仅固定插值倍数1/2/4/8/16；原滤波波形/窗口/计数门不变。不是滤波参数搜索，不新增观测。','','|管线|倍数|复积净推进°|半圈计数|分别展开计数|最大相邻增量°|','|---|---:|---:|---:|---:|---:|']
 for r in rows:lines.append('|%s|%d|%.6f|%d|%d|%.3f|'%(r['pipeline'],r['factor'],r['advance_deg'],r['halfturn_count'],r['separate_halfturn_count'],r['max_product_increment_deg']))
 lines+=['','插值收敛只描述该周期Fourier延拓的解析轨迹，不证明神经相位真值。原生样本支路若变化，说明原生复积展开对采样敏感；若各管线高倍率仍分歧，说明不仅是离散展开所致。无幅值筛除，无自动LOCKED/SLIPPING重判。','原S345 NOT_JUDGED、28格失败、严格exit1及未获物理交接均保持。','完整轨迹：'+str(o/'TRACES.npz')]
 (o/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n');(o/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
 Path('/home/sensen/Desktop/S345_WINDING_SAMPLING.png').write_bytes((o/'WINDING_SAMPLING.png').read_bytes())
 dump(o/'MANIFEST_SHA256.json',{p.name:sha(p) for p in o.iterdir() if p.is_file()});print(json.dumps(dict(output=str(o),rows=rows)))
if __name__=='__main__':main()
