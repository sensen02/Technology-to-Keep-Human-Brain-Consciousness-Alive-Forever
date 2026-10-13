#!/usr/bin/env python3
"""Counterfactual decoder replay ONLY; no neural/body simulation or controls applied.

Recorded intact and motor_readout_cut have different feedback histories. Do not
interpret replay differences as paired-input experiments or improved behavior.
Reference stays 100 Hz; filter time constants are prespecified sensitivity probes,
NOT biological calibration and NOT chosen using standing performance.
"""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/autonomous_rate_round8'
SOURCE=ROOT/'outputs/autonomous_sensorimotor_round7/exploratory.json'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def exponential(rates,dt,tau,state=None):
    """Causal exact first-order update for substep-binned instantaneous rates."""
    x=np.asarray(rates,float)
    if x.ndim!=2 or not np.isfinite(x).all() or (x<0).any():raise ValueError('finite nonnegative 2D rates required')
    if not np.isfinite(dt) or not np.isfinite(tau) or dt<=0 or tau<=0:raise ValueError('positive finite clocks required')
    z=np.zeros(x.shape[1]) if state is None else np.asarray(state,float).copy()
    if z.shape!=(x.shape[1],) or not np.isfinite(z).all():raise ValueError('invalid state')
    out=np.empty_like(x);a=np.exp(-dt/tau)
    for i,v in enumerate(x):
        z=a*z+(1-a)*v;out[i]=z
    return out,z

def stats(a):
    a=np.asarray(a,float)
    return {'mean_hz':float(a.mean()),'median_hz':float(np.median(a)),
            'p90_hz':float(np.quantile(a,.9)),'p99_hz':float(np.quantile(a,.99)),
            'max_hz':float(a.max()),'saturated_fraction_at_100Hz':float((a>=100).mean()),
            'exact_zero_fraction':float((a==0).mean())}

def selftests():
    rng=np.random.default_rng(0);x=rng.integers(0,3,(120,4)).astype(float)*200
    for tau in (.02,.05,.1):
        whole,_=exponential(x,.0005,tau)
        a,s=exponential(x[:37],.0005,tau);b,s=exponential(x[37:81],.0005,tau,s);c,s=exponential(x[81:],.0005,tau,s)
        np.testing.assert_array_equal(whole,np.vstack((a,b,c)))
        prefix,_=exponential(x[:71],.0005,tau)
        np.testing.assert_array_equal(prefix,whole[:71])
        zero,_=exponential(np.zeros_like(x),.0005,tau)
        assert np.isfinite(whole).all() and not np.any(zero)
    for bad in (np.array([[np.nan]]),np.array([[-1.]])):
        try:exponential(bad,.0005,.02)
        except ValueError:pass
        else:raise AssertionError('invalid rate accepted')
    return 'PASS: exact chunking invariance, prefix causality, finite, zero spikes -> zero, invalid input rejection'

def main():
    tests=selftests();d=json.loads(SOURCE.read_text())
    dt=d['translation_laws']['neural_dt_s'];chunk=d['translation_laws']['chunk_s'];nsub=round(chunk/dt)
    assert np.isclose(nsub*dt,chunk)
    mappings=d['actuator_mapping'];selected=[r for r in mappings if r['local']]
    assert len(selected)==44
    sizes=np.array([len(r['local']) for r in selected]);ng=len(selected)
    allmn=sorted(set(i for r in selected for i in r['local']))
    group_meta=[{'actuator':r['name'],'local_indices':r['local'],'root_ids':r['root_ids'],
                 'n_neurons':len(r['local']),'one_spike_5ms_rate_increment_hz':1/(len(r['local'])*chunk)} for r in selected]
    report={'status':'COUNTERFACTUAL_DECODER_REPLAY_NOT_BODY_IMPROVEMENT', 'description':__doc__,
            'selftests':tests,'reference_hz':100,'prespecified_tau_ms':[20,50,100],
            'source_sha256':{str(SOURCE):sha(SOURCE),str(Path(__file__).resolve()):sha(__file__)},
            'recorded_run_source_sha256':d['source_sha256'],
            'dt_s':dt,'command_interval_s':chunk,'groups':group_meta,
            'group_size_histogram':{str(n):int((sizes==n).sum()) for n in np.unique(sizes)},
            'denominator_note':'All aggregate command statistics exclude46 unmapped zero-output actuators; groups may share MNs. Active MN metrics deduplicate.',
            'cases':{}}
    plot_data={}
    for name in ('intact','motor_readout_cut'):
        commands=d['cases'][name]['commands'];nt=len(commands);steps=nt*nsub
        count=np.zeros((steps,ng),int);per_neuron=np.zeros((nt,len(allmn)),int)
        for k,row in enumerate(commands):
            ev=np.asarray(row['spikes_sparse_time_neuron'],int).reshape(-1,2)
            assert not len(ev) or ((ev[:,0]>=0).all() and (ev[:,0]<nsub).all())
            for j,r in enumerate(selected):
                hit=ev[np.isin(ev[:,1],r['local'])]
                count[k*nsub:(k+1)*nsub,j]=np.bincount(hit[:,0],minlength=nsub)
            for j,i in enumerate(allmn):per_neuron[k,j]=np.count_nonzero(ev[:,1]==i)
        inst=count/(sizes[None,:]*dt)
        box=count.reshape(nt,nsub,ng).sum(axis=1)/(sizes[None,:]*chunk)
        stored=np.array([[row['mn_rate_hz'][r['actuator']] for r in selected] for row in commands])
        np.testing.assert_allclose(box,stored,rtol=0,atol=0)
        recorded_ctrl=np.array([[row['ctrl'][r['actuator']] for r in selected] for row in commands])
        np.testing.assert_array_equal(recorded_ctrl,np.clip(box/100,0,1) if name=='intact' else np.zeros_like(box))
        variants={'boxcar5ms':box}
        for ms in (20,50,100):
            filt,_=exponential(inst,dt,ms/1000)
            variants['exp'+str(ms)+'ms']=filt[nsub-1::nsub]
        total_count=count.sum(axis=0);whole_rate=total_count/(sizes*steps*dt)
        group_rows=[]
        for j,g in enumerate(group_meta):
            group_rows.append({'actuator':g['actuator'],'n_neurons':int(sizes[j]),
                'whole_episode_mean_hz':float(whole_rate[j]),
                'active_MN_fraction_whole_episode':float((per_neuron[:,[allmn.index(i) for i in g['local_indices']]].sum(axis=0)>0).mean()),
                'statistics':{v:stats(a[:,j]) for v,a in variants.items()}})
        windowcount=count.reshape(nt,nsub,ng).sum(axis=1)
        saturated=box>=100;single=(windowcount==1)
        full100=count.reshape(3,round(.1/dt),ng).sum(axis=1)/(sizes[None,:]*.1)
        item={'statistics':{v:stats(a) for v,a in variants.items()},'per_group':group_rows,
              'whole_episode_mean_group_rate':stats(whole_rate),
              'fraction_groups_whole_episode_mean_ge100Hz':float((whole_rate>=100).mean()),
              'fraction_nonoverlapping_100ms_group_windows_ge100Hz':float((full100>=100).mean()),
              'saturated_5ms_windows_single_spike_fraction':float((saturated&single).sum()/max(saturated.sum(),1)),
              'fraction_5ms_group_windows_zero_spikes':float((windowcount==0).mean()),
              'fraction_unique_MNs_active_whole_episode':float((per_neuron.sum(axis=0)>0).mean()),
              'mean_fraction_unique_MNs_active_per_5ms':float((per_neuron>0).mean()),
              'n_unique_MNs':len(allmn),'recorded_control_saturated_fraction':float((recorded_ctrl>=1).mean()),
              'counterfactual_cut_semantics':'Filters analyze pre-cut MN spikes; actual recorded cut controls remain zero. No alternative controls applied.',
              'traces_hz':{v:a.tolist() for v,a in variants.items()}}
        report['cases'][name]=item;plot_data[name]=variants
    report['interpretation']=[
        'A5ms group mean has quantization step200/N Hz. N1 one spike=200Hz and N2 one spike=100Hz; a single spike can saturate without sustained100Hz firing.',
        'Episode/100ms rates measure sustained aggregate recorded firing independently of short-window quantization. They are simulated activity, not genuine physiological measurements.',
        'Causal filtering trades transient peaks for lag and residual activity; exact-zero intervals naturally decrease. Tau values are sensitivity probes, not selected optimum.',
        'Any group episode rate>=100Hz cannot have its high average explained solely by one short-window spike. Smoothing does not calibrate neural or muscle physiology.',
        'Recommendation: expose count/window/group-size diagnostics, characterize recruitment data before choosing filter/reference; no production modification recommended from replay alone.'
    ]
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'report.json').write_text(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(13,9))
    for col,name in enumerate(plot_data):
        vs=plot_data[name];t=(np.arange(len(vs['boxcar5ms']))+1)*chunk*1000
        for key,a in vs.items():
            axs[0,col].plot(t,(a>=100).mean(axis=1),label=key)
            s=np.sort(a.ravel());axs[1,col].plot(s,np.arange(1,len(s)+1)/len(s),label=key)
        axs[0,col].set(title=name+' — counterfactual decoder',xlabel='Time(ms), command interval',ylabel='Fraction44 groups rate>=100Hz',ylim=(0,1))
        axs[1,col].axvline(100,ls='--',color='black');axs[1,col].set(xlabel='Group mean firing rate(Hz)',ylabel='Empirical CDF')
    for ax in axs.flat:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Recorded MN spike replay: 5ms quantization vs causal exponential filters\nReference100Hz unchanged; NOT body improvement or biological calibration',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(OUT/'sensitivity.png',dpi=140);plt.close(fig)
    lines=['# Round8：记录脉冲的解码灵敏度分析','',__doc__,'',tests,'','只分析44个已映射肌肉群体，排除46个未映射零值；不运行网络、身体或应用控制。','']
    for name,c in report['cases'].items():
        lines+=['## '+name,'|解码|饱和比例|平均Hz|零值比例|','|---|---:|---:|---:|']
        for v,s in c['statistics'].items():lines.append(f"|{v}|{s['saturated_fraction_at_100Hz']:.4f}|{s['mean_hz']:.2f}|{s['exact_zero_fraction']:.4f}|")
        lines += [f"整段平均≥100Hz的群体比例：{c['fraction_groups_whole_episode_mean_ge100Hz']:.4f}；100ms窗≥100Hz比例：{c['fraction_nonoverlapping_100ms_group_windows_ge100Hz']:.4f}。",f"5ms饱和窗中只有一个脉冲的比例：{c['saturated_5ms_windows_single_spike_fraction']:.4f}。",f"记录实际控制饱和比例：{c['recorded_control_saturated_fraction']:.4f}。",'']
    lines+=['## 结论边界','5ms计数量化和持续较高的模拟发放必须分开。滤波只能改变瞬态/时延，不能让未经验证的网络或肌肉参数成为生理标定。运动读出切断组分析的是切断前脉冲，实际命令仍然全零；两个分支已有不同反馈历史，不是匹配输入。','建议先显示每群体N、脉冲计数和时窗，依据招募/生理数据再选解码器。本轮不推荐自动修改生产增益或滤波常数，更不按站立效果挑参数。']
    (OUT/'REPORT.zh-CN.md').write_text('\n'.join(lines))
    print(tests)
    for name,c in report['cases'].items():
        print(name,{v:round(s['saturated_fraction_at_100Hz'],4) for v,s in c['statistics'].items()},'whole>=100',c['fraction_groups_whole_episode_mean_ge100Hz'],'singleSpikeSat',c['saturated_5ms_windows_single_spike_fraction'])
    print(OUT)

if __name__=='__main__':main()
