"""Compare prespecified physical-feedback doses, not optimization or calibration.

Two new runs (.01/.025 nA maximum) and existing round8 .05 baseline. Same network,
body and decoder; dose changes feed back into body state, so later inputs differ.
"""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/autonomous_dose_round9'
SOURCES={.01:OUT/'dose_001/exploratory.json',.025:OUT/'dose_0025/exploratory.json',.05:ROOT/'outputs/autonomous_sensorimotor_round8/exploratory.json'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    reports={dose:json.loads(p.read_text()) for dose,p in SOURCES.items()}
    base=reports[.05]
    rows=[]
    for dose,d in reports.items():
        assert d['sensory_mode']=='front-club-motion' and d['motor_mapping_policy']=='anatomy-candidate'
        assert d['mapped_actuator_count']==44
        assert d['tier_config']==base['tier_config'] and d['tier_fingerprint']==base['tier_fingerprint']
        assert d['source_sha256']==base['source_sha256'], 'source changed between matched configurations'
        assert d['actuator_mapping']==base['actuator_mapping']
        assert d['translation_laws']['max_sensory_nA']==dose
        selected=[i for i,m in enumerate(d['actuator_mapping']) if m['local']]
        unselected=[i for i,m in enumerate(d['actuator_mapping']) if not m['local']]
        mn=np.unique([i for m in d['actuator_mapping'] for i in m['local']])
        for case,c in d['cases'].items():
            commands=c['commands'];ctrl=np.asarray([r['ctrl'] for r in commands]);network=0;mn_count=0
            assert len(commands)==60 and commands[0]['injected_current_sum_nA']==0
            assert np.all(ctrl[:,unselected]==0)
            for r in commands:
                assert r['command_origin_audit']['status']=='PASS'
                assert r['nonsensory_injected_current_max_nA']==0
                for f in r['features'].values():
                    expected=0. if case=='sensory_cut' else dose*min(abs(f['angular_velocity_rad_s'])/10.,1.)
                    assert f['current_nA']==expected
                ev=np.asarray(r['spikes_sparse_time_neuron'],int).reshape(-1,2)
                assert len(ev)==r['network_spike_count']
                network+=len(ev);mn_count+=int(np.isin(ev[:,1],mn).sum())
            assert c['invariants']['background_max_nA']==0
            if case=='motor_readout_cut':assert np.all(ctrl==0) and c['invariants']['cut_act_max']==0
            rows.append({'dose_max_nA':dose,'case':case,'network_spikes':network,'unique_MN_spikes':mn_count,
                'final_height_mm':c['summary']['final_height_mm'],'min_height_mm':c['summary']['min_height_mm'],
                'mapped_control_saturation_fraction':float((ctrl[:,selected]>=1).mean()),
                'mapped_control_mean':float(ctrl[:,selected].mean()),'first_nonzero_neural_interval_s':next((r['time_s'] for r in commands if r['network_spike_count']),None),
                'initial_current_zero':True,'all_command_origin_audits_pass':True})
    result={'status':'DOSE_SENSITIVITY_NOT_OPTIMIZATION_OR_CALIBRATION','new_simulation_runs':2,
        'existing_baseline_reused':str(SOURCES[.05]),'duration_s':.3,'rows':rows,
        'input_sha256':{str(p):sha(p) for p in SOURCES.values()},'source_sha256':{str(Path(__file__)):sha(__file__)},
        'scope':'Only maximum sensory current differs; no weights/controller/decoder changes. Closed-loop inputs later diverge. No dose declared optimal or physiologically calibrated.'}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'summary.json').write_text(json.dumps(result,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(13,9))
    for dose,d in reports.items():
        c=d['cases']['intact'];b=c['body_per_step'];r=c['commands']
        axs[0,0].plot([v['time_s'] for v in b],[v['height_mm'] for v in b],label=f'{dose:g} nA max')
        axs[0,1].plot([v['time_s'] for v in r],[v['network_spike_count'] for v in r],label=f'{dose:g}')
        chosen=[i for i,m in enumerate(d['actuator_mapping']) if m['local']]
        ctrl=np.asarray([v['ctrl'] for v in r])[:,chosen]
        axs[1,0].plot([v['time_s'] for v in r],(ctrl>=1).mean(axis=1),label=f'{dose:g}')
        axs[1,1].plot([v['time_s'] for v in r],[v['injected_current_sum_nA'] for v in r],label=f'{dose:g}')
    axs[0,0].set_ylabel('Thorax height (mm)');axs[0,1].set_ylabel('Network spikes / 5ms')
    axs[1,0].set_ylabel('44 mapped control saturation fraction');axs[1,1].set_ylabel('Sensory current sum (nA)')
    for ax in axs.flat:ax.set_xlabel('Time (s)');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Actual club-feedback dose sensitivity (intact branches)\nNOT optimized standing, calibrated receptors, or completed autonomy')
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(OUT/'dose_comparison.png',dpi=140);plt.close(fig)
    lines=['# Round9：实际club反馈的预设剂量敏感性','',
        '只新增两次完整运行（.01/.025nA），复用第八轮.05nA基线。所有运行0.3秒、44候选输出，网络权重/身体/解码器不变。没有按站立挑参数。','',
        '|最大单细胞感觉电流nA|分支|网络脉冲|去重MN脉冲|最后高度mm|映射控制饱和比例|','|---|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f"|{r['dose_max_nA']}|{r['case']}|{r['network_spikes']}|{r['unique_MN_spikes']}|{r['final_height_mm']:.6f}|{r['mapped_control_saturation_fraction']:.4f}|")
    lines+=['','所有540区间命令来源检查通过，初始感觉电流均0；46未匹配输出始终0，切断运动读出控制/激活保持0。',
        '这只定位给定有限时段和未标定模型的剂量敏感性；不是生理阈值精确估计。各剂量身体变化后感觉轨迹不同，不是匹配脉冲回放。无剂量被宣称最优或校准。',
        '结果仍为手工探索仿真原型，不是自主行走完成。']
    (OUT/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n')
    for r in rows:print(r)
    return 0
if __name__=='__main__':raise SystemExit(main())
