from scope import *
import shutil
seq=json.loads((HERE/'SEQUENCE_ACCEPTANCE.json').read_text());assert seq['all_pass'];verify_sources(seq['source_sha256'])
physical=[]
for p in sorted(HERE.glob('*/*/TRACE.npz')):
 if p.parent.parent.name=='COMMANDS':continue
 r=json.loads((p.parent/'RESULT.json').read_text());physical.append(dict(path=str(p),steps=r['steps']))
assert len(physical)==56 and sum(r['steps'] for r in physical)==1120000
assert len(list((HERE/'COMMANDS').glob('*/TRACE.npz')))==24
write(HERE/'RUN_TOTALS.json',dict(baseline_evidence=str(HERE/'FINAL_ACCEPTANCE.json'),sequence_evidence=str(HERE/'SEQUENCE_ACCEPTANCE.json'),
 physical_episodes=56,physical_steps=1120000,neural_ticks=480000,training_updates=0,physical=physical,checkpoint_sha256=sha(WEIGHT)))
old=HERE/'REPORT_BASELINE.zh-CN.md'
if not old.exists():shutil.copy2(HERE/'REPORT.zh-CN.md',old)
summary=[]
for r in seq['rows']:
 s=r['release_stats'][1];summary.append(f"| {r['family']} | {s['max_post_forward_off_command']:.6f} | {s['max_post_forward_off_distance']:.6f}mm | {r['forward_arrived_hold']} |")
addon=f'''

## 追加：解除制动时的回弹校正

图像/原生曲线复核发现，基础同时撤推进和制动时，DNg60颈后驱动尾段可回升至.195797，教师/学生额外移动.068237/.143811mm。原持续推进下持稳门与末窗仍通过，但不代表中间完全无回弹；旧FINAL_ACCEPTANCE与REPORT_BASELINE保持。

按RELEASE_SEQUENCE_PREREG，在新相位9337、周期63ticks上，forward发放.24→1.50s，制动.82→1.70s，即先撤推进、200ms后再撤制动；实际仍走同一固定颈部，身体/颈部参数及网络权重全不变。这是脑侧指定发放时序校正，不是Eon自行学会规划。追加三组教师—学生、三次独立学生重放全部通过原主动停/身体门和教师相对门，原生18数组重放精确。

| 组 | forward_off后颈后驱动峰值 | 学生forward_off后最大再移动 | 持稳窗推进到达数 |
|---|---:|---:|---:|
{chr(10).join(summary)}

追加门≤.035驱动、≤.01mm再移动均通过；没有用重新训练或改颈部时间常数抹掉尾段。旧/新例相位和刺激时序不同，比较图展示两套工程配置，不作为只有一个身体变量改变的因果隔离证明。完整发放时序、输入身份和数据见SEQUENCE_ACCEPTANCE.json。

追加9个身体回合180000原生步、3份实际颈后命令60000tick，累计56回合1120000原生步、480000颈部tick、训练0；RUN_TOTALS把基础与追加阶段分开，基础FINAL_ACCEPTANCE计数不追改。RELEASE_SEQUENCE.mp4三行四列为旧同时撤除/新延后撤除各教师学生，1600×720、400帧/25fps/16s/.125x，只去反射不改姿态，全解码并实际打开RELEASE_SEQUENCE_METRICS图。
'''
(HERE/'REPORT.zh-CN.md').write_text(old.read_text()+addon)
with (HERE/'RUN_CHECKLIST.zh-CN.md').open('a') as f:f.write('\n\n追加释放时序：RELEASE_SEQUENCE_PREREG、commands_sequence.py/sequence_run.py/sequence_replay.py/sequence_verify_v2.py/sequence_media.py均exit0，新相位9337三组追加全门通过/18数组精确；新增180k身体步60k颈tick，合计1120k/480k、训练0。旧基线未覆写；新增视频1600×720/400帧/.125x已全解码、实际开图，详见SEQUENCE_ACCEPTANCE及RUN_TOTALS。\n')
for name in ('PIVOT_STOP_CURRENT.json','BODY_COMMAND_CURRENT.json','BODY_STAGE_CURRENT.json'):
 p=BASE/name;value=json.loads(p.read_text());value.update(preferred_brake_release='stop forward spikes then retain brake spikes200ms before release',sequenced_brake_evidence=str(HERE/'SEQUENCE_ACCEPTANCE.json'));write(p,value)
delivery=Path(json.loads((HERE/'DELIVERY.json').read_text())['directory']);names=('REPORT.zh-CN.md','RUN_CHECKLIST.zh-CN.md','SEQUENCE_ACCEPTANCE.json','RUN_TOTALS.json','RELEASE_SEQUENCE_METRICS.png','RELEASE_SEQUENCE.mp4')
files=json.loads((HERE/'DELIVERY.json').read_text())['files']
for n in names:shutil.copy2(HERE/n,delivery/n);assert sha(HERE/n)==sha(delivery/n);files[n]=sha(delivery/n)
write(HERE/'DELIVERY.json',dict(directory=str(delivery),files=files))
print('Final release sequence3/3; cumulative1120000 native steps480000 neck ticks; delivered')
