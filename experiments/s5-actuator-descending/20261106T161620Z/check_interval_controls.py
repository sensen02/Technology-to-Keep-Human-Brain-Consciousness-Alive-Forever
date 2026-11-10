import json
from pathlib import Path
import numpy as np
R=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z');old=R/'scaled_body/20261106T193236Z';new=R/'interval_body/20261106T194505Z';checks=[]
for vel in [0.,-.1,.1]:
 for dt in [.03125,.015625]:
  for cond in ['zero_output','sensory_cut']:
   name=f'muscle_tendon_{cond}_p250_dt{dt}.npz';a=np.load(old/f'v{vel}'/name);b=np.load(new/f'v{vel}'/name);neural=bool(np.array_equal(a['spikes_packed'],b['spikes_packed']));assert neural
   physical=bool(np.array_equal(a['rows'],b['rows']))
   if cond=='zero_output':assert physical and np.array_equal(a['commands'],b['commands'])
   checks.append(dict(velocity=vel,dt_ms=dt,condition=cond,neural_bitwise_identical=neural,physical_bitwise_identical=physical))
report=dict(importance='核心必需',verified=True,comparisons=checks,interpretation='cut inputs unchanged across readout revisions, all full-network spikes match; zero output also preserves all physical states; sensory-cut physical changes reflect readout change')
(new/'CROSS_VERSION_CONTROLS.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report))
