"""Stage supplemental low cameras with a frozen image-trained checkpoint."""
import json,os
from pathlib import Path
d=Path('all_training');(d/'images').mkdir(parents=True)
base=json.loads(Path('seeded/transforms.json').read_text())['frames']
more=json.loads(Path('low/calibration.json').read_text())['frames']
assert len(more)==64 and all(p['split']=='train' for p in more)
for root,items in [(Path('seeded'),base),(Path('low'),more)]:
    for it in items:
        name=it['file_path']
        os.link(root/name,d/name)
os.link('seeded-checkpoints/export_18500.ply',d/'init.ply')
(d/'transforms.json').write_text(json.dumps(dict(frames=base+more,ply_file_path='init.ply')))
print('Training records',len(base)+len(more))
