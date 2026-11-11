"""Freeze genuinely fresh sensory test before correction updates, not motor tests."""
import json,itertools
import numpy as np
from common import HERE,PREVIOUS,GEOMETRY,module,digest,save_json,guard
# Read immutable original implementation using original common module explicitly.
import common
# generate_data imports frozen_sources from old common; bind it only for loading pure episode().
common.frozen_sources=common.original.frozen_sources
legacy=module('_balanced_sensor_original_generation',PREVIOUS/'generate_data.py')

def main():
 guard();old=json.loads((PREVIOUS/'DATASET.json').read_text());g=old['geometry'];rot=np.asarray(g['head_rotation']);forward=rot[:2,0];left=np.array([-forward[1],forward[0]])
 configs=[(a,s,o,seed) for a,s,o,seed in itertools.product([-.10,-.05,.05,.10],[.6,1.25],[.30,.60],[20261124,20261125])]
 configs +=[(0.,0.,o,seed) for o,seed in itertools.product([.30,.60],[20261124,20261125])]
 configs +=[(0.,s,o,seed) for s,o,seed in itertools.product([.6,1.25],[.30,.60],[20261124,20261125])]
 rows=[legacy.episode(*cfg,np.asarray(g['sensor_xy']),np.asarray(g['sampling_midpoint_xy']),forward,left) for cfg in configs]
 p=HERE/'fresh_test.npz'
 if p.exists():raise FileExistsError(p)
 np.savez_compressed(p,receptors=np.stack([r[0] for r in rows]),teacher_turn=np.stack([r[1] for r in rows]),environment_concentration=np.stack([r[2] for r in rows]),time_s=np.arange(240)*.005)
 meta=dict(kind='FRESH_UNSEEN_SENSOR_TEST_FIXED_BEFORE_UPDATES',episodes_config=[r[3] for r in rows],episodes=len(rows),geometry=g,path=str(p),sha256=digest(p),old_train_path=str(PREVIOUS/'train.npz'),old_train_sha256=digest(PREVIOUS/'train.npz'),old_heldout_status='DEVELOPMENT_VALIDATION_HISTORY_NOT_FRESH_TEST',source_hashes={str(f):digest(f) for f in (HERE/'PREREG.json',HERE/'METRIC_CLARIFICATION_BEFORE_TRAINING.json',HERE/'common.py',HERE/'generate_test.py',PREVIOUS/'generate_data.py',PREVIOUS/'DATASET.json',GEOMETRY)})
 save_json(HERE/'FRESH_TEST.json',meta);guard();print(json.dumps(dict(episodes=len(rows),food=32,no_food=4,center=8,frozen_before_training=True)))
if __name__=='__main__':main()
