"""Find exact overlapping snippets; expose missing source-recording identities.
No policy performance is examined; longest selection is duration-only.
"""
from pathlib import Path
import json,hashlib
import h5py,numpy as np
ROOT=Path(__file__).resolve().parent;OUT=Path('/home/sensen/Desktop/engine修复验收/flybody-teacher/20261107T180844Z')
FULL=ROOT/'dataset/walking-dataset_female-only_snippets-16252_trk-files-0-9.hdf5'
SMALL=ROOT/'dataset/walking-dataset-small_female-only_snippets-100_min-len-0.5s_trk-files-0-9.hdf5'

def digest(g):
    h=hashlib.sha256()
    for key in ['root_qpos','qpos']:
        x=g[key][()];h.update(str(x.shape).encode());h.update(str(x.dtype).encode());h.update(x.tobytes())
    return h.hexdigest()

def main():
    with h5py.File(SMALL) as small:hsmall={key:digest(g) for key,g in small['trajectories'].items()}
    fullmap={};attrs=set();fields=set()
    with h5py.File(FULL) as full:
        lens=full['trajectory_lengths'][()];longest=int(np.argmax(lens))
        for key,g in full['trajectories'].items():
            fullmap.setdefault(digest(g),[]).append(key);attrs.update(g.attrs.keys());fields.add(tuple(g.keys()))
        picked=full['trajectories'][str(longest).zfill(len(str(len(lens))))]
        allfinite=all(np.isfinite(picked[k][()]).all() for k in picked)
    overlap={key:fullmap.get(d,[]) for key,d in hsmall.items()}
    report=dict(full=str(FULL),small=str(SMALL),method='SHA256 exact shape,dtype,bytes of root_qpos and qpos; detects identical full snippets, not partial overlapping intervals',
        small_exact_matches=sum(bool(v) for v in overlap.values()),small_to_full=overlap,full_duplicate_groups=[v for v in fullmap.values() if len(v)>1],
        all_trajectory_attribute_names=sorted(attrs),all_trajectory_field_sets=[list(v) for v in sorted(fields)],
        dataset_split='NO_OFFICIAL_TRAIN_TEST_SPLIT; small and full must not be called independent',
        recording_and_animal_ids='Use full metadata findings to assess; if absent, source-group generalization split NOT_ESTABLISHED',
        next_reference=dict(rule='maximum full-dataset length, ties lowest index; chosen without policy scoring',index=longest,frames=int(lens[longest]),usable_steps=int(lens[longest]-65),finite=bool(allfinite)))
    with (OUT/'PARTITION_AUDIT.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in ['small_to_full','full_duplicate_groups']},indent=2))

if __name__=='__main__':main()
