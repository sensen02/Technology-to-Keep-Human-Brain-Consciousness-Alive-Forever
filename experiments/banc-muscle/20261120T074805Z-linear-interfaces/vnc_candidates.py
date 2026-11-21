"""Index published local pairs from unresolved VNC sources for optional signed functional gains."""
import json
from pathlib import Path
import sys

import numpy as np
from pyarrow import array, compute as pc, feather

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / '20261120T061904Z-independent'
sys.path.insert(0, str(PARENT))
from repair_graph import META, PUBLISHED, permitted_receiver, sha


def build():
    parent = json.loads((PARENT / 'LOCAL_GRAPH.json').read_text())
    for source in (META, PUBLISHED):
        assert sha(source) == parent['input_sha256'][str(source)]
    with np.load(PARENT / 'NETWORK.npz') as archive:
        roots = set(map(str, archive['root_ids']))
    metadata = feather.read_table(META, columns=['banc_888_id','region','super_class','body_part_effector',
                                                  'proofread','neurotransmitter_verified']).to_pylist()
    meta = {r['banc_888_id']: r for r in metadata if r['banc_888_id'] in roots}
    sources = [r for r in parent['remaining_terminal_roots'] if meta[r]['proofread'] == 'TRUE'
               and meta[r]['region'] == 'ventral_nerve_cord'
               and meta[r]['super_class'] == 'ventral_nerve_cord_intrinsic'
               and meta[r]['neurotransmitter_verified'] not in ('acetylcholine', 'gaba')]
    prior = PARENT.parents[1] / 'open-brain/20261120T034137Z-sensory-graph-threshold/GRAPH.json'
    sensory = set(json.loads(prior.read_text())['source_memberships'])
    all_edges = feather.read_table(PUBLISHED, columns=['pre','post','count'])
    rows = all_edges.filter(pc.is_in(all_edges['pre'], value_set=array(sources))).to_pylist()
    pairs = sorted((r['pre'], r['post'], r['count']) for r in rows if r['pre'] != r['post']
                   and r['post'] in meta and permitted_receiver(meta[r['post']], r['post'], sensory))
    assert pairs and len(pairs) == len({p[:2] for p in pairs}) and all(p[2] > 0 for p in pairs)
    pre, post, count = zip(*pairs)
    np.savez_compressed(HERE / 'VNC_CANDIDATES.npz', pre=pre, post=post, published_count=count)
    print('eligible source roots, pairs:', len(set(pre)), len(pre))


if __name__ == '__main__':
    build()
