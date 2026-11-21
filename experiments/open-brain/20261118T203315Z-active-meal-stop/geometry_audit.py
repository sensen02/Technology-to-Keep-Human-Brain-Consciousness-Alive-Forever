"""Audit set 4 BANC fragments against the existing VNC connective registration."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(trial, marker_directory, output):
    receipt = trial / 'GEOMETRY_RECEIPTS.json'
    records = json.loads(receipt.read_text())
    paths = [marker_directory / f'neck_connective_y{y}.parquet' for y in (92500, 121000)]
    member = [set(map(str, pq.read_table(p, columns=['pt_root_id'])['pt_root_id'].to_pylist()))
              for p in paths]
    fragments = {}
    for root, row in records.items():
        asset, = row['assets']
        path = trial / 'geometry' / asset['filename']
        assert sha(path) == asset['sha256']
        tree = np.loadtxt(path, comments='#')
        assert np.isfinite(tree).all()
        y = tree[:, 3]
        fragments[root] = dict(
            asset=asset, nodes=len(y), swc_y_nm_range=[float(y.min()), float(y.max())],
            reached_vnc_connective_y_planes=[bool(y.min() <= plane <= y.max())
                                           for plane in (370000, 484000)],
            vnc_connective_marker_identity=[root in ids for ids in member])
    with output.open('x') as stream:
        json.dump(dict(annotation_planes_voxels=[92500, 121000], registered_planes_nm=[370000, 484000],
                       source_sha256={str(p): sha(p) for p in [receipt, *paths]},
                       status='BANC_BRAIN_SIDE_FRAGMENTS', fragments=fragments,
                       full_peripheral_continuity_verified=False),
                  stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument('trial', type=Path)
    args.add_argument('markers', type=Path)
    args.add_argument('output', type=Path)
    config = args.parse_args()
    main(config.trial, config.markers, config.output)
