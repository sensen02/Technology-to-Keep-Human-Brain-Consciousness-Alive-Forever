"""Audit publication v888 DNg97 geometry and direct VNC output."""
import base64
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import pyarrow.parquet as parquet

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
VISUAL = HERE.parent / '20261116T023521Z'
NECK = PROJECT / 'experiments/neck-real-structure'
GEOM = NECK / '20261112T214803Z'
DOWNLOADS = GEOM / 'geometry_downloads'
sys.path.insert(0, str(GEOM/'geometry_sources'))
from validate_swc import parse_swc, sha, validate

ROOT = {
    'left': ('720575941519464046', 'l2', '1777944761612880', 86488, 'FAAZyWDzvlsvMPic5AXQUw=='),
    'right': ('720575941572713886', 'skeleton', '1778894375558608', 4417676, '32UQtMFatm3gNEdKJDjFsw=='),
}


def ref(path):
    return dict(path=str(path.resolve()), sha256=sha(path))


def independent_crossings(swc, geometry, points):
    nodes, _, _ = parse_swc(swc)
    front, back = [geometry['plane_intersections'][str(plane)] for plane in geometry['plane_y_nm']]
    by_front = defaultdict(list)
    for posterior in back:
        cursor = posterior['child']
        ancestry = set()
        while cursor != -1:
            ancestry.add(cursor)
            cursor = nodes[cursor][3]
        for candidate in front:
            if candidate['child'] in ancestry:
                by_front[candidate['child']].append(posterior)
    ranked = []
    for candidate in front:
        descendants = by_front[candidate['child']]
        if not descendants:
            continue
        distance = min(math.dist(candidate['point_nm'], [x*s for x,s in zip(point, [4,4,45],strict=True)])
                       for point in points)
        ranked.append((distance, candidate['child'], candidate, descendants))
    assert ranked, 'No same-branch continuation to posterior cervical plane'
    distance, _, anterior, posterior = min(ranked)
    return dict(front=anterior, distance_from_front_annotation_nm=distance, posterior=posterior)


def main():
    metadata = PROJECT/'data/banc888/banc_888_meta.feather'
    old_meta = NECK/'20261112T225331Z/dnp09_geometry/720575941433155799_METADATA.json'
    columns = ['root_id','root_888','banc_888_id','cell_type','side','region','super_class','tract','proofread']
    table = feather.read_table(metadata, columns=columns)
    by_root = defaultdict(list)
    for row in table.to_pylist():
        by_root[row['root_888']].append(row)
    points_file = DOWNLOADS/'neck_connective_y92500.parquet'
    points = parquet.read_table(points_file)
    points = points.filter(pc.equal(points['valid'],True)).to_pylist()
    source = VISUAL/'banc_888_edgelist_simple_v2.feather'
    edge_headers = VISUAL/'edges_v2.download.headers'
    assert 'x-goog-generation: 1780396134870867' in edge_headers.read_text()
    with source.open('rb') as data:
        assert base64.b64encode(hashlib.file_digest(data,'md5').digest()).decode() == 'OUQG+Km98JPIlfla/09sSQ=='
    roots = [record[0] for record in ROOT.values()]
    outgoing = {root:{} for root in roots}
    with pa.memory_map(str(source),'r') as data:
        reader = ipc.open_file(data)
        for index in range(reader.num_record_batches):
            batch = reader.get_batch(index)
            selected = batch.filter(pc.is_in(batch.column('pre'), value_set=pa.array(roots)))
            for edge in selected.select(['pre','post','count']).to_pylist():
                targets = outgoing[edge['pre']]
                assert edge['post'] not in targets and edge['count'] > 0
                targets[edge['post']] = edge['count']
    all_targets = set().union(*(record.keys() for record in outgoing.values()))
    targets = {rid:by_root[rid][0] for rid in sorted(all_targets) if len(by_root[rid]) == 1
               and by_root[rid][0]['region'] == 'ventral_nerve_cord'}
    common = [ref(path) for path in (metadata,old_meta,points_file,DOWNLOADS/'neck_connective_y121000.md',
                                   DOWNLOADS/'banc_swc_skeletons.md',source,edge_headers)]
    ports = []
    geometries = {}
    for side, (root,resolution,generation,length,expected_md5) in ROOT.items():
        row, = by_root[root]
        assert row['side']==side and row['cell_type']=='DNg97' and row['super_class']=='descending'
        assert row['tract']=='cervical_connective' and row['proofread']=='TRUE'
        assert row['banc_888_id']==root
        swc = HERE/f'{root}_{resolution}.swc'
        headers = HERE/f'{root}.download.headers'
        assert f'x-goog-generation: {generation}' in headers.read_text()
        assert swc.stat().st_size==length
        assert base64.b64encode(hashlib.md5(swc.read_bytes()).digest()).decode()==expected_md5
        matched = [point['pt_position'] for point in points if str(point['pt_root_id'])==root]
        assert matched
        geometry = validate(swc,dict(schema='BANC_V888_SWC_GEOMETRY_CHECK_V1',dataset='BANC',
            materialization=888,root_id=root,coordinate_space='BANC_NATIVE',coordinate_units='nm',
            source_evidence=common+[ref(headers)],swc_sha256=sha(swc),
            version_identity_evidence=dict(root_888=root,verified=True,source_row=row),
            planes=json.loads(old_meta.read_text())['planes']))
        assert geometry['is_single_tree'] and geometry['each_plane_intersected']
        branch = independent_crossings(swc, geometry, matched)
        # L2 is a deliberately lower-resolution skeleton. Use a wide but explicit
        # proximity guard without pretending that it has full-SWC precision.
        assert branch['distance_from_front_annotation_nm'] < (10000 if resolution=='l2' else 3000)
        geometry.update(crossing=branch, annotated_front_plane_voxels=matched,
                        source_swc_resolution=resolution,gcs_generation=generation,md5_base64=expected_md5)
        geometries[root] = geometry
        assert targets.keys() & outgoing[root].keys(), 'No exact-joined VNC targets'
        path = HERE/f'{root}_GEOMETRY.json'
        path.write_text(json.dumps(geometry,indent=2)+'\n')
        ports.append(dict(root_id=root, direction='DN',side=side,source_reviewed=True,
                          structural_evidence_status='SOURCE_GEOMETRY_REVIEWED',source_cell_type='DNg97',
                          native_tuning_established=False,evidence=[ref(path),ref(swc),ref(headers)]))
    edges = [dict(pre=root,post=target,synapses=outgoing[root][target]) for root in roots
             for target in targets if target in outgoing[root]]
    graph = HERE/'OFFICIAL_GRAPH.json'
    graph.write_text(json.dumps(dict(roots=roots, targets=list(targets.values()),connections=edges,
        sources=common,release='BANC v888 v2',publication_generation='1780396134870867',
        unjoined_post_root_ids=sorted(rid for rid in all_targets if len(by_root[rid])!=1),
        engineering_projection=True,native_tuning_established=False),indent=2)+'\n')
    prior = VISUAL/'DNa02_BINDING.json'
    binding = json.loads(prior.read_text())
    binding.update(status='SCOPED_DNG97_FORWARD_TASK_SOURCE_BINDING',ports=ports+[
        port for port in binding['ports'] if port['direction']=='AN'],
        source_files=common+[ref(graph),ref(HERE/'audit_source.py'),ref(prior)],
        an_runtime_scope='Archived port retained to match transport schema, inactive.')
    (HERE/'DNg97_BINDING.json').write_text(json.dumps(binding,indent=2)+'\n')
    for root in roots:
        mine=[edge for edge in edges if edge['pre']==root]
        print(root,geometries[root]['nodes'],geometries[root]['crossing']['distance_from_front_annotation_nm'],
              len(mine),sum(edge['synapses'] for edge in mine),flush=True)


if __name__=='__main__':
    main()
