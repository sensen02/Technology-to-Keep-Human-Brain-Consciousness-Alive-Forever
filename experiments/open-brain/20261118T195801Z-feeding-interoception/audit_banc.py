"""Record BANC enteric identities for the published FAFB StN candidates."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import json

import pyarrow.feather as feather


def audit(metadata, binding, output):
    groups = json.loads(binding.read_text())['groups']
    published = {row['flywire_root']: group for group, rows in groups.items() for row in rows}
    columns = ['banc_888_id', 'fafb_match', 'cell_type', 'nerve', 'region', 'super_class',
               'tract', 'body_part_sensory', 'neurotransmitter_verified', 'neurotransmitter_predicted']
    table = feather.read_table(metadata, columns=columns)
    rows = [row for row in table.to_pylist()
            if (row['fafb_match'] in published or 'stomodeal_nerve' in (row['nerve'] or ''))]
    matched = [row for row in rows if row['fafb_match'] in published]
    counts = Counter(row['fafb_match'] for row in matched)
    for row in rows:
        root = row['fafb_match']
        row['paper_group'] = published.get(root)
        if root in counts:
            row['candidate_multiplicity'] = counts[root]
    corroborated = [row for row in matched if 'stomodeal_nerve' in (row['nerve'] or '')
                    and row['body_part_sensory'] == 'anterior_digestive_tract']
    assert corroborated and all(row['super_class'] == 'sensory' for row in corroborated)
    report = dict(
        status='BANC_ENTERIC_CANDIDATES_ONLY', molecular_identity='unresolved',
        entire_metadata_stn_annotated=len([r for r in rows if 'stomodeal_nerve' in (r['nerve'] or '')]),
        paper_root_count=len(published),
        fafb_roots_with_banc_candidates=len(counts),
        banc_rows_with_fafb_candidates=len(matched),
        roots_with_unique_banc_candidate=sum(n == 1 for n in counts.values()),
        corroborated_banc_rows=len(corroborated),
        corroborated_type_counts=dict(Counter(row['cell_type'] for row in corroborated)),
        rows=rows,
        source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in (metadata, binding, Path(__file__).resolve())})
    with output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('banc_metadata', type=Path)
    parser.add_argument('binding', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    audit(args.banc_metadata, args.binding, args.output)
