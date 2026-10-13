"""Campaniform channel composition audit.

Read-only. Cross-tabulates, for the campaniform cells the published run actually uses:
  (a) their annotation cell type,
  (b) whether they carry a Verified transmitter label,
  (c) whether an external ontology route gives them a leg-segment identity
      (cell type SNpp53 -> trochanter campaniform sensillum neuron, per the research ledger),
  (d) cell types whose BANC campaniform label is contradicted elsewhere.
No label is filled, no segment is inferred, and no channel semantics are changed.
"""
from __future__ import annotations
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import json, sys, time, hashlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_neural_muscle_closure as closure
FEATHER = HERE / 'data/banc888/banc_888_meta.feather'
FRONT = HERE / 'outputs/model_branches/sensory_frontend_balance/sensory_frontend_balance_traces.npz'
RESEARCH = HERE / 'outputs/neural_muscle_closure/research_contact_identity_sources.json'
OUT = HERE / 'outputs/model_branches/campaniform_composition'
# From the research ledger: SNpp53 maps to FBbt:00048059 (trochanter campaniform sensillum neuron).
SEGMENT_IDENTIFIED_TYPES = {'SNpp53': 'trochanter campaniform sensillum neuron (FBbt:00048059)'}
# Cell types whose BANC campaniform class label is contradicted by external typing.
CONFLICTING_TYPES = {'SNpp39': 'typed elsewhere as femoral chordotonal club neuron (FBbt:00049556)',
                     'SNpp45': 'typed elsewhere as adult SNta23 (FBbt:20011274)',
                     'SNpp43': 'typed elsewhere as adult SNxx33 (FBbt:20011329)',
                     'SNxx30': 'typed elsewhere as uncharacterised adult SNxx30 (FBbt:20011326)'}
START = time.time()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(report):
    import pandas as pd
    with np.load(FRONT) as z:
        roots = [str(r) for r, c in zip(z['cell_root_id'], z['cell_channel']) if str(c) == 'campaniform_strain']
    f = pd.read_feather(FEATHER, columns=['banc_888_id', 'cell_type', 'cell_sub_class',
                                          'neurotransmitter_verified', 'side', 'nerve'])
    f['banc_888_id'] = f['banc_888_id'].astype(str)
    sub = f[f['banc_888_id'].isin(roots)].copy()
    assert len(sub) == len(roots) == 60, (len(sub), len(roots))

    def verified(v):
        return bool(v is not None and not (isinstance(v, float) and np.isnan(v)) and str(v).strip())

    sub['verified'] = [verified(v) for v in sub['neurotransmitter_verified']]
    by_type = {}
    for ctype, group in sub.groupby(sub['cell_type'].astype(str)):
        by_type[ctype] = dict(cells=int(len(group)),
                              verified=int(group['verified'].sum()),
                              segment_identified=ctype in SEGMENT_IDENTIFIED_TYPES,
                              conflicting_label=ctype in CONFLICTING_TYPES)
    identified = sub[sub['cell_type'].astype(str).isin(SEGMENT_IDENTIFIED_TYPES)]
    conflicting = sub[sub['cell_type'].astype(str).isin(CONFLICTING_TYPES)]
    result = dict(
        status='MEASURED_CAMPANIFORM_CHANNEL_COMPOSITION',
        label='manual read-only composition audit, not a sensory model and not a mapping',
        question='Within the campaniform channel the run actually uses, do transmitter evidence and cell identity overlap?',
        scope=dict(tier_campaniform_cells=len(roots), by_cell_type=by_type,
                   verified_cells=int(sub['verified'].sum()),
                   verified_types=sorted(sub[sub['verified']]['cell_type'].astype(str).unique())),
        segment_identified=dict(
            rule='cell type SNpp53 only; the segment is the trochanter and no named sensillum is assigned',
            cells=int(len(identified)), verified=int(identified['verified'].sum()),
            cell_sub_classes=sorted(identified['cell_sub_class'].astype(str).unique()),
            nerves=sorted(identified['nerve'].astype(str).unique())),
        label_conflicts=dict(
            rule='BANC campaniform class label contradicted by external typing', cells=int(len(conflicting)),
            verified=int(conflicting['verified'].sum()),
            types={k: CONFLICTING_TYPES[k] for k in sorted(set(conflicting['cell_type'].astype(str)))},
            note='these rows are still counted as campaniform by the deployed channel'),
        overlap=dict(
            segment_identified_and_verified=int(identified['verified'].sum()),
            verified_without_segment_identity=int(sub['verified'].sum()) - int(identified['verified'].sum())),
        conclusion='Within this channel the two kinds of evidence do not overlap: the 11 cells that can be given a '
                   'segment (trochanter) have NO Verified transmitter, while the 12 cells that do have a Verified label '
                   'are typed only as generic CS/bCS or as types contradicted elsewhere. The channel therefore cannot be '
                   'relabelled "trochanter campaniform" nor "load or contact sensor".',
        not_licensed=['attaching a named sensillum to any row',
                      'calling any row femoral, tibial or tarsal',
                      'claiming a load/force encoder (the only Drosophila computation on record is negative)',
                      'claiming a foot-contact channel (no source measures one)',
                      'filling any transmitter label'],
        provenance=dict(feather=str(FEATHER), feather_sha256=sha(FEATHER),
                        frontend_trace=str(FRONT), frontend_trace_sha256=sha(FRONT),
                        research_ledger=str(RESEARCH), research_ledger_sha256=sha(RESEARCH) if RESEARCH.exists() else None,
                        canonical_xml_sha256=sha(closure.cp.CANONICAL_XML)),
        limitations=['One annotation release; the external typing comes from the research ledger, not re-queried here.',
                     'Generic labels such as CS and bCS carry no segment information.',
                     'No channel semantics, gain or label was changed by this audit.'],
        memory_guard=report('completed'), wall_s=time.time() - START)
    if not np.isfinite(result['memory_guard'].get('previous_soft_limit_mb', 0.0)):
        result['memory_guard']['previous_soft_limit_mb'] = 'unlimited'
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'campaniform_composition.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(scope=result['scope'], segment_identified=result['segment_identified'],
                          label_conflicts=dict(cells=result['label_conflicts']['cells'],
                                               verified=result['label_conflicts']['verified'],
                                               types=result['label_conflicts']['types']),
                          overlap=result['overlap']), indent=2))
    return result


if __name__ == '__main__':
    with closure.mg.guarded_run('campaniform_composition', **closure.mg.BUDGETS['neural_muscle_closure']) as report:
        main(report)
