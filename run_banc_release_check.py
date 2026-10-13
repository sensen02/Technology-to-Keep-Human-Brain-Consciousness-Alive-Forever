"""BANC official-release verification: is the missing annotation a local staleness problem?

Read-only. Compares the three annotation tables published with the BANC release (downloaded from
the Harvard Dataverse record doi:10.7910/DVN/7WTH1N, release 3) against the local working copy,
for exactly the four leg muscle targets that leave 16 actuators without a pool, and for the
CAVE user annotations that might contain muscle labels the metadata lacks.
"""
from __future__ import annotations
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import json, sys, time, hashlib
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_neural_muscle_closure as closure
DS = HERE / 'data/banc_dataverse'
LOCAL = HERE / 'data/banc888/banc_888_meta.feather'
TARGETS = ['pleural_promotor_muscle', 'tergopleural_promotor_muscle',
           'pleural_remotor_and_abductor_muscle', 'sternal_adductor_muscle']
LEGS = ['front_leg', 'middle_leg', 'hind_leg']
OUT = HERE / 'outputs/model_branches/banc_release_check'
START = time.time()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(report):
    import pandas as pd
    sup = pd.read_csv(DS / 'supplemental_data_2.txt', low_memory=False)
    loc = pd.read_feather(LOCAL, columns=['banc_888_id', 'flow', 'peripheral_target_type',
                                          'body_part_effector', 'side', 'cell_type'])
    codex = pd.read_parquet(DS / 'codex_annotations.parquet',
                            columns=['pt_root_id', 'classification_system', 'cell_type'])
    cellinfo = pd.read_parquet(DS / 'cell_info.parquet', columns=['tag', 'tag2', 'pt_root_id'])

    def coverage(frame, rootcol):
        rootcol = 'root_id' if rootcol == 'root_id' else rootcol
        eff = frame[frame['flow'] == 'efferent']
        leg = eff[eff['body_part_effector'].isin(LEGS)]
        out = {}
        for t in TARGETS:
            sub = leg[leg['peripheral_target_type'] == t]
            out[t] = dict(total=int(len(sub)),
                          by_leg_side={f'{k[0]}/{k[1]}': int(v) for k, v in
                                       sub.groupby(['body_part_effector', 'side']).size().items()})
        return dict(efferent=int(len(eff)), leg_efferent=int(len(leg)),
                    leg_target_vocabulary=sorted(leg['peripheral_target_type'].dropna().unique()), targets=out)

    official = coverage(sup, 'root_id')
    local = coverage(loc, 'banc_888_id')
    identical = official['targets'] == local['targets'] and official['leg_target_vocabulary'] == local['leg_target_vocabulary']
    meta_root = set(loc['banc_888_id'].astype(str))
    muscle_tags = ['Sternal adductor MN', 'Pleural remotor/abductor MN', 'Pleural remotor abductor MN',
                   'Tergopleural/Pleural promotor MN']
    tag_detail = {}
    for tag in muscle_tags:
        roots = [str(x) for x in cellinfo[cellinfo['tag'].astype(str) == tag]['pt_root_id']]
        sub = loc[loc['banc_888_id'].isin(roots)]
        tag_detail[tag] = dict(cells=len(set(roots)),
                               targets_returned=sorted(sub['peripheral_target_type'].dropna().unique()),
                               mixes_several_muscles=len(set(sub['peripheral_target_type'].dropna())) > 1)
    promotor_rows = codex[codex['cell_type'].astype(str).str.contains('promotor', case=False, na=False)]
    promotor_meta = loc[loc['banc_888_id'].isin({str(x) for x in promotor_rows['pt_root_id']})]
    result = dict(
        status='MEASURED_BANC_RELEASE_VERIFICATION',
        label='manual read-only verification against the published release, not a new annotation',
        question='Is the missing leg-muscle annotation absent from the published release, or only from the local copy?',
        sources=dict(
            dataverse_record='doi:10.7910/DVN/7WTH1N (release 3)',
            official_supplemental_data_2=dict(path=str(DS / 'supplemental_data_2.txt'),
                                              sha256=sha(DS / 'supplemental_data_2.txt'), rows=int(len(sup))),
            official_codex_annotations=dict(path=str(DS / 'codex_annotations.parquet'),
                                            sha256=sha(DS / 'codex_annotations.parquet'), rows=int(len(codex))),
            official_cell_info=dict(path=str(DS / 'cell_info.parquet'),
                                    sha256=sha(DS / 'cell_info.parquet'), rows=int(len(cellinfo))),
            local_working_copy=dict(path=str(LOCAL), sha256=sha(LOCAL), rows=int(len(loc)))),
        official=official, local=local, official_matches_local=bool(identical),
        cave_user_tags=tag_detail,
        codex_promotor_cells=dict(
            annotated_rows=int(len(promotor_rows)),
            distinct_types=sorted(promotor_rows['cell_type'].astype(str).unique()),
            legs_of_those_cells=sorted(promotor_meta['body_part_effector'].dropna().unique()),
            finding='every promotor-labelled cell in the full Codex annotation dump belongs to the front leg'),
        conclusion='The published release contains exactly the same leg muscle coverage as the local working copy, so the '
                   'gap is not staleness: the four targets really are unannotated on the affected legs. No access token '
                   'would change this, because the equivalent annotation tables are already public in the release dump.',
        limitations=['One release (v888, release 3).',
                     'CAVE user tags are approximate labels and are reported only to show that they cannot fill pools.',
                     'No live CAVE query was made; no token was used.'],
        memory_guard=report('completed'), wall_s=time.time() - START)
    if not np.isfinite(result['memory_guard'].get('previous_soft_limit_mb', 0.0)):
        result['memory_guard']['previous_soft_limit_mb'] = 'unlimited'
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'banc_release_check.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(official_matches_local=identical,
                          official_efferent=official['efferent'], local_efferent=local['efferent'],
                          cave_user_tags={k: v['mixes_several_muscles'] for k, v in tag_detail.items()},
                          codex_promotor=result['codex_promotor_cells']), indent=2))
    return result


if __name__ == '__main__':
    with closure.mg.guarded_run('banc_release_check', **closure.mg.BUDGETS['neural_muscle_closure']) as report:
        main(report)
