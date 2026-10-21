"""Independent source-array and global split verification, no fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from tools_s4_calcium_paired_contract import validate


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify(bundle):
    bundle=Path(bundle);summary=json.loads((bundle/'summary.json').read_text())
    ledger=json.loads((bundle/'unique_identity_manifest.json').read_text())
    sources={};checks={};subjects={};recordings={};primary_seen=set();unique_rows=0
    for name,info in summary['files'].items():
        from urllib.parse import urlparse,unquote
        p=Path(unquote(urlparse(info['source']['uri']).path));checks[name+'/sha']=sha(p)==info['source']['sha256']
        sources[name]=pq.read_table(p).to_pylist()
    byid={g['group_id']:g for g in ledger['groups']}
    checks['unique_ledger']=len(byid)==len(ledger['groups'])==116
    for g in ledger['groups']:
        rows=[sources[g['source_file']][i] for i in g['source_row_indices']]
        unique_rows+=len(rows)
        checks[g['group_id']+'/rows']=len(rows)==g['samples'] and all(int(r['animal_id'])==g['animal_id'] and r['trial']==g['trial'] and r['roi']==g['roi'] for r in rows)
        for registry,key in ((subjects,g['subject_id']),(recordings,g['raw_recording_id'])):
            if key in registry:checks[key+'/split']=registry[key]==g['split']
            registry[key]=g['split']
        expected='holdout' if g['animal_id'] in ([9,10] if g['source_file'].startswith('claw') else [13,14]) else 'train'
        checks[g['group_id']+'/holdout']=g['split']==expected
    for p in ledger['packages']:
        path=bundle/p['package'];d=json.loads(path.read_text());saved=json.loads((bundle/p['validation']).read_text())
        checks[p['package']+'/sha']=sha(path)==p['sha256']
        current=validate(d);checks[p['package']+'/validation']=current==saved and current['contract_valid'] and not current['calibration_ready']
        checks[p['package']+'/primary_unique']=p['primary_group_id'] not in primary_seen;primary_seen.add(p['primary_group_id'])
        for trial,identity in zip(d['trials'],(p['primary_group_id'],p['anchor_group_id']),strict=True):
            g=byid[identity];rows=[sources[g['source_file']][i] for i in g['source_row_indices']]
            checks[p['package']+'/'+identity+'/preserved']=trial['time_s']==[r['time'] for r in rows] and trial['fluorescence']==[r['calcium'] for r in rows] and np.array_equal(trial['angle_rad'],np.deg2rad([r['L1C_flex'] for r in rows])) and trial['split']==g['split'] and trial['subject_id']==g['subject_id'] and trial['identity_level']=='roi_population' and all(v=='unknown' for v in trial['sync'].values())
    checks['complete_primary']=primary_seen==set(byid)
    checks['unique_rows']=unique_rows==52092
    checks['live_provenance']=all(sha(p)==h for p,h in summary['provenance_sha256'].items())
    return dict(pass_all=all(checks.values()),checks=checks,packages=len(ledger['packages']),unique_rows=unique_rows,unique_groups=len(byid),biological_validation=False,scope='independent input/array conversion/subject split qualification, not fitting')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('bundle',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=verify(a.bundle)
    with a.output.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print('checks',len(r['checks']),'pass',r['pass_all']);raise SystemExit(0 if r['pass_all'] else 1)


if __name__=='__main__':main()
