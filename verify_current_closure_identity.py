"""Bounded regression verifier: compare current complete run to explicit reference."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from engine.embodied.closure_bundle import validate_bundle
from test_closure_probe_records import verify
HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs/neural_muscle_closure'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reference',required=True);args=parser.parse_args()
    current=validate_bundle(OUT);reference=validate_bundle(Path(args.reference));p=Path(current['path']);old=Path(reference['path'])
    d=json.loads((p/'closure_results.json').read_text());m=json.loads((p/'motor_manifest.json').read_text())
    assert d['provenance']==m['provenance']
    from engine.embodied.source_snapshot import verify_source_snapshot
    archive=d['provenance']['source_snapshot']
    assert hashlib.sha256((p/archive['file']).read_bytes()).hexdigest()==archive['sha256']
    verify_source_snapshot(p/archive['file'],d['provenance']['code_hashes'])
    checks={}
    with np.load(p/'closure_traces.npz',allow_pickle=False) as z, np.load(old/'closure_traces.npz',allow_pickle=False) as a:
        assert set(z.files)==set(a.files)
        for key in a.files:
            np.testing.assert_array_equal(z[key],a[key]);checks[key]=True
        for arm,summary in d['measured']['episodes'].items():verify(summary,z[arm+'_spike_counts'],require_raw_equals_emitted=arm!='spike_output_silence')
    result=dict(status='EXACT_ARRAY_AND_PROBE_REGRESSION',current_bundle=current,reference_bundle=reference,all_arrays_identical=all(checks.values()),array_checks=checks,probe_counts_reconciled=True,limitation='Identical reference behavior still falls; interface validation is not autonomous success')
    dest=OUT/'current_closure_identity.json';dest.write_text(json.dumps(result,indent=2)+'\n');print(dest);print(current['path'])
if __name__=='__main__':main()
