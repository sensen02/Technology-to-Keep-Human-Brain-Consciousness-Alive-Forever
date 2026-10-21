"""Assess declared sensory calcium metadata without inventing spikes or EM identity."""
import argparse
import json
from pathlib import Path
from tools_s4_paired_data_contract import load


def qualify(data):
    errors, gaps = [], []
    if not isinstance(data, dict):
        return dict(structural_valid=False, calibration_ready=False, biological_validation=False,
                    errors=['object required'], gaps=[])
    for key in ('asset_uri', 'asset_revision', 'roi_identity', 'reporter', 'fluorescence_metric',
                'baseline_method', 'angle_convention', 'time_unit', 'sync_evidence', 'side_evidence'):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(key+': nonempty declaration or explicit unknown required')
        elif value.strip().lower() == 'unknown':
            gaps.append(key+': unknown')
    if data.get('modality') != 'calcium':
        errors.append('modality must be calcium, never spike')
    if data.get('identity_level') not in ('roi_population', 'single_cell', 'unknown'):
        errors.append('identity_level must be explicit')
    if data.get('identity_level') != 'single_cell':
        gaps.append('single-cell identity not established')
    if data.get('response_origin') not in ('measured', 'predicted', 'unknown'):
        errors.append('response_origin must be explicit')
    if data.get('response_origin') != 'measured':
        gaps.append('response is not declared measured')
    if type(data.get('asset_bytes_verified')) is not bool:
        errors.append('asset_bytes_verified must be bool')
    if data.get('asset_bytes_verified') is not True:
        gaps.append('asset bytes/schema not inspected')
    # No root ID is required: population ROI is not an EM neuron. An optional
    # mapping must have exact identity and independently reviewed evidence.
    mapping = data.get('banc_mapping')
    if mapping is not None:
        if not isinstance(mapping, dict):
            errors.append('banc_mapping must be null or object')
        else:
            root = mapping.get('root_id')
            if not isinstance(root, str) or not root.isdecimal() or root.startswith('0'):
                errors.append('mapping root_id must be an exact positive decimal string')
            if not isinstance(mapping.get('evidence'), str) or not mapping['evidence'].strip():
                errors.append('mapping evidence required')
    return dict(structural_valid=not errors, calibration_ready=False, biological_validation=False,
                errors=errors, gaps=gaps,
                scope='metadata qualification only; declarations not independently scientifically verified',
                limitations=['No fluorescence-to-spike inference', 'No BANC mapping inferred from subtype/ROI',
                             'No binary schema inspected by this tool', 'No fit or biological holdout validation'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    data,digest=load(a.manifest);report=qualify(data)
    report.update(manifest=str(a.manifest.resolve()),manifest_sha256=digest)
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print('STRUCTURAL_PASS_WITH_GAPS' if report['structural_valid'] else 'REJECTED')
    raise SystemExit(0 if report['structural_valid'] else 1)


if __name__=='__main__':main()
