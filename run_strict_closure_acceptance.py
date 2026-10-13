"""Fail-closed scientific acceptance, distinct from interface regression PASS.
No simulation or parameter changes. CLI exits 2 for verified unmet acceptance.
"""
from pathlib import Path
import json,hashlib
from engine.embodied.closure_bundle import validate_bundle
from engine.embodied.source_snapshot import verify_source_snapshot
from engine.embodied.audit_provenance import inspect_audit, verify_artifact
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/neural_muscle_closure'

def evaluate(interface, residual_activation, standing, translation_validated):
    checks=dict(interface_invariants=bool(interface),zero_command_has_zero_activation=not residual_activation,
                demonstrated_standing=bool(standing),biological_translation_validated=bool(translation_validated))
    return dict(status='PASS' if all(checks.values()) else 'NOT_ACCEPTED',checks=checks,
                failed_checks=[key for key,value in checks.items() if not value],
                meaning='Interface PASS alone cannot satisfy scientific acceptance; failure is not a claim of general biological impossibility.')

def main():
    bundle=validate_bundle(OUT);p=Path(bundle['path']);d=json.loads((p/'closure_results.json').read_text())
    prov=d['provenance'];archive=prov['source_snapshot']
    verify_artifact(p/archive['file'],archive['sha256']);verify_source_snapshot(p/archive['file'],prov['code_hashes'])
    duration=inspect_audit(ROOT,'neural_duration_probe')
    long_path=OUT/'neural_duration_probe.json';long=json.loads(long_path.read_text())
    floor_path=OUT/'native_floor_audit.json';floor=json.loads(floor_path.read_text())
    source=Path(floor['source_bundle']['path']);validate_bundle(source)
    for name,h in floor['source_bundle']['sha256'].items():verify_artifact(source/name,h)
    verify_artifact(floor['traces']['path'],floor['traces']['sha256'])
    verify_artifact(floor['provenance']['canonical_xml'],floor['provenance']['canonical_xml_sha256'])
    # Frozen model fact, not an inference that this small effect caused falling.
    residual=floor['protocol']['all_90_activations_positive_after_every_step']
    result=evaluate(all(d['measured']['interface_acceptance'].values()),residual,
                    all(r['standing_status']=='PASS' for r in long['episodes'].values()),False)
    result.update(current_bundle=bundle,source_archive_verified=True,
        evidence=dict(duration=duration,floor_bundle=floor['source_bundle'],
          duration_json_sha256=hashlib.sha256(long_path.read_bytes()).hexdigest(),floor_json_sha256=hashlib.sha256(floor_path.read_bytes()).hexdigest()),
        validation_basis='Translation remains explicitly uncalibrated in the source model; no new external validation supplied.',
        limitations=['Historical duration evidence is not a fresh 200ms replay of current code; source drift is retained in evidence.',
                     'No claim all possible compliant models are impossible; no invented signs or stabilizer is authorized.',
                     'Native floor magnitude is not established as the collapse cause.',
                     'Hand-built research prototype, not a finished product.'])
    dest=OUT/'strict_closure_acceptance.json';dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','checks','failed_checks')},indent=2));print(dest)
    return 0 if result['status']=='PASS' else 2
if __name__=='__main__':raise SystemExit(main())
