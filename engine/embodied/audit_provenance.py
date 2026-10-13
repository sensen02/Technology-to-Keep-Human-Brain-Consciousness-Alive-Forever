"""Read-only historical evidence checks, separate from live code freshness.

Digests are consistency anchors, not signatures or proof of execution. These
legacy audits did not archive source bytes. A well-formed recorded source hash
cannot authenticate unavailable originals; even matched live bytes do not prove
that a historical execution used them. No replay is performed here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from .closure_bundle import validate_bundle

AUDITS = ('touch_path_audit', 'motor_state_audit', 'connected_feedback_audit',
          'early_feedback_probe', 'neural_pose_comparison', 'neural_duration_probe')


def captured_hash(value):
    """Require a canonical captured SHA-256; this does NOT verify source bytes."""
    if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise ValueError(f'Invalid captured SHA-256: {value!r}')
    return value


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact(path, expected):
    """Hard-fail changed/missing artifacts and immutable model/data inputs."""
    if sha256(path) != captured_hash(expected):
        raise ValueError(f'Artifact SHA-256 mismatch: {path}')


def live_source_freshness(hashes, *, strict=False):
    """Compare live bytes only; missing/drift never invalidates saved evidence.

    strict=True is for callers explicitly requiring *current* source identity,
    not for historical artifact acceptance. It fails on drift AND missing files.
    """
    rows = {}
    for path, expected in hashes.items():
        captured_hash(expected)
        try:
            current = sha256(path)
        except FileNotFoundError:
            current = None
        status = 'missing' if current is None else ('matched' if current == expected else 'drift')
        rows[str(path)] = dict(captured_sha256=expected, current_sha256=current,
                              status=status, current_match=status == 'matched')
    if strict and any(row['status'] != 'matched' for row in rows.values()):
        raise ValueError('Current source freshness required: drift or missing source')
    return rows


def validate_saved_bundle(reference):
    """Validate the exact saved run, never follow the moving current index.

    Older records embed a full manifest rather than its raw-byte digest. In that
    case semantic manifest equality is the retained anchor (not a fabricated SHA).
    """
    path = Path(reference['path'])
    if path.name != reference['run_id'] or path.parent.name != 'runs':
        raise ValueError('Expected exact saved runs/<run_id> path')
    digest = reference.get('manifest_sha256')
    if digest is None and 'manifest' not in reference:
        raise ValueError('Saved bundle has no manifest anchor')
    if digest is not None:
        captured_hash(digest)
    bundle = validate_bundle(path, manifest_sha256=digest)
    if bundle['run_id'] != reference['run_id']:
        raise ValueError('Saved bundle run identity mismatch')
    if 'manifest' in reference and bundle['manifest'] != reference['manifest']:
        raise ValueError('Saved manifest mismatch')
    if 'files' in reference and bundle['manifest']['files'] != reference['files']:
        raise ValueError('Saved manifest files mismatch')
    for entry in bundle['manifest']['files'].values():
        captured_hash(entry['sha256'])
    return bundle


def inspect_audit(root, name, *, strict=False):
    """Validate retained evidence and return separately labelled freshness rows."""
    root = Path(root)
    out = root / 'outputs/neural_muscle_closure'
    if name not in AUDITS:
        raise ValueError(f'Unsupported audit: {name}')
    doc = json.loads((out / (name + '.json')).read_text())
    p = doc.get('provenance', {})
    reference = doc.get('source_bundle')
    anchor_origin = 'audit source_bundle'
    retained_anchor = None
    if name == 'neural_duration_probe':
        anchor_path = out / 'history/63ab7236_identity.json'
        retained_anchor = dict(path=str(anchor_path), sha256=sha256(anchor_path))
        reference = json.loads(anchor_path.read_text())['current_bundle']
        if reference['path'] != p['current_bundle']:
            raise ValueError('Duration/identity historical bundle path mismatch')
        anchor_origin = 'history/63ab7236_identity.json:current_bundle (retained exact duration path)'
        source_rows = p['frozen_source_hashes']
        own_hash = source_rows[str(root / 'run_neural_duration_probe.py')]
        verify_artifact(out / (name + '.npz'), p['npz_sha256'])
        verify_artifact(p['numerical_prefix_artifact'], p['numerical_prefix_sha256'])
    elif name == 'neural_pose_comparison':
        # Pose retained only prior NPZ digest; the companion probe retains the
        # full manifest for that SAME path. Never substitute the current bundle.
        reference = json.loads((out / 'early_feedback_probe.json').read_text())['source_bundle']
        if reference['path'] != p['prior_bundle']:
            raise ValueError('Pose/probe historical bundle path mismatch')
        anchor_origin = 'early_feedback_probe.source_bundle (same exact prior path)'
        verify_artifact(Path(p['prior_bundle']) / 'closure_traces.npz', p['prior_npz_sha256'])
        verify_artifact(p['repaired_source'], p['repaired_source_sha256'])
        verify_artifact(out / (name + '.npz'), p['npz_sha256'])
        own_hash = p['runner_sha256']
        source_rows = p['frozen_source_hashes']
    elif name == 'touch_path_audit':
        own_hash = doc['audit_source_sha256']
        source_rows = doc['source_checksums']
        verify_artifact(doc['spike_npz'], doc['spike_npz_sha256'])
    else:
        own_hash = p['code_sha256']
        source_rows = p['source_code_hashes']
        verify_artifact(doc['traces']['path'], doc['traces']['sha256'])
    bundle = validate_saved_bundle(reference)
    original = json.loads((Path(bundle['path']) / 'closure_results.json').read_text())['provenance']
    run_hashes = original['code_hashes']
    audit_hashes = {}
    immutable = dict(p.get('inputs', {}))
    for path, row in source_rows.items():
        if isinstance(row, dict):
            recorded = row.get('saved_sha256', row.get('recorded_sha256'))
            captured_hash(recorded)
            # Recorded run-source claims must agree with the anchored run.
            if path in run_hashes and run_hashes[path] != recorded:
                raise ValueError(f'Captured run source hash disagrees with bundle: {path}')
            value = row['current_sha256']  # current AT CAPTURE, not current now
        else:
            value = row
        captured_hash(value)
        if Path(path).suffix == '.py':
            audit_hashes[path] = value
        else:
            immutable[path] = value
            if isinstance(row, dict) and recorded != value:
                raise ValueError(f'Immutable input changed at capture: {path}')
    for path, value in run_hashes.items():
        captured_hash(value)
        if Path(path).suffix != '.py':
            verify_artifact(path, value)
    # XML and data remain immutable even when source freshness is stale.
    input_fields = {
        'canonical_xml_sha256': 'outputs/muscles_six_legs/fruitfly_six_leg_standing.xml',
        'pool_result_sha256': 'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json',
        'sensory_lookup_sha256': 'outputs/neural_rebuild/sensory_channel_lookup.json',
        'connectome_sha256': 'data/flywire/banc_connectome.npz',
        'source_feather_sha256': 'data/banc888/banc_888_meta.feather',
    }
    for field, relative in input_fields.items():
        if field in original:
            verify_artifact(root / relative, original[field])
        if field in p:
            verify_artifact(root / relative, p[field])
    for path, value in immutable.items():
        verify_artifact(path, value)
    own_path = p.get('code', str(root / ('run_' + name + '.py')))
    own = live_source_freshness({own_path: own_hash}, strict=strict)
    audit_sources = live_source_freshness(audit_hashes, strict=strict)
    run_sources = live_source_freshness({k: v for k, v in run_hashes.items()
                                         if Path(k).suffix == '.py'}, strict=strict)
    rows = list(own.values()) + list(audit_sources.values()) + list(run_sources.values())
    return dict(audit=name, source_bundle=bundle, manifest_anchor=anchor_origin,
                retained_anchor=retained_anchor,
                historical_artifact_integrity='validated', immutable_inputs='matched',
                source_snapshot='unavailable', original_source_authenticated=False,
                execution_equivalence_claimed=False,
                limitation='Original source bytes were not archived; captured hashes are not authentication of originals or proof of execution.',
                own_source=own, audit_time_sources=audit_sources, run_sources=run_sources,
                live_source_status=('missing' if any(r['status'] == 'missing' for r in rows)
                                    else 'drift' if any(r['status'] == 'drift' for r in rows) else 'matched'))


def freshness_report(root):
    return dict(schema_version=1, replay_performed=False,
                interpretation='Historical artifact consistency and live source freshness are separate. No execution equivalence is asserted.',
                audits={name: inspect_audit(root, name) for name in AUDITS})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    # Emit only; caller chooses a new report destination. Never edit old evidence.
    print(json.dumps(freshness_report(args.root), indent=2, sort_keys=True))
