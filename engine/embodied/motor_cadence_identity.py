"""Fail-closed frozen identity check for a bounded motor-cadence comparison.

This does not make historic results directly comparable after source changes;
any drift must be declared as a new experiment with a fresh 5ms baseline.
"""
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_frozen_inputs(weights, initial_pose, historical_source_hashes, entry_source):
    weights, initial_pose, historical_source_hashes, entry_source = map(
        Path, (weights, initial_pose, historical_source_hashes, entry_source))
    source = json.loads(historical_source_hashes.read_text())
    expected_weights = '28c75427a717dddc74b0f90ff7c9f4ce7f3a8eeeb6d7420263d55ddf4f7a5152'
    expected_pose = source.get(str(initial_pose.resolve()))
    expected_entry = source.get(str(entry_source.resolve()))
    checks = dict(weights_sha256=sha256(weights), initial_pose_sha256=sha256(initial_pose),
                  entry_sha256=sha256(entry_source), historical_entry_sha256=expected_entry)
    if not expected_pose or checks['initial_pose_sha256'] != expected_pose:
        raise ValueError('frozen initial pose identity mismatch')
    if checks['weights_sha256'] != expected_weights:
        raise ValueError('frozen original FIR weight identity mismatch')
    checks['historical_entry_equal'] = checks['entry_sha256'] == expected_entry
    checks['fresh_5ms_baseline_required'] = not checks['historical_entry_equal']
    return checks
