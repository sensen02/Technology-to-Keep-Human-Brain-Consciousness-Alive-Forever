"""Fit a frozen spike-only interface from an explicitly labeled NPZ dataset.

NPZ: spikes[T,n], targets[B,nu], optional connected[B], metadata (scalar JSON
string). command_ticks=10 in metadata gives B=T/10, matching 5ms closure;
explicit target_alignment=next_interval_after_completed_source_block and arrays
source_start_s/source_end_s/target_start_s/target_end_s[B] are required.
BEHAVIOR support_<field> raw records are preflighted, but behavioral fitting
remains disabled until a separate target provenance contract is established.
Absent command_ticks retains the legacy one-tick interface-test mode.
Metadata must set target_kind=INTERFACE_TEST; labels are synthetic interface
 tests, never evidence of behavior or biological calibration.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from engine.embodied.motor_training import fit
from engine.embodied.motor_data_qualification import qualify_training_dataset


def publish_artifacts(translator, receipt, output, receipt_path):
    """Serialize both before no-clobber publication; roll back caught failures.

    Reuses runtime.save unchanged. Same-filesystem hard links publish complete
    files exclusively (including rejecting dangling symlinks). Two paths are
    not a crash-atomic transaction; process kill/power loss between links is
    outside this rollback guarantee.
    """
    output, receipt_path = Path(output), Path(receipt_path)
    if output == receipt_path or output.parent != receipt_path.parent:
        raise ValueError('artifact pair requires distinct paths in the same directory')
    with tempfile.TemporaryDirectory(prefix='.motor-training-', dir=output.parent) as directory:
        staged_model = Path(directory)/'model.json'
        staged_receipt = Path(directory)/'receipt.json'
        translator.save(staged_model)
        model_bytes = staged_model.read_bytes()
        def reject_constant(value):
            raise ValueError('nonfinite model JSON: '+value)
        json.loads(model_bytes, parse_constant=reject_constant)
        # Ensure overflowed numeric literals also cannot survive parsing.
        json.dumps(json.loads(model_bytes), allow_nan=False)
        payload = dict(receipt, artifact_sha256=hashlib.sha256(model_bytes).hexdigest())
        receipt_text = json.dumps(payload, indent=2, allow_nan=False)+'\n'
        staged_receipt.write_text(receipt_text)
        published = []
        try:
            for staged, destination in ((staged_model, output), (staged_receipt, receipt_path)):
                os.link(staged, destination)  # atomic exclusive creation, never replace
                published.append((staged, destination))
        except BaseException:
            for staged, destination in reversed(published):
                # Never delete a concurrent replacement belonging to somebody else.
                try:
                    if os.path.samestat(staged.stat(), destination.lstat()):
                        destination.unlink()
                except FileNotFoundError:
                    pass
            raise
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--steps', default=100, type=int)
    parser.add_argument('--learning-rate', default=0.01, type=float)
    args = parser.parse_args(argv)
    receipt_path = Path(str(args.output) + '.training_receipt.json')
    if os.path.lexists(args.output) or os.path.lexists(receipt_path):
        parser.error('output or training receipt already exists; use a new output path')
    manifest = json.loads(args.manifest.read_text())
    # Existing project manifests are canonical_physics result envelopes.
    # Also accept the raw manifest used by the runtime API and unit fixtures.
    if isinstance(manifest, dict) and 'measured' in manifest:
        manifest = manifest['measured']
    with np.load(args.dataset, allow_pickle=False) as dataset:
        required = {'spikes', 'targets', 'metadata'}
        if not required.issubset(dataset.files):
            parser.error('dataset requires spikes, targets and metadata')
        metadata_array = dataset['metadata']
        if metadata_array.shape != () or metadata_array.dtype.kind != 'U':
            parser.error('metadata must be a scalar JSON string')
        try:
            metadata = json.loads(metadata_array.item())
            # Reject nonstandard JSON NaN/Infinity recursively before fitting or
            # exporting; otherwise receipt serialization could fail after save.
            json.dumps(metadata, allow_nan=False)
        except (ValueError, TypeError) as exc:
            parser.error('metadata must be finite JSON: '+str(exc))
        try:
            qualification = qualify_training_dataset(metadata, dataset)
        except ValueError as exc:
            parser.error(str(exc))
        command_ticks = qualification['command_ticks']
        result = fit(manifest, dataset['spikes'], dataset['targets'],
                     dataset['connected'] if 'connected' in dataset.files else None,
                     steps=args.steps, lr=args.learning_rate, command_ticks=command_ticks)
    result.receipt.update(dataset_metadata=metadata, data_qualification=qualification,
                          target_kind='INTERFACE_TEST',
                          manifest_path=str(args.manifest.resolve()),
                          dataset_path=str(args.dataset.resolve()),
                          manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                          dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest())
    # Both serialized artifacts are validated before either destination appears.
    result.receipt = publish_artifacts(result.translator, result.receipt, args.output, receipt_path)
    print(json.dumps(dict(output=str(args.output.resolve()), receipt=str(receipt_path.resolve()),
                          initial_loss=result.losses[0], final_loss=result.losses[-1],
                          target_kind='INTERFACE_TEST'), allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
