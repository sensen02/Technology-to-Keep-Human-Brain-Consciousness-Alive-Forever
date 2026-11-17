"""Write the result addendum separately from the already exclusive-created RESULT."""
import argparse
from pathlib import Path

from approach_runtime import turn, sha


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    args = parser.parse_args()
    folder = args.trial.resolve()
    root = Path(__file__).resolve().parent
    result = turn.read(folder / "RESULT.json")
    files = {**result["files"], **result["supplemental_hashes"]}
    assert all(sha(folder / name) == value for name,value in files.items())
    more = {}
    for name in ("GEOMETRY.npz","OBSTACLE.json"):
        if (folder / name).is_file():
            more[name] = sha(folder / name)
    source_identity = turn.read(root / "INPUT_IDENTITY.json")
    runtime = {Path(name):digest for name,digest in source_identity["sources"].items()
               if Path(name) not in (root / "archive_receipt.py", root / "audit_pair.py")}
    assert all(sha(name) == digest for name,digest in runtime.items())
    turn.stand.put(folder / "ARCHIVE_RECEIPT.json", dict(
        result_sha256=sha(folder / "RESULT.json"), files=dict(**files, **more),
        original_body_identity_sha256=sha(folder.with_name(folder.name + "_FIXTURE") / "IDENTITY.json"),
        pinned_input_identity_sha256=sha(root / "INPUT_IDENTITY.json"),
        inputs_verified=True, native_steps=result["native_steps"], behavior_status=result["passed"],
        failure="Runner attempted a second exclusive RESULT creation after all physical archives were committed",
        recovery="Verify archived result, all recorded file digests and inputs, then write this separate addendum"))
