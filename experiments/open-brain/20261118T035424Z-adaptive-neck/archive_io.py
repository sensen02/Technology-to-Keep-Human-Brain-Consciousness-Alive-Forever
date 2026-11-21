"""Read immutable P1 data owned by the evidence archive."""
from pathlib import Path
import hashlib
import json
import os

import numpy as np

HERE = Path(__file__).resolve().parent
ARCHIVE = Path(os.environ.get("P1_ARCHIVE_ROOT", "/run/media/sensen/Data2/cell_wound_prototype")).resolve()
P1 = ARCHIVE / "experiments/open-brain"
ASSAY = P1 / "20261118T031132Z-flat-neck-assay"
BIDIR = P1 / "20261117T051249Z-bidirectional"
FIXED = P1 / "20261117T164207Z-fixed-neck"
BINDING = P1 / "20261117T235930Z-ascending-reference-fit/BINDING.json"


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as out:
        json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write("\n")


def load_case(seed):
    case = ASSAY / "FORMAL" / f"seed{seed}"
    meta = read(case / "RESULT.json")
    path = case / "NEURAL.npz"
    assert digest(path) == meta["neural_sha256"]
    with np.load(path, allow_pickle=False) as arrays:
        data = {key: arrays[key].copy() for key in arrays.files}
    return case, meta, data
