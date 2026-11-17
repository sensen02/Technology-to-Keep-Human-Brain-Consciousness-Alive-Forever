"""Fingerprint the running source files, weights and published binding."""
import hashlib
import platform
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy
import torch

import far_navigation as nav
import approach_runtime as rt
import approach_vision
import obstacle
import verify_native

here = Path(__file__).resolve().parent
turn = rt.turn
xml = Path(nav.BODY.PAD_XML)
inputs = [*here.glob("*.py"), here / "drive.sh", *rt.HERE.glob("*.py"),
          rt.BINDING, rt.PIVOT, rt.PACE, approach_vision.CHECKPOINT,
          nav.WEIGHTS, nav.LIB, verify_native.SO, xml, Path(nav.BODY.__file__),
          turn.STARTS / "BANK.json", turn.STARTS / "nominal.npz", turn.TEACHER,
          turn.PARENT / "PROTOTYPES.npz", rt.TURN / "CODEC.pt"]
for node in ET.parse(xml).iter():
    if "file" in node.attrib:
        asset = (xml.parent / node.attrib["file"]).resolve()
        if asset.is_file():
            inputs.append(asset)
for name in ("forward_link","optics","camera","pace_lookup","pace_extended","speed_teacher",
             "control_body","turning","stand","verify_native","walking_bench","bench_runtime","codec"):
    module = sys.modules.get(name)
    if module is not None:
        inputs.append(Path(module.__file__))
source = {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(inputs))}
turn.stand.put(here / "INPUT_IDENTITY.json", dict(
    scope="Fingerprint capture after physical jobs had started; jobs' actual result bindings verify at collection",
    python=platform.python_version(), mujoco=mujoco.__version__,
    numpy=numpy.__version__, torch=torch.__version__, sources=source))
