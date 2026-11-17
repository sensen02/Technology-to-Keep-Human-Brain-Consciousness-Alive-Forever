"""Thin immutable body neural-interface reuse. No Torch import or model construction here."""
from pathlib import Path
import importlib.util
HERE=Path(__file__).resolve().parent
BODY_INTERFACE=HERE.parent/'body_interface'
_s=importlib.util.spec_from_file_location('_dnp09_frozen_body_interface',BODY_INTERFACE/'interface.py');_m=importlib.util.module_from_spec(_s);_s.loader.exec_module(_m)
InterfaceError=_m.InterfaceError
require=_m.require
sha=_m.sha
source_ports=_m.source_ports
validate_delivery=_m.validate_delivery
BODY_SENSORY_INDICES=_m.BODY_SENSORY_INDICES
EXCLUDED_INDICES=_m.EXCLUDED_INDICES
extract_body_sensory=_m.extract_body_sensory
make_neural_interfaces=_m.make_neural_interfaces
