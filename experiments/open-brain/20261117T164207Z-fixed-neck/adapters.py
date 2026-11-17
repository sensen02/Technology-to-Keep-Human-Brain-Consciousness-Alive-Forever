"""Load fixed dynamics into isolated historical ground regression entry points."""
import json
from pathlib import Path
import sys

from fixed_neck import FixedCircuit, HERE, PARENT, full_catalog
sys.path.insert(0,str(HERE.parent/'20261117T062447Z-body-commands/PHASE'))
import phase
import body_run
import expansion
import physical

LegacyReceiver=physical.Receiver
CALIBRATION=HERE/'READOUT.json'


class CalibratedReceiver(LegacyReceiver):
    def __init__(self,circuit):
        super().__init__(circuit)
        self._raw=self.filtered.copy()
        self.maps=json.loads(CALIBRATION.read_text())['maps']

    def update(self,voltage,interval):
        published=self.filtered
        self.filtered=self._raw
        super().update(voltage,interval)
        self._raw=self.filtered.copy()
        import numpy as np
        self.filtered={key:float(np.interp(value,self.maps['/'.join(key)]['new'],
                                                    self.maps['/'.join(key)]['old']))
                       for key,value in self._raw.items()}


def install(mapped):
    body_run.Circuit=FixedCircuit
    expansion.Circuit=FixedCircuit
    receiver=CalibratedReceiver if mapped else LegacyReceiver
    body_run.Receiver=receiver
    expansion.Receiver=receiver
    physical.Circuit=FixedCircuit
    physical.Receiver=receiver
