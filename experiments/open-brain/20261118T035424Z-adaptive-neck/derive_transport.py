"""Derive the variable-delay queue from the pinned P1 transmission implementation."""
from pathlib import Path
import argparse

HERE = Path(__file__).resolve().parent


def main(archive):
    parent = archive / "experiments/open-brain/20261118T031132Z-flat-neck-assay"
    source = (parent / "assay_transport.py").read_text()
    source = source.replace(
        "HERE=Path(__file__).resolve().parent.parent/'20261117T164207Z-fixed-neck'",
        "import os\nHERE=Path(os.environ['P1_ARCHIVE_ROOT']).resolve()/'experiments/open-brain/20261117T164207Z-fixed-neck'")
    source = source.replace(
        "velocity_multiplier=1.):",
        "velocity_multiplier=1.,max_extra_delay=0):")
    source = source.replace(
        "        self._pending=np.zeros",
        "        self.max_extra_delay=int(max_extra_delay)\n"
        "        if self.max_extra_delay<0:raise ValueError('negative delay budget')\n"
        "        self._pending=np.zeros")
    source = source.replace("int(self.neck_delay.max())+1,batch",
                            "int(self.neck_delay.max())+self.max_extra_delay+1,batch")
    source = source.replace("def step(self,drive,cut,eon_output=None):",
                            "def step(self,drive,cut,eon_output=None,extra_delay=None):")
    source = source.replace(
        "        for delay in np.unique(self.neck_delay):\n"
        "            cols=np.flatnonzero(self.neck_delay==delay)\n"
        "            target=(self._tick+int(delay))%len(self._pending)\n"
        "            self._pending[target][:,cols] |= events[:,cols] & ~cut[:,cols]",
        """        delay_at_emission=np.broadcast_to(self.neck_delay,cut.shape)
        if extra_delay is not None:
            if (extra_delay.shape!=cut.shape or not np.issubdtype(extra_delay.dtype,np.integer)
                    or (extra_delay<0).any() or (extra_delay>self.max_extra_delay).any()):
                raise ValueError('invalid lane-port delay')
            delay_at_emission=delay_at_emission+extra_delay
        active_events=events & ~cut
        for delay in np.unique(delay_at_emission[active_events]):
            target=(self._tick+int(delay))%len(self._pending)
            sending=active_events & (delay_at_emission==delay)
            if (self._pending[target] & sending).any():
                raise ValueError('two events collide in the same registered axon tick')
            self._pending[target] |= sending""")
    assert "delay_at_emission" in source and "P1_ARCHIVE_ROOT" in source
    (HERE / "dynamic_transport.py").write_text(source)
    timing = (parent / "assay_timing.py").read_text()
    timing = timing.replace("from neural_runtime import DT, eon\nfrom circuit import Circuit\n", "")
    timing = timing.replace("from assay_transport import", "from dynamic_transport import")
    timing = timing.replace("                             REVERSAL_MV, RISE_MS, FixedCircuit)",
        "                             REVERSAL_MV, RISE_MS, FixedCircuit)\n"
        "from circuit import Circuit, eon\n\nDT=.0001")
    timing = timing.replace("feedback_lanes, batch):\n        super().__init__(catalog, batch=batch)",
        "feedback_lanes, batch, max_extra_delay=0):\n"
        "        super().__init__(catalog, batch=batch, max_extra_delay=max_extra_delay)")
    assert "max_extra_delay=max_extra_delay" in timing
    (HERE / "dynamic_timing.py").write_text(timing)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    main(parser.parse_args().archive)
