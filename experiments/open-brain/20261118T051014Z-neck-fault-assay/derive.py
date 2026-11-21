"""Prepare the frozen runner with a localized axon-transport intervention."""
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ORIGIN = HERE.parent / "20261118T031132Z-flat-neck-assay"


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new, 1)


def main():
    transport = (ORIGIN / "assay_transport.py").read_text()
    transport = replace_once(transport, "def step(self,drive,cut,eon_output=None):",
        "def step(self,drive,cut,eon_output=None,departure_loss=None):")
    transport = replace_once(transport, "        for delay in np.unique(self.neck_delay):",
        """        admitted = events & ~cut
        if departure_loss is not None:
            if departure_loss.shape != events.shape or departure_loss.dtype != np.bool_:
                raise ValueError("transit outcome must index each registered axon and lane")
            admitted &= ~departure_loss
        for delay in np.unique(self.neck_delay):""")
    transport = replace_once(transport, "events[:,cols] & ~cut[:,cols]", "admitted[:,cols]")
    (HERE / "fault_transport.py").write_text(transport)
    timing = replace_once((ORIGIN / "assay_timing.py").read_text(),
                          "from assay_transport import", "from fault_transport import")
    (HERE / "fault_timing.py").write_text(timing)

    runner = (ORIGIN / "run_assay.py").read_text()
    runner = replace_once(runner, "from assay_timing import", "from fault_timing import")
    runner = replace_once(runner, "from schedule import ARMS, CUT_ROOT, CUT_ON, RESTORE, SEEDS, AssaySchedule",
        "from fault_schedule import ARMS, FAULT_ON, FAULT_OFF, SEEDS, LesionSchedule, GEOMETRY")
    runner = replace_once(runner, "MODES=ARMS\n", "MODES=ARMS\nLANES=len(MODES)\n")
    runner = re.sub(r"\b4\b", "LANES", runner)
    runner = runner.replace("[0,1,2,3]", "list(range(LANES))")
    runner = replace_once(runner, "target_blocked=((LANES,),bool),\n        scheduled_event=((LANES,),np.int8)",
        "target_departure_loss=((LANES,),bool)")
    runner = replace_once(runner,
        "    assert an_roots[0]==CUT_ROOT and bound['records'][0]['body_input_side']=='left'\n"
        "    schedule=AssaySchedule(len(neck_ports),neck.port_index[CUT_ROOT])",
        """    schedule=LesionSchedule(neck,seed,n)
    assert schedule.target_roots[1]==an_roots[0] and bound['records'][0]['body_input_side']=='left'
    assert schedule.target_roots[-1] in ground_roots
    cut=np.zeros((LANES,len(neck_ports)),dtype=bool)""")
    runner = replace_once(runner, "    pins=sources(extra_files);pins.update(assets)",
        """    extra_files.add(GEOMETRY)
    extra_files.update(Path(p) for p in read(GEOMETRY)['source_sha256'])
    pins=sources(extra_files);pins.update(assets)""")
    runner = replace_once(runner, "target_root=CUT_ROOT,cut_on_tick=CUT_ON,restore_tick=RESTORE",
        "target_roots=schedule.target_roots,entry_to_fault_ticks=schedule.offsets.tolist(),\n"
        "        fault_on_tick=FAULT_ON,transient_restore_tick=FAULT_OFF")
    runner = replace_once(runner, "        cut=schedule.step(tick)\n"
        "        neural['target_blocked'][tick]=cut[:,schedule.target]\n"
        "        neural['scheduled_event'][tick]=schedule.scheduled_event(tick)\n"
        "        emitted,arrivals,_,voltage=neck.step(drive,cut,eon_output=edrive)",
        """        loss=schedule.step(tick)
        neural['target_departure_loss'][tick]=schedule.departure_loss[tick]
        emitted,arrivals,_,voltage=neck.step(drive,cut,eon_output=edrive,departure_loss=loss)""")
    runner = replace_once(runner, "if time.monotonic()-began>600", "if time.monotonic()-began>1200")
    assert "CUT_ROOT" not in runner and "'target_departure_loss'" in runner
    compile(runner, "run_fault.py", "exec")
    (HERE / "run_fault.py").write_text(runner)


if __name__ == "__main__":
    main()
