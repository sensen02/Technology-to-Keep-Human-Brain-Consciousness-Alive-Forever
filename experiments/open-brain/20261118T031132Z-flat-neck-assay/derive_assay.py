"""Make a narrowly derived runner and preserve the prior experiment."""
from pathlib import Path

directory = Path(__file__).resolve().parent
prior = directory.parent / "20261118T020425Z-front-touch-timing-repair"
source = (prior / "fixed_transport.py").read_text()
source = source.replace("        if cut.shape!=(len(self.ports),) or cut.dtype!=np.bool_:",
    """        valid_shapes = ((len(self.ports),), (self.batch, len(self.ports)))
        if cut.shape not in valid_shapes or cut.dtype!=np.bool_:""")
source = source.replace("        self._pending[:,:,cut]=False", """        cut = np.broadcast_to(cut, (self.batch, len(self.ports)))
        self._pending &= ~cut[None, :, :]""")
source = source.replace("cols=np.flatnonzero((self.neck_delay==delay)&~cut)",
                        "cols=np.flatnonzero(self.neck_delay==delay)")
source = source.replace("|= events[:,cols]", "|= events[:,cols] & ~cut[:,cols]")
assert "[:,:,cut]" not in source
(directory/"assay_transport.py").write_text(source)

source = (prior/"timing.py").read_text()
source = source.replace("from fixed_transport import", "from assay_transport import")
(directory/"assay_timing.py").write_text(source)

source = (prior/"live_repair.py").read_text()
source = source.replace("from timing import CanonicalEvents, FrontTouchNeck",
                        """from assay_timing import CanonicalEvents, FrontTouchNeck
from schedule import ARMS, CUT_ROOT, CUT_ON, RESTORE, SEEDS, AssaySchedule""")
source = source.replace("MODES=('repaired','fixed_canonical','fixed_raw','matched_reference_boundary')","MODES=ARMS")
source = source.replace("matched_lanes=(0,3),feedback_lanes=(0,),batch=4",
                        "matched_lanes=range(4),feedback_lanes=range(4),batch=4")
source = source.replace("central_hits=((len(context_groups(index)[chosen]),),bool)).items()}",
    """central_hits=((len(context_groups(index)[chosen]),),bool),target_blocked=((4,),bool),
        scheduled_event=((4,),np.int8)).items()}""")
source = source.replace("    cut=np.zeros(len(neck_ports),dtype=bool)",
    """    assert an_roots[0]==CUT_ROOT and bound['records'][0]['body_input_side']=='left'
    schedule=AssaySchedule(len(neck_ports),neck.port_index[CUT_ROOT])""")
source = source.replace("matched_source_lanes=[0,3],matched_feedback_lanes=[0],raw_arrival_lanes=[2]",
    """matched_source_lanes=list(range(4)),matched_feedback_lanes=list(range(4)),
        target_root=CUT_ROOT,cut_on_tick=CUT_ON,restore_tick=RESTORE""")
source = source.replace("        emitted,arrivals,_,voltage=neck.step(drive,cut,eon_output=edrive)",
    """        cut=schedule.step(tick)
        neural['target_blocked'][tick]=cut[:,schedule.target]
        neural['scheduled_event'][tick]=schedule.scheduled_event(tick)
        emitted,arrivals,_,voltage=neck.step(drive,cut,eon_output=edrive)""")
source = source.replace("        events[2]=arrivals[2,an_cols]\n","")
source = source.replace("matched_source_lane_ticks=2*n","matched_source_lane_ticks=4*n")
source = source.replace("p.add_argument('seed',type=int);", "p.add_argument('seed',type=int,choices=SEEDS);")
source = source.replace("extra_files.add(REPAIR/'PROTOCOL.zh-CN.md')",
    """extra_files.update({REPAIR/'PROTOCOL.zh-CN.md',
                        BASE/'20261118T020425Z-front-touch-timing-repair/live_repair.py',
                        BASE/'20261118T020425Z-front-touch-timing-repair/timing.py',
                        BASE/'20261118T020425Z-front-touch-timing-repair/fixed_transport.py'})""")
assert "events[2]=" not in source and "matched_lanes=range(4)" in source
(directory/"run_assay.py").write_text(source)
