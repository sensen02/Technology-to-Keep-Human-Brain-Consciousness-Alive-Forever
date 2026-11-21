"""Derive a live test from the verified four-arm flat platform."""
from pathlib import Path

root = Path(__file__).resolve().parent
origin = root.parent / "20261118T031132Z-flat-neck-assay"
transport = (origin/"assay_transport.py").read_text()
transport = transport.replace("def step(self,drive,cut,eon_output=None):",
                              "def step(self,drive,cut,eon_output=None,delay_ticks=None):")
begin = transport.index("        for delay in np.unique(self.neck_delay):")
end = transport.index("        self._tick+=1",begin)
transport = transport[:begin] + '''        delays = (np.broadcast_to(self.neck_delay, events.shape)
                  if delay_ticks is None else np.asarray(delay_ticks))
        if (delays.shape != events.shape or delays.dtype.kind not in "iu"
                or (delays < 1).any() or (delays >= len(self._pending)).any()):
            raise ValueError("delay schedule is outside the allocated integer queue")
        for delay in np.unique(delays):
            send = (delays == delay) & events & ~cut
            target=(self._tick+int(delay))%len(self._pending)
            if (self._pending[target] & send).any():
                raise ValueError("two separate emissions assigned to one arrival")
            self._pending[target] |= send
''' + transport[end:]
(root/"implant_transport.py").write_text(transport)
timing=(origin/"assay_timing.py").read_text().replace("from assay_transport import","from implant_transport import")
(root/"implant_timing.py").write_text(timing)

src=(origin/"run_assay.py").read_text().replace("from assay_timing import","from implant_timing import")
src=src.replace("from schedule import ARMS, CUT_ROOT, CUT_ON, RESTORE, SEEDS, AssaySchedule",
                "from profiles import MODES, SEEDS, read_profiles")
src=src.replace("MODES=ARMS\n","")
src=src.replace("target_blocked=((4,),bool),\n        scheduled_event=((4,),np.int8)",
                "departure_delay_ticks=((4,3),np.int16)")
src=src.replace("    assert an_roots[0]==CUT_ROOT and bound['records'][0]['body_input_side']=='left'\n"
                "    schedule=AssaySchedule(len(neck_ports),neck.port_index[CUT_ROOT])",
                """    delay_profiles,selected_columns,selected_roots=read_profiles(neck)
    required_slots=int(delay_profiles.max())+1
    if len(neck._pending)<required_slots:
        neck._pending=np.zeros((required_slots,4,len(neck_ports)),dtype=bool)
    cut=np.zeros((4,len(neck_ports)),dtype=bool)""")
src=src.replace("    pins=sources(extra_files);pins.update(assets)",
                """    extra_files.update(REPAIR/name for name in ('MECHANICS.json','MECHANICS.npz','CABLE.json','TRANSPORT_PROFILES.npz'))
    pins=sources(extra_files);pins.update(assets)""")
src=src.replace("target_root=CUT_ROOT,cut_on_tick=CUT_ON,restore_tick=RESTORE",
                "selected_roots=selected_roots, delay_source='offline FE and paired NEURON geometry proxy'")
src=src.replace("        cut=schedule.step(tick)\n"
                "        neural['target_blocked'][tick]=cut[:,schedule.target]\n"
                "        neural['scheduled_event'][tick]=schedule.scheduled_event(tick)\n"
                "        emitted,arrivals,_,voltage=neck.step(drive,cut,eon_output=edrive)",
                """        delays=delay_profiles[tick]
        neural['departure_delay_ticks'][tick]=delays[:,selected_columns]
        emitted,arrivals,_,voltage=neck.step(drive,cut,eon_output=edrive,delay_ticks=delays)""")
assert "AssaySchedule" not in src and "departure_delay_ticks" in src
(root/"run_live.py").write_text(src)
