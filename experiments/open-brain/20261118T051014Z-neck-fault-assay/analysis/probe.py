"""Exercise the lesion boundaries using the actual fixed transit queue."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "20261118T002600Z-ascending-behavior-comparison"))
from neural_runtime import BIDIR, read, write
from fault_schedule import ARMS, FAULT_OFF, FAULT_ON, LesionSchedule, registered_targets
from fault_transport import FixedCircuit


def main():
    specs = registered_targets()
    an, dn = specs
    num = len(ARMS)
    ports = read(BIDIR/"CATALOG.json")["ports"]
    neck = FixedCircuit({"ports": {p["root"]: ports[p["root"]] for p in specs}}, batch=num)
    offset, delay = dn["entry_to_fault_ticks"], dn["total_delay_ticks"]
    on = FAULT_ON - offset
    off = FAULT_OFF - offset
    source = np.r_[on-1,on,on+1,np.arange(on+100,on+117),off-1,off,off+1]
    sample = []
    n = int(off + delay + 4)
    schedule = LesionSchedule(neck,10901,n)
    dn_column = neck.port_index[dn["root"]]
    cut = np.zeros((num, len(neck.ports)), dtype=bool)
    edrive = np.zeros_like(cut)
    drive = np.zeros((num,len(neck.roots)),dtype=np.float32)
    for tick in range(n):
        edrive[:,dn_column] = tick in source
        emitted, arrivals, _, _ = neck.step(drive,cut,eon_output=edrive,
                                            departure_loss=schedule.step(tick))
        assert np.array_equal(emitted[:,dn_column],edrive[:,dn_column])
        sample.append(arrivals[:,dn_column])
    received = np.asarray(sample)
    assert np.array_equal(np.flatnonzero(received[:,0]),source+delay)
    assert np.array_equal(np.flatnonzero(received[:,-1]),source[[0,-2,-1]]+delay)
    keep = ~schedule.lesion_loss[source+offset,4]
    active = source[(source>=on)&(source<off)]
    assert 0 < np.count_nonzero(schedule.lesion_loss[active+offset,4]) < len(active)
    assert np.array_equal(np.flatnonzero(received[:,4]),source[keep]+delay)
    assert received[on-1+delay,-1] and on-1+delay > FAULT_ON
    assert not received[off-1+delay,-1] and off-1+delay > FAULT_OFF
    staged = ROOT/"QUEUE_PROBE.next.json"
    write(staged,dict(
        root=dn["root"], dt_ms=.1, departure_ticks=source.tolist(), lesion_ticks=(source+offset).tolist(),
        arrival_ticks={arm:np.flatnonzero(received[:,lane]).tolist() for lane,arm in enumerate(ARMS)},
        independent_dropout_interleaves_loss_and_delivery=True,
        source_events_preserved=True, pre_on_transit_delivered=True,
        prior_lost_event_not_revived_at_release=True))
    staged.replace(ROOT/"QUEUE_PROBE.json")
    print("fixed queue boundary probe passed", flush=True)


if __name__ == "__main__":
    main()
