"""Selected axon latency, in-flight cancellation, and paired selectivity."""
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "20261118T002600Z-ascending-behavior-comparison"
sys.path.insert(0, str(PRIOR))
from neural_runtime import BIDIR, FIT, read, write, sha
from assay_timing import FrontTouchNeck
from schedule import CUT_ROOT


def expected_arrivals(spikes, delay, blocked):
    """A pulse requires an open axon from emission through arrival, inclusive."""
    steps = len(spikes)
    prefix = np.r_[0, blocked.astype(int).cumsum()]
    departure_through_arrival_open = prefix[delay+1:] - prefix[:steps-delay] == 0
    result = np.zeros_like(spikes, dtype=bool)
    result[delay:] = spikes[:-delay] & departure_through_arrival_open
    return result


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    catalog = read(BIDIR/"CATALOG.json")["ports"]
    order = [row["banc_root"] for row in read(FIT/"BINDING.json")["records"]]
    assert order[0] == CUT_ROOT
    ports = {root: catalog[root] for root in order}
    model = FrontTouchNeck({"ports": ports}, order, (0,1), (0,1), batch=2)
    columns = [model.validate_inputs([receptor for receptor, _ in port["incoming_receptors"]], "body")
               for port in ports.values()]
    drive = np.zeros((2,len(model.roots)), dtype=np.float32)
    mask = np.zeros((2,2), dtype=bool)
    t0 = tc = None
    emission, arrival, blocked = [], [], []
    for step in range(1500):
        drive.fill(0)
        if step % 23 == 0:
            for selected in columns:
                drive[:,selected] = 68.75
        if t0 is None and emission and emission[-1][0,0] and step > 100:
            # Deliberately interrupt a pulse known to be in flight. This event-
            # triggered stress test is separate from the fixed live schedule.
            t0 = step
            tc = step + 200
        mask[1,0] = bool(t0 is not None and t0 <= step < tc)
        e, a, _, _ = model.step(drive, mask)
        emission.append(e)
        arrival.append(a)
        blocked.append(mask.copy())
    emission, arrival, blocked = map(np.asarray,(emission,arrival,blocked))
    assert t0 is not None and tc is not None
    delay = int(model.neck_delay[0])
    canceled_time = t0 - 1 + delay
    assert arrival[canceled_time,0,0] and not arrival[canceled_time,1,0]
    assert np.array_equal(emission[:,0],emission[:,1])
    assert np.array_equal(arrival[:,0,1],arrival[:,1,1])
    assert not arrival[t0:tc,1,0].any()
    for lane in range(2):
        for index in range(2):
            assert np.array_equal(arrival[:,lane,index],
                 expected_arrivals(emission[:,lane,index],int(model.neck_delay[index]),blocked[:,lane,index]))
    restored = tc + int(np.flatnonzero(arrival[tc:,1,0])[0])
    assert restored >= tc + delay
    data = HERE/"QUEUE_PROBE.npz"
    np.savez_compressed(data, emitted=emission, arrived=arrival, blocked=blocked)
    write(HERE/"QUEUE_PROBE.json",dict(pass_=True,probe_ticks=1500,lane_ticks=3000,
        target=CUT_ROOT,actual_delay_ticks=delay,cut_tick=t0,opening_tick=tc,
        in_flight_pulse_lost_at_tick=canceled_time,first_new_postreopen_tick=restored,
        same_emission_for_both_lanes=True,other_AN_arrivals_exact=True,trace_sha256=sha(data)))
    print("In-flight cancellation and other-port isolation verified.",flush=True)


if __name__=="__main__":
    main()
