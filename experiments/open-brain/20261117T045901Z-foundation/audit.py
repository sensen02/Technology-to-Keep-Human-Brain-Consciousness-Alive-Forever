"""Independently verify saved axon delivery and first-hop voltage with NumPy."""
import argparse
from pathlib import Path

import numpy as np

from sources import HERE, UPSTREAM, digest, load, write


def expected_arrivals(emitted, delay, cut_window, affected):
    first, last = cut_window
    arrived = np.zeros_like(emitted)
    discarded = np.zeros(emitted.shape[-1], dtype=np.int64)
    for tick, trial, port in np.argwhere(emitted):
        due = tick + delay
        if affected[port] and tick < last and due >= first:
            discarded[port] += 1
        elif due < emitted.shape[0]:
            arrived[due, trial, port] = 1
    return arrived, discarded


def verify_vnc(counts, delivered, spikes, voltages, params, dt):
    n_ticks, n_trials, _ = delivered.shape
    n_cells = counts.shape[1]
    shape = n_trials, n_cells
    rest = np.float32(params["vRest"])
    reset = np.float32(params["vReset"])
    synaptic_decay = np.float32(1 - dt / params["tauSyn"])
    membrane_step = np.float32(dt / params["tauMem"])
    synaptic_lag = int(params["tDelay"] / dt) + 1
    refractory_ticks = round(params["tRefrac"] / dt)
    gain = np.float32(params["wScale"])
    weight = counts.astype(np.float32) * gain
    conductance = np.zeros(shape, dtype=np.float32)
    voltage = np.full(shape, params["v0"], dtype=np.float32)
    elapsed = np.full(shape, refractory_ticks, dtype=np.int32)
    fired = np.zeros(shape, dtype=bool)
    peak_error = 0.0
    for tick in range(n_ticks):
        elapsed = np.where(fired, 0, elapsed + 1)
        current = conductance * synaptic_decay
        if tick >= synaptic_lag:
            sources = delivered[tick - synaptic_lag].astype(np.float32)
            current += (sources @ weight) * (elapsed >= refractory_ticks)
        voltage += membrane_step * (conductance - (voltage - rest))
        fired = voltage > params["vThreshold"]
        assert np.array_equal(fired, spikes[tick]), tick
        voltage[fired] = reset
        peak_error = max(peak_error, float(np.max(np.abs(voltage - voltages[tick]))))
        current[fired] = 0.0
        conductance = current
    assert peak_error <= 0.00002, peak_error
    return peak_error


def main(directory):
    config = load(directory / "INPUTS.json")
    probe = load(directory / "PROBE.json")
    for name, source in config["upstream"].items():
        assert digest(UPSTREAM / name) == source["sha256"], name
    for name, source in config["implementation"].items():
        assert digest(HERE / name) == source, name
    for record in config["ports"]:
        assert digest(Path(record["swc"])) == record["swc_sha256"]

    cases = config["cases"]
    first, last = config["cut_ticks"]
    delay = config["neck_delay_ticks"]
    with np.load(directory / "BRAIN.npz") as brain:
        emitted = brain["emitted"]
        assert emitted.shape == (config["steps"], len(cases), 4)
        assert probe["native_equivalence"]["all_states_bitexact"]
        assert (brain["all_cells_spike_count"] >= emitted.sum(axis=-1)).all()
    counts = np.load(directory / "BANC_COUNTS.npy", allow_pickle=False)
    assert np.count_nonzero(counts) == 2791 and len(config["vnc_ids"]) == counts.shape[1]
    assert counts.sum(axis=1).tolist() == [row["vnc_synapses"] for row in config["ports"]]
    evidence = {}
    normal = None
    for mode, cuts in config["mode_cuts"].items():
        path = directory / f"VNC_{mode}.npz"
        with np.load(path) as trace:
            delivered, voltage, spikes = trace["delivered"], trace["voltage_mv"], trace["spikes"]
        expected, discarded = expected_arrivals(emitted, delay, (first, last), cuts)
        assert np.array_equal(delivered, expected), mode
        assert not np.any(delivered[:, 0]) and not np.any(spikes[:, 0])
        assert np.all(voltage[:, 0] == config["numerical_parameters"]["vRest"])
        error = verify_vnc(counts, delivered, spikes, voltage, config["numerical_parameters"], config["dt_ms"])
        first_release = {}
        for lane, port in ((1, 0), (2, 1), (3, 2), (3, 3)):
            positive = np.flatnonzero(delivered[last:, lane, port])
            if positive.size:
                first_release[f"{cases[lane]}/{port}"] = int(last + positive[0])
        if mode == "intact":
            normal = delivered
            for lane, ports in ((1, (0,)), (2, (1,)), (3, (2, 3))):
                relevant = np.any(counts[list(ports)] > 0, axis=0)
                assert np.max(voltage[:, lane, relevant]) > config["numerical_parameters"]["vRest"]
        else:
            assert not np.any(delivered[first:last + delay, :, cuts])
            assert np.array_equal(delivered[:first], normal[:first])
            assert np.array_equal(delivered[last + delay:], normal[last + delay:])
            assert np.array_equal(delivered[:, :, np.logical_not(cuts)], normal[:, :, np.logical_not(cuts)])
            assert discarded[np.array(cuts)].min() > 0 and first_release
        evidence[mode] = dict(discarded_by_port=discarded.tolist(),
                              first_release_at_or_after_restore=first_release,
                              max_voltage_error_mv=error, vnc_spikes_bitexact=True,
                              spike_and_voltage_values=int(voltage.size))
        print(f"{mode}: first-hop recomputation max {error:.8g} mV", flush=True)
    paths = [directory / name for name in ("INPUTS.json", "PROBE.json", "BANC_COUNTS.npy", "BRAIN.npz")]
    paths.extend(directory / f"VNC_{mode}.npz" for mode in evidence)
    write(directory / "AUDIT.json", dict(result="PASS", evidence=evidence,
          hashes={path.name: digest(path) for path in paths}, independent_numerical_audit=True,
          emitted_counts=emitted.sum(axis=0).tolist(), scope="downstream first-hop neural bench"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
