"""Run a frozen FlyWire brain and replay its emitted spikes through BANC DN axons."""
import argparse
import importlib.metadata
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow.parquet as parquet
from scipy.sparse import csr_matrix
import torch

from sources import COMMIT, HERE, UPSTREAM, digest, flywire_annotations, make_binding, verify_upstream, write

sys.path.insert(0, str(UPSTREAM / "code"))
import run_pytorch as eon


TICKS = 1600
STIM_START, STIM_END = 200, 1100
CUT_START, CUT_END = 450, 800
NECK_DELAY = 20
RATE_HZ = 100.0
SEED = 1729


def load_connectome(root_index):
    columns = ["Postsynaptic_Index", "Presynaptic_Index", "Excitatory x Connectivity"]
    table = parquet.read_table(eon.path_con, columns=columns)
    post, pre, weights = [column.to_numpy() for column in table.columns]
    size = len(root_index)
    assert size == 138639 and len(weights) == 15091983
    assert all(x.min() >= 0 and x.max() < size for x in (post, pre))
    matrix = csr_matrix((weights.astype(np.float32), (post, pre)), shape=(size, size))
    assert matrix.nnz == len(weights) and matrix.has_canonical_format
    # Integer-weight sums are exactly representable in float32 for any binary source volley.
    assert float(np.abs(matrix).sum(axis=1).max()) < 2**24
    return matrix


class FrozenCPUBrain(eon.TorchModel):
    """Upstream dynamics with an equivalent sparse reduction over active sources."""

    def __init__(self, batch, weights):
        super().__init__(batch, weights.shape[0], eon.DT, eon.MODEL_PARAMS, weights)
        self.outgoing = weights.transpose().tocsr()

    def recurrent_volley(self, spikes):
        result = np.zeros(spikes.shape, dtype=np.float32)
        for lane, row in enumerate(spikes.numpy()):
            active = np.flatnonzero(row)
            if active.size:
                result[lane] = self.outgoing[active].sum(axis=0)
        return torch.from_numpy(result)

    def forward(self, rates, *state, generator):
        drive = self.poisson(rates, generator=generator)
        volley = self.recurrent_volley(state[2])
        return self.neurons(volley * self.scale, drive * self.scale, *state), drive


def native_parity(weights, left, pfl3):
    size = weights.shape[0]
    sparse = torch.sparse_csr_tensor(weights.indptr, weights.indices, weights.data, size=weights.shape)
    stock = eon.TorchModel(1, size, eon.DT, eon.MODEL_PARAMS, sparse)
    candidate = FrozenCPUBrain(1, weights)
    initial = stock.state_init()
    ref, test = initial, candidate.state_init()
    ref_rng, test_rng = [torch.Generator().manual_seed(SEED) for _ in range(2)]
    rates = torch.zeros((1, size))
    rates[0, [left, *pfl3]] = RATE_HZ
    active_volley_ticks = mixed_volley_ticks = 0
    for tick in range(80):
        fast = candidate.recurrent_volley(ref[2])
        original = torch.matmul(ref[2], sparse.transpose(0, 1))
        assert torch.equal(original, fast), tick
        active_volley_ticks += bool(torch.any(original))
        mixed_volley_ticks += bool(torch.any(original > 0) and torch.any(original < 0))
        ref = stock(rates, *ref, generator=ref_rng)
        test, _ = candidate(rates, *test, generator=test_rng)
        assert all(torch.equal(a, b) for a, b in zip(ref, test, strict=True)), tick
    assert active_volley_ticks > 0 and mixed_volley_ticks > 0
    return dict(ticks=80, all_states_bitexact=True, active_volley_ticks=active_volley_ticks,
                mixed_signed_volley_ticks=mixed_volley_ticks)


def protocols(weights, root_index, ports):
    index_root = {index: root for root, index in root_index.items()}
    targets = [row["flywire_index"] for row in ports]
    classes = flywire_annotations("consolidated_cell_types")
    pfl_roots = sorted(int(root) for root, row in classes.items()
                       if row["primary_type"] == "PFL3" and int(root) in root_index)
    pfl = [root_index[root] for root in pfl_roots]
    synapses = weights[targets[:2]][:, pfl].toarray()
    steer = []
    for index in range(2):
        selected = np.flatnonzero((synapses[index] > 0) & (synapses[index] > synapses[1 - index]))
        assert selected.size > 0
        steer.append([pfl[p] for p in selected])
    p9 = [root_index[root] for root in eon.get_experiment("p9")["neu_exc"]]
    stim = {
        "off": [],
        "direct_DNa02_left": targets[:1],
        "direct_DNa02_right": targets[1:2],
        "direct_DNg100_bilateral": targets[2:],
        "PFL3_to_left_DNa02": steer[0],
        "PFL3_to_right_DNa02": steer[1],
        "DNp09_bilateral": p9,
    }
    return targets, stim, {case: [str(index_root[i]) for i in indices] for case, indices in stim.items()}


def record_brain(weights, targets, stim):
    batch = len(stim)
    model = FrozenCPUBrain(batch, weights)
    generator = torch.Generator().manual_seed(SEED)
    state = model.state_init()
    amplitudes = torch.zeros((batch, weights.shape[0]))
    for lane, indices in enumerate(stim.values()):
        amplitudes[lane, indices] = RATE_HZ
    zero = torch.zeros_like(amplitudes)
    spikes = np.zeros((TICKS, batch, len(targets)), dtype=np.uint8)
    voltage = np.zeros(spikes.shape, dtype=np.float32)
    injected_count = [0] * batch
    full_spikes = np.zeros((TICKS, batch), dtype=np.int32)
    for tick in range(TICKS):
        active = STIM_START <= tick < STIM_END
        state, drive = model(amplitudes if active else zero, *state, generator=generator)
        spikes[tick] = state[2][:, targets].numpy()
        voltage[tick] = state[3][:, targets].numpy()
        full_spikes[tick] = state[2].sum(1).numpy()
        if active:
            for lane, indices in enumerate(stim.values()):
                injected_count[lane] += int(torch.count_nonzero(drive[lane, indices]))
        if (tick + 1) % 400 == 0:
            print(f"Brain {tick + 1}/{TICKS} ticks", flush=True)
    assert not np.any(full_spikes[:, 0]) and not np.any(full_spikes[:STIM_START])
    for lane, relevant in ((1, (0,)), (2, (1,)), (3, (2, 3))):
        assert all(spikes[STIM_START:STIM_END, lane, column].any() for column in relevant)
    assert np.isfinite(voltage).all()
    return spikes, voltage, full_spikes, injected_count


class NeckLink:
    """Tick-synchronous same-axon carrier followed by fixed BANC first-hop synapses."""

    def __init__(self, batch, counts):
        self.pending = np.zeros((NECK_DELAY, batch, counts.shape[0]), dtype=np.uint8)
        self.weights = torch.from_numpy(counts.astype(np.float32) * eon.MODEL_PARAMS["wScale"])
        self.cells = eon.AlphaLIF(batch, counts.shape[1], eon.DT, eon.MODEL_PARAMS)
        self.state = self.cells.state_init()
        self.zero = torch.zeros((batch, counts.shape[1]))
        self.tick = 0

    def step(self, emitted, cut):
        assert emitted.dtype == np.uint8 and emitted.shape == self.pending.shape[1:]
        assert not np.any(emitted > 1) and cut.dtype == np.bool_ and cut.shape == (emitted.shape[1],)
        self.pending[:, :, cut] = 0
        head = self.tick % NECK_DELAY
        delivered = self.pending[head].copy()
        self.pending[head] = emitted
        self.pending[head, :, cut] = 0
        self.tick += 1
        volley = torch.from_numpy(delivered.astype(np.float32)) @ self.weights
        self.state = self.cells(volley, self.zero, *self.state)
        return delivered, self.state[2].numpy(), self.state[3].numpy()


def replay(directory, mode, brain, counts, cut_ports):
    steps, batch, channels = brain.shape
    link = NeckLink(batch, counts)
    delivered = np.zeros_like(brain)
    spikes = np.zeros((steps, batch, counts.shape[1]), dtype=np.uint8)
    voltage = np.zeros(spikes.shape, dtype=np.float32)
    open_links = np.zeros(channels, dtype=bool)
    for tick, emitted in enumerate(brain):
        cut = cut_ports if CUT_START <= tick < CUT_END else open_links
        delivered[tick], spikes[tick], voltage[tick] = link.step(emitted, cut)
    assert np.isfinite(voltage).all()
    np.savez_compressed(directory / f"VNC_{mode}.npz", delivered=delivered, spikes=spikes, voltage_mv=voltage)
    return dict(delivered_by_trial_and_port=delivered.sum(0).tolist(),
                vnc_spikes_by_trial=spikes.sum(axis=(0, 2)).tolist(),
                vnc_peak_mv_by_trial=voltage.max(axis=(0, 2)).tolist())


@torch.inference_mode()
def main(directory):
    start = perf_counter()
    source = verify_upstream()
    root_index, _ = eon.get_hash_tables(eon.path_comp)
    ports, vnc_ids, counts = make_binding(root_index)
    weights = load_connectome(root_index)
    targets, stim, recorded_stim = protocols(weights, root_index, ports)
    parity = native_parity(weights, targets[0], stim["PFL3_to_left_DNa02"])
    print("Native sparse parity verified", parity, flush=True)
    mode_cuts = {"intact": [False] * 4, "cut_all": [True] * 4,
                 "cut_left_DNa02": [True, False, False, False]}
    write(directory / "INPUTS.json", dict(upstream_commit=COMMIT, upstream=source,
          implementation={name: digest(HERE / name) for name in ("sources.py", "run_probe.py", "PROTOCOL.zh-CN.md")},
          neurons=len(root_index), flywire_rows=weights.nnz, dt_ms=eon.DT, seed=SEED,
          steps=TICKS, stim_hz=RATE_HZ, stim_ticks=[STIM_START, STIM_END],
          cases=list(stim), external_input_flywire_roots=recorded_stim, ports=ports, vnc_ids=vnc_ids,
          cut_ticks=[CUT_START, CUT_END], mode_cuts=mode_cuts, neck_delay_ticks=NECK_DELAY,
          neck_delay_basis="engineering assumption", numerical_parameters=eon.MODEL_PARAMS,
          vnc_scope="first-hop DN targets only", epochs=0, physics_steps=0,
          environment={name: importlib.metadata.version(name) for name in ("torch", "scipy", "pyarrow", "numpy")}))
    np.save(directory / "BANC_COUNTS.npy", counts, allow_pickle=False)
    spikes, voltage, full_spikes, injected_count = record_brain(weights, targets, stim)
    np.savez_compressed(directory / "BRAIN.npz", emitted=spikes, voltage_mv=voltage,
                        all_cells_spike_count=full_spikes)
    del weights
    modes = {}
    for mode, cut in mode_cuts.items():
        modes[mode] = replay(directory, mode, spikes, counts, np.array(cut))
        print(f"BANC {mode} saved", flush=True)
    write(directory / "PROBE.json", dict(native_equivalence=parity, injected_counts=injected_count,
          brain_spikes=dict(before=spikes[:STIM_START].sum(0).tolist(),
                            stimulus=spikes[STIM_START:STIM_END].sum(0).tolist(),
                            after_withdrawal=spikes[STIM_END:].sum(0).tolist()),
          vnc=modes, seconds=perf_counter() - start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir()
    torch.set_num_threads(1)
    main(args.output)
