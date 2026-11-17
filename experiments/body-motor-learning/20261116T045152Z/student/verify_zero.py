"""Compare the zero-correction policy with every frozen nominal and quiet decision."""
import numpy as np
import torch

from residual import BASE, HERE, INPUTS, Residual, base_sources, read, stand, wrap


def verify():
    torch.set_num_threads(1)
    model = Residual(np.zeros(INPUTS), np.ones(INPUTS), .15)
    policy = wrap(model)
    archives = {}
    for name in ("nominal", "quiet"):
        root = BASE / f"PROTO_{name}"
        source = root / "DECISIONS.npz"
        sha = stand.sha(source)
        assert sha == read(root / "RESULT.json")["decision_sha256"]
        with np.load(source, allow_pickle=False) as z:
            inputs, intent, actions = (z[k] for k in ("x", "intent", "action"))
        first = np.asarray([policy(x, a) for x, a in zip(inputs, intent, strict=True)])
        last = np.asarray([policy(x, a) for x, a in zip(inputs[::-1], intent[::-1], strict=True)])[::-1]
        np.testing.assert_array_equal(first, actions)
        np.testing.assert_array_equal(last, actions)
        assert first.tobytes() == actions.tobytes() == last.tobytes()
        archives[name] = dict(source=str(source), sha256=sha, decisions=len(intent),
                              walking=int(intent.sum()), zero_intent=int((intent == 0).sum()), bitwise=True)
    with torch.no_grad():
        model.net[-1].bias[:21].fill_(1000.)
        model.net[-1].bias[21:].fill_(-1000.)
        y = model(torch.as_tensor(inputs[0], dtype=torch.float32)).numpy()
        np.testing.assert_array_equal(abs(y), np.full(42, .15, dtype=np.float32))
    stat = dict(archives=archives, baseline_sources=base_sources(),
                forced_saturation_42_outputs=True, physical_steps=0)
    stand.put(HERE / "ZERO_COMPATIBILITY.json", stat)
    print(stat, flush=True)


if __name__ == "__main__":
    verify()
