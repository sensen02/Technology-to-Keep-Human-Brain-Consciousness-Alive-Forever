"""Train both endpoints through a frozen projection and test held distortion profiles."""
from pathlib import Path
import json

import torch
import torch.nn.functional as F

from codec import CONTRACT, PROFILES, Codec, digest, graph, measurement


HERE = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(1)
    torch.manual_seed(743)
    rng = torch.Generator().manual_seed(195150)
    fixed = graph()
    model = Codec(fixed.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=.004)
    journal = []
    expected = torch.tensor([[.07, .90], [.04,.04], [.90,.07]])
    anchor_labels = torch.arange(3)
    checkpoint = HERE / "CODEC.pt"
    for step in range(401):
        if step:
            opt.zero_grad(set_to_none=True)
            labels = torch.randint(3, (240,), generator=rng)
            activations = model.encode(labels)
            sampled = measurement(activations, fixed, "combined" if step % 3 else "clean", rng)
            loss = F.cross_entropy(model.decode(sampled), labels) + .45 * F.mse_loss(model.encode(anchor_labels), expected)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            opt.step()
            journal.append(float(loss.detach()))
    with checkpoint.open("xb") as sink:
        torch.save(dict(weights=model.state_dict(), graph_sha256=digest(CONTRACT).hexdigest()), sink)
    restored = Codec(fixed.shape[1]).eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True)["weights"])
    for parameter, saved in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(parameter, saved, rtol=0., atol=0.)
    test = {}
    with torch.inference_mode():
        labels = anchor_labels.repeat_interleave(2048)
        signatures = restored.encode(anchor_labels)
        for i, condition in enumerate(PROFILES):
            rng = torch.Generator().manual_seed(887720 + i)
            raw = restored.encode(labels)
            target = measurement(raw, fixed, condition, rng)
            predicted = restored.decode(target).argmax(1)
            confusion = torch.bincount(labels * 3 + predicted, minlength=9).reshape(3, 3)
            test[condition] = dict(confusion=confusion.tolist(), accuracy=float(predicted.eq(labels).float().mean()),
                                   per_class=(confusion.diag()/2048).tolist())
        cut = torch.zeros((1, fixed.shape[1]))
        assert restored.decode(cut).argmax(1).item() == 1
    summary = dict(graph_sha256=digest(CONTRACT).hexdigest(), checkpoint_sha256=digest(checkpoint).hexdigest(),
                   steps=len(journal), loss=journal, signatures=signatures.tolist(), test=test,
                   clean_pass=min(test["clean"]["per_class"])>=.99, combined_pass=min(test["combined"]["per_class"])>=.95,
                   packet_loss_test="Stateful dropout and expiry run in separate physical/control replay.",
                   tensor_replay_bitwise=True)
    with (HERE / "CODEC_RESULT.json").open("x") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")
    print({k:v["accuracy"] for k,v in test.items()})
    print(signatures.tolist())


if __name__ == "__main__":
    main()
