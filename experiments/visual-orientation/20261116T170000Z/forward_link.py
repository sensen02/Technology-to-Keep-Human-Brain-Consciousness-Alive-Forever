"""Scoped DNg100 transport and body-side readout on official VNC targets."""
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parents[1] / "neck-real-structure/20261112T225331Z/dnp09_interface"
sys.path.insert(0, str(ARCHIVE))
from source_transport import SourceTransport

BINDING = HERE / "DNg100_BINDING.json"
GRAPH = HERE / "DNg100_GRAPH.json"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@lru_cache(maxsize=1)
def projection(graph_digest):
    assert sha(GRAPH) == graph_digest
    graph = json.loads(GRAPH.read_text())
    roots = graph["roots"]
    counts = defaultdict(lambda: [0, 0])
    for edge in graph["edges"]:
        values = counts[edge["post"]]
        index = roots.index(edge["pre"])
        assert not values[index]
        values[index] = edge["count"]
    return roots, tuple(tuple(count / max(values) for count in values)
                        for _, values in sorted(counts.items()))


class ForwardLink:
    """Only delivered activity can change the current propulsion request."""

    expiry_ms = 65

    def __init__(self):
        self.transport = SourceTransport(BINDING, sha(BINDING))
        self.graph_sha = sha(GRAPH)
        self.root_ids, self.weights = projection(self.graph_sha)
        assert self.root_ids == self.transport.ports["DN"]
        self.latest = (0., 0.)
        self.delivered_step = None
        self.decoded = 0
        self.cut = False

    def switch(self, cut):
        if self.cut != cut:
            self.transport.set_cut("DN", cut)
            self.cut = cut
            self.latest = (0., 0.)
            self.delivered_step = None
            self.decoded = 0

    def send(self, intent):
        if int(intent) not in (0, 1) or int(intent) != intent:
            raise ValueError("DNg100 pilot accepts one propulsion level and zero")
        if not self.cut:
            activity = float(intent)
            self.transport.send(dict(direction="DN", root_ids=self.root_ids,
                                     values=[activity, activity], capture_step=self.transport.step,
                                     source_bindings_sha256=self.transport.digest,
                                     **{"from": "brain", "to": "neck"}))

    def advance(self):
        for packet in self.transport.advance():
            self.latest = tuple(packet["values"])
            self.delivered_step = packet["release_step"]
            left, right = self.latest
            target_activity = [a * left + b * right for a, b in self.weights]
            neutral_boundary = .5 * sum(a + b for a, b in self.weights)
            self.decoded = int(sum(target_activity) > neutral_boundary)

    def action(self):
        if self.delivered_step is None or self.transport.step - self.delivered_step >= self.expiry_ms:
            return 0
        return self.decoded
