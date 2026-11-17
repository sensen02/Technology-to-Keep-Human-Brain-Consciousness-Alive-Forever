"""Source-qualified engineering boundary for bilateral DNa02 activity."""
from pathlib import Path
import sys

from bench_runtime import HERE, PROJECT, turn

ARCHIVE = PROJECT / "experiments/neck-real-structure/20261112T225331Z/dnp09_interface"
sys.path.insert(0, str(ARCHIVE))
from source_transport import SourceTransport

BINDING = HERE / "DNa02_BINDING.json"


class Neck:
    def __init__(self):
        self.transport = SourceTransport(BINDING, turn.stand.sha(BINDING))
        self.ids = self.transport.ports["DN"]
        self.latest = (0., 0.)
        self.cut = False
        self.delivered_step = None

    def switch(self, cut):
        if self.cut != cut:
            self.transport.set_cut("DN", cut)
            if cut:
                self.latest = (0., 0.)
                self.delivered_step = None
            self.cut = cut

    def send(self, yaw):
        if not self.cut:
            self.transport.send({"from": "brain", "to": "neck", "direction": "DN",
                                 "root_ids": self.ids, "values": [float(yaw == 1), float(yaw == -1)],
                                 "capture_step": self.transport.step,
                                 "source_bindings_sha256": self.transport.digest})

    def action(self):
        left, right = self.latest
        return int(left > right) - int(right > left)

    def advance(self):
        for packet in self.transport.advance():
            self.latest = tuple(packet["values"])
            self.delivered_step = packet["release_step"]
