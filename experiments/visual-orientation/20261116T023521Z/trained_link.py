"""Connect the trained endpoints through actual DNa02 packet delivery."""
from codec import CONTRACT, RuntimeCodec, digest
from neck_runtime import Neck


class TrainedLink(Neck):
    expiry_ms = 65

    def __init__(self, checkpoint, profile, seed):
        super().__init__()
        self.codec = RuntimeCodec(checkpoint, profile, seed)
        self.checkpoint = checkpoint
        self.profile = profile
        self.seed = seed
        self.decoded = 0
        self.measurements = []
        self.last_valid = None
        self.graph_sha = digest(CONTRACT).hexdigest()

    def switch(self, cut):
        super().switch(cut)
        if cut:
            self.decoded = 0
            self.last_valid = None

    def send(self, yaw):
        if self.cut:
            return
        self.transport.send({"from": "brain", "to": "neck", "direction": "DN",
                             "root_ids": self.ids, "values": self.codec.send(yaw),
                             "capture_step": self.transport.step,
                             "source_bindings_sha256": self.transport.digest})

    def action(self):
        if self.last_valid is None or self.transport.step-self.last_valid >= self.expiry_ms:
            return 0
        return self.decoded

    def advance(self):
        for packet in self.transport.advance():
            self.latest = tuple(packet["values"])
            self.delivered_step = packet["release_step"]
            decoded, values = self.codec.read(self.latest)
            self.measurements.append((self.delivered_step, decoded, values))
            if decoded is not None:
                self.last_valid = self.delivered_step
                self.decoded = decoded
