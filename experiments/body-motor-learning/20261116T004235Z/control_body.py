"""Stateless body-side entry point for the six validated movement commands."""
from turning import MODES, load_proto, straight_policy


def load_body_policy():
    straight = straight_policy()
    turns = {mode: load_proto(mode) for mode in MODES}

    def action(observation, advance, yaw):
        if advance not in (0, 1) or yaw not in (-1, 0, 1):
            raise ValueError("Expected advance 0/1 and yaw -1/0/+1")
        if yaw == 0:
            return straight(observation, advance)
        family = "arc" if advance else "pivot"
        side = "left" if yaw > 0 else "right"
        return turns[f"{family}_{side}"](observation)

    return action
