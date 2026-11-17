"""Signed reference retrieval with a previously qualified moving-turn fallback."""
from policy_directional import make


def load_policy(checkpoint):
    return make(checkpoint,use_primitives=True)
