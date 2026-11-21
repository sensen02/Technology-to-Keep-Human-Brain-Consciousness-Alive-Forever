"""Restricted diagonal rate encoder: only causal post-neck channel features."""
import torch


class AscendingRateAdapter(torch.nn.Module):
    def __init__(self, channels, rate_cap_hz=300.):
        super().__init__()
        self.mix_logits = torch.nn.Parameter(torch.zeros(channels, 3, dtype=torch.float64))
        self.log_gain = torch.nn.Parameter(torch.zeros(channels, dtype=torch.float64))
        self.rate_cap_hz = float(rate_cap_hz)

    def forward(self, features_hz, cut_mask):
        if features_hz.shape[-2:] != self.mix_logits.shape or cut_mask.shape != (self.mix_logits.shape[0],):
            raise ValueError('post-neck channel order/shape mismatch')
        if cut_mask.dtype != torch.bool or not torch.isfinite(features_hz).all() or (features_hz < 0).any():
            raise ValueError('expected finite nonnegative causal rates and explicit channel cuts')
        level = (features_hz*self.mix_logits.softmax(-1)).sum(-1)
        gain = self.log_gain.clamp(-2., 2.).exp()
        rate = self.rate_cap_hz*(-torch.expm1(-gain*level/self.rate_cap_hz))
        return rate.masked_fill(cut_mask, 0.)
