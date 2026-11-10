# Conductance-adaptation lower-bound argument

This is a numerical model-domain result, not evidence that the assumed adaptation reversal is biological. The original −90 mV engineering acceptance floor remains unchanged. The new hypothesis additionally has a derived −80 mV lower bound for its declared input domain.

Write `a=C/dt>0`, `D=a+gL+ge+gi+gA>0`. One active-neuron backward-Euler step before threshold is

`V*=(a V + gL EL + ge EE + gi EI + gA EA + I)/D`.

For the fixed hypothesis, `EL=-65`, `EE=0`, `EI=-70`, `EA=-80` mV, reset/initial voltage −65 mV, and every conductance is nonnegative. If `V>=L=-80` and `I>=0`, then

`V*-L = [a(V-L)+gL(EL-L)+ge(EE-L)+gi(EI-L)+gA(EA-L)+I]/D >= 0`.

Thus an active step cannot go below L in real arithmetic. Exponential decay and the nonnegative per-spike gA increment preserve conductance nonnegativity. Sparse arrivals add nonnegative source-derived conductance increments. Refractory and spike-reset voltage is −65 mV, also above L. Induction from the reset state proves the poststep lower bound. End-step threshold/reset additionally keeps recorded poststep voltage at or below −50 mV; threshold reset is part of the spiking neuron, not an added voltage clip.

Both `LocalCircuit.advance` and `AdaptationConductanceNetwork.step` reject negative and nonfinite current before stepping. The circuit separately rejects any nonzero sensory or descending current outside its own source identity mask. The proof does not apply after arbitrary external edits to `net.v`, receptor/reversal parameters or signed weights outside this domain. No silent projection restores an invalid externally modified state. The implementation checks conductance nonnegativity/finite state and leaves voltage un-clipped.

The numerical check uses tolerance 1e−10 mV on the −80 mV bound to account for floating-point roundoff; it is not an acceptance-band relaxation. Poststep finite state is also checked on every executed step. Finite-input validation alone does not prove absence of intermediate overflow under arbitrarily huge finite currents; the real-arithmetic argument assumes finite numerical operations, and postspike reset can hide a pre-reset overflow. Actual archived currents remain bounded qualified source inputs, and independent reference checks exercise their finite arithmetic. Arbitrarily large finite injection is outside this tested engineering operating range.

The per-spike increment `.002nA/15mV=.00013333333333333334uS` reproduces the prior outward current increment only at `V=-65mV`, because `gA*(-80-(-65))=-.002nA`. At other voltages the mechanism differs intentionally; it opposes depolarization and reverses below its assumed reversal instead of applying a voltage-independent negative current. The reversal, time constant and matching rule are explicit illustrative hypotheses, not receptor/electrophysiological measurements.
