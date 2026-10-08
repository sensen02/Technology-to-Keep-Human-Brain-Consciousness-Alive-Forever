"""Fixed-geometry recording diagnostic. One-sided PSDs in V²/Hz.
Analog anti-aliasing precedes ideal sampling; causal digital SOS is NOT anti-alias.
No measured amplifier parameters or biological-current claims.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy import signal
from .electrode_frontend import ContactGeometry, InterfaceImpedance, BOLTZMANN_J_PER_K, contact_area_average

BANDS = {"spike": (300., 5000.), "lfp": (.5, 300.), "wide": (.1, 10000.)}

def positive(name, value, zero=False):
    value = float(value)
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return value

def shared_reference_covariance(channel_psds, reference_psd):
    """Independent channel minus common reference: diag(Si)+Sref*11T.

    Standalone algebra helper, not connected to the single-channel budget.
    PSD units must match; last channel_psds dimension enumerates channels.
    Assumes independent reference and channels, unity reference transfer.
    """
    s = np.asarray(channel_psds, dtype=float)
    r = np.asarray(reference_psd, dtype=float)
    if s.ndim < 1 or s.shape[-1] == 0 or not np.isfinite(s).all() or np.any(s < 0):
        raise ValueError("channel PSDs must be nonempty, finite and nonnegative")
    if not np.isfinite(r).all() or np.any(r < 0):
        raise ValueError("reference PSD must be finite and nonnegative")
    try:
        r = np.broadcast_to(r, s.shape[:-1])
    except ValueError as exc:
        raise ValueError("reference PSD shape incompatible with frequency axes") from exc
    n = s.shape[-1]
    return s[..., :, None]*np.eye(n) + r[..., None, None]*np.ones((n,n))

@dataclass(frozen=True)
class RecordingConfig:
    diameter_um: float = 7.
    pitch_um: float = 20.
    fs_Hz: float = 30000.
    band: str = "spike"
    hp_order: int = 2
    lp_order: int = 4
    digital_order: int = 2
    voltage_nV_rtHz: float = 10.
    current_fA_rtHz: float = 10.
    voltage_1f_corner_Hz: float = 100.
    current_1f_corner_Hz: float = 100.

    def __post_init__(self):
        if self.diameter_um != 7. or self.pitch_um != 20.:
            raise ValueError("fixed diameter 7 um, pitch 20 um")
        if self.band not in BANDS:
            raise ValueError("unknown band")
        positive("fs_Hz", self.fs_Hz)
        if self.fs_Hz < 30000 or BANDS[self.band][1] >= self.fs_Hz/2:
            raise ValueError("sampling >=30kHz and Nyquist above passband required")
        for name, required in (("hp_order", 2), ("lp_order", 4), ("digital_order", 2)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value != required:
                raise ValueError(f"{name} must be integer {required}")
        for name in ("voltage_nV_rtHz", "current_fA_rtHz", "voltage_1f_corner_Hz", "current_1f_corner_Hz"):
            positive(name, getattr(self, name), zero=True)

class RecordingChain:
    def __init__(self, config=RecordingConfig(), interface=None):
        self.config = config
        self.contact = ContactGeometry.from_diameter(config.diameter_um)
        self.interface = interface or InterfaceImpedance(self.contact, amplifier_input_ohm=1e9)
        if not isinstance(self.interface, InterfaceImpedance):
            raise TypeError("interface must be InterfaceImpedance")
        if self.interface.contact != self.contact:
            raise ValueError("interface must match fixed bare 7um contact")
        lo, hi = BANDS[config.band]
        self.hp_zpk = signal.butter(2, 2*np.pi*lo, btype="highpass", analog=True, output="zpk")
        self.lp_zpk = signal.butter(4, 2*np.pi*hi, btype="lowpass", analog=True, output="zpk")
        self.digital_sos = signal.butter(2, (lo, hi), btype="bandpass", fs=config.fs_Hz, output="sos")

    def analog_response(self, f):
        f = np.asarray(f, dtype=float)
        if not np.isfinite(f).all() or np.any(f < 0):
            raise ValueError("frequencies must be finite and nonnegative")
        return signal.freqs_zpk(*self.hp_zpk, worN=2*np.pi*f)[1]*signal.freqs_zpk(*self.lp_zpk, worN=2*np.pi*f)[1]

    def digital_response(self, f):
        f = np.asarray(f, dtype=float)
        if not np.isfinite(f).all() or np.any(f < 0):
            raise ValueError("frequencies must be finite and nonnegative")
        folded = np.abs((f+self.config.fs_Hz/2)%self.config.fs_Hz-self.config.fs_Hz/2)
        return signal.sosfreqz(self.digital_sos, worN=folded, fs=self.config.fs_Hz)[1]

    def input_psds(self, f):
        f = np.asarray(f, dtype=float)
        ze = self.interface.impedance_ohm(f)
        rin = self.interface.amplifier_input_ohm
        h = self.interface.transfer(f)
        zl = ze if math.isinf(rin) else 1/(1/ze+1/rin)
        kt4 = 4*BOLTZMANN_J_PER_K*self.interface.temperature_K
        c = self.config
        return {"contact_thermal": kt4*ze.real*np.abs(h)**2,
                "input_resistor_thermal": np.zeros_like(f) if math.isinf(rin) else kt4/rin*np.abs(zl)**2,
                "amplifier_voltage": (c.voltage_nV_rtHz*1e-9)**2*(1+c.voltage_1f_corner_Hz/f),
                "amplifier_current": (c.current_fA_rtHz*1e-15)**2*(1+c.current_1f_corner_Hz/f)*np.abs(zl)**2}

    def budget(self, points_per_decade=1200, f_min=1e-6, f_max=3e6):
        """Integrate analog tails and digital-folded aliases using one-sided PSD.
        ADC sampling is ideal and analytic; no quantization, jitter or clipping.
        """
        positive("f_min", f_min); positive("f_max", f_max)
        nyq = self.config.fs_Hz/2
        if f_max <= max(f_min, nyq):
            raise ValueError("integration bound must exceed Nyquist")
        if isinstance(points_per_decade, bool) or not isinstance(points_per_decade, int) or points_per_decade < 50:
            raise ValueError("points_per_decade must be integer >=50")
        n = int(np.ceil(np.log10(f_max/f_min)*points_per_decade))+1
        f = np.unique(np.r_[np.geomspace(f_min, f_max, n), nyq])
        a2 = np.abs(self.analog_response(f))**2
        d2 = np.abs(self.digital_response(f))**2
        psds = self.input_psds(f)
        def integ(y, mask=None):
            return float(np.trapezoid(y if mask is None else y[mask], f if mask is None else f[mask]))
        components = {k: integ(v*a2*d2) for k, v in psds.items()}
        total = sum(psds.values())*a2
        base = f <= nyq; alias = f >= nyq
        bv = integ(total*d2, base); av = integ(total*d2, alias)
        pb = integ(total, base); pa = integ(total, alias)
        return {"components_V2": components, "total_V2": sum(components.values()),
                "noise_rms_uV": math.sqrt(sum(components.values()))*1e6,
                "baseband_V2": bv, "alias_V2": av, "alias_fraction_of_baseband": av/bv,
                "pre_ADC_baseband_V2": pb, "pre_ADC_above_Nyquist_V2": pa,
                "pre_ADC_alias_fraction": pa/pb,
                "f_min_Hz": f_min, "f_max_Hz": f_max,
                "points_per_decade": points_per_decade, "integration_points": len(f)}

    def demo(self, current_nA=.1, duration_s=2., oversample=8):
        """Charge-balanced source/return toy; causal zero-state signal.
        RMS covers full duration including silence/tails. Noise comparison is
        stationary expected RMS over that same window, not a peak comparison.
        Time-domain analog uses high-rate bilinear approximation; PSD is exact.
        """
        positive("current_nA", current_nA, zero=True); positive("duration_s", duration_s)
        if duration_s < .1:
            raise ValueError("duration >=0.1s required")
        if isinstance(oversample, bool) or not isinstance(oversample, int) or oversample < 4:
            raise ValueError("oversample must be integer >=4")
        fs = self.config.fs_Hz*oversample
        t = np.arange(round(duration_s*fs))/fs
        pulse = np.exp(-.5*((t-.03)/.0002)**2)-np.exp(-.5*((t-.0306)/.0002)**2)
        pulse -= pulse.mean()
        pulse /= np.max(np.abs(pulse))
        source = np.array([0., 0., 20.]); ret = np.array([0., 0., 120.])
        sigma = self.interface.conductivity_S_m
        def field(x):
            return (1/np.linalg.norm(x-source, axis=1)-1/np.linalg.norm(x-ret, axis=1))*1e6/(4*np.pi*sigma)
        ohm = float(contact_area_average(field, self.contact))
        raw = current_nA*1e-9*ohm*pulse
        r = self.interface.amplifier_input_ohm
        if math.isinf(r):
            loaded = raw
        else:
            rc = self.interface.r_ct_ohm; rs = self.interface.r_spread_ohm; tau = rc*self.interface.c_dl_F
            b, a = signal.bilinear([r*tau, r], [(r+rs)*tau, r+rs+rc], fs=fs)
            loaded = signal.lfilter(b, a, raw)
        analog = loaded
        for zpk in (self.hp_zpk, self.lp_zpk):
            analog = signal.sosfilt(signal.zpk2sos(*signal.bilinear_zpk(*zpk, fs)), analog)
        sampled = analog[::oversample]
        digital = signal.sosfilt(self.digital_sos, sampled)
        return {"time_s": t[::oversample], "raw_V": raw[::oversample], "analog_V": sampled,
                "digital_V": digital, "signal_rms_uV": float(np.sqrt(np.mean(digital**2))*1e6),
                "duration_s": duration_s, "toy_peak_current_nA": current_nA,
                "discrete_net_charge_C": float(np.sum(current_nA*1e-9*pulse)/fs),
                "source_return_transfer_ohm": ohm, "analog_demo_oversample": oversample}
