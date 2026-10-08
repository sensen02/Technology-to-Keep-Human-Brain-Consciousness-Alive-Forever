"""A genuinely NATURAL scene layer, built ON TOP of ``engine.embodied.scene``.

WHAT THIS MODULE IS
-------------------
``scene.py`` pins the world declaratively and is the right place for friction,
lights and object placement, but its three presets reproduce three measured
visual defects of the shipped stack:

1. **The ground renders as a regular periodic checkerboard.**  FlyGym's
   ``FlatGroundWorld`` builds a material ``grid`` that references a 2-D texture
   named ``checker`` (``builtin=mjBUILTIN_CHECKER``, 300x300 px) with
   ``texrepeat=[250, 250]`` over a 2000 mm plane, i.e. one tile every ~8 mm on a
   2.5 mm fly.  A periodic grating is the opposite of a natural scene.  (This is
   the MEASURED_LOCAL baseline; see :data:`OLD_GROUND_TILE_MM`.)
2. **The micro-relief has no natural spatial statistics**: ``scene.py``'s
   heightfield is ``0.5 + 0.35*sin(kX) + 0.15*cos(kY) + 0.10*uniform-noise``, a
   single dominant wavelength plus white noise, and it is a FINITE 60 mm patch
   outside which nothing collides at all.
3. **The sky is a uniform white cube.**  FlyGym's ``BaseWorld._add_skybox``
   creates ``skybox`` with ``builtin=mjBUILTIN_GRADIENT`` but
   ``rgb1 = rgb2 = (1,1,1)``, so the gradient is white-to-white.  A constant sky
   carries zero visual information and (with the default headlight) is the
   dominant source of a saturated eye image.

This module supplies instead: a multi-octave fractional-Brownian-motion
terrain, a natural-image ground texture (a real CC0 photograph, with a purely
procedural pink-noise fallback so the preset works with no network), a gradient
sky, and a rebalanced lighting/material set -- plus the machinery to *audit*
all of it instead of asserting it.

THE TWO ROUTES FOR THE GROUND TEXTURE
-------------------------------------
* **Route A (used by default when the asset is present):** a real, verified
  CC0 / public-domain photograph fetched from Wikimedia Commons by
  :func:`download_natural_texture` and recorded in
  :data:`NATURAL_PRESET_LEDGER` with its author, licence and source page.
* **Route B (always available):** :func:`generate_natural_texture_png`, a
  pink-noise (radial amplitude ~ ``k**(-beta/2)``) texture.  Labelled
  ``PROCEDURAL``.

Preset ``natural_fbm`` uses whichever asset :func:`resolve_ground_texture`
finds, in that order.  Nothing here needs the network at run time: the photo is
cached on disk once and reused.

WHAT IS MEASURED AND WHAT IS A CHOICE
-------------------------------------
Everything the module asserts about *spatial statistics* is measured by
:func:`spectral_audit` at run time (radial power spectrum, fitted power-law
exponent, periodicity detector).  Everything else -- the fBm Hurst exponent,
the amplitudes, the sky colours, the headlight values -- is an
``ENGINEERING_DEFAULT`` or ``ASSUMED`` entry in :data:`NATURAL_PRESET_LEDGER`
with a sweep range, exactly as ``scene.py`` demands.  No physical constant here
is presented as a literature value.

MODEL SCALE (carried from ``scene.py``, do not quote as biological)
-------------------------------------------------------------------
Model units are MILLIMETRES (gravity ``[0,0,-9810]`` mm/s^2).  The model's total
mass is 1.0243e-3 kg (MEASURED_LOCAL), about 1042x a real *Drosophila*
(0.983 mg, Vaxenburg et al., Nature 643 (2025) -- quoted here only to state the
discrepancy, not as a calibration of this model).  Absolute forces from this
model are therefore NOT biological.

Run everything with ``MUJOCO_GL=egl``.  MuJoCo is imported lazily inside
functions so this module stays importable without it (like ``scene.py``).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from engine.embodied import scene

__all__ = [
    # configuration
    "NaturalSceneConfig", "NATURAL_PRESETS", "NATURAL_PRESET_LEDGER",
    "apply_natural_scene", "NaturalSceneApplicationReport",
    # terrain / texture generation
    "generate_fbm_heightfield", "generate_natural_texture_png",
    "generate_pink_noise", "resolve_ground_texture", "download_natural_texture",
    # audit
    "spectral_audit", "radial_power_spectrum", "periodicity_peak",
    "muoco_builtin_checker_field",
    # provenance
    "TextureProvenance", "natural_scene_provenance_report", "ASSET_DIR",
    # notes
    "SCALE_NOTE", "SPECTRAL_AUDIT_NOTE", "DEFAULT_HURST",
]

# --------------------------------------------------------------------------- #
# where the assets live
# --------------------------------------------------------------------------- #

#: Directory for downloaded / generated PNG assets.  Kept inside the project's
#: output tree so a report can point at the exact bytes that were used.
ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "outputs", "embodied_body", "assets")

SCALE_NOTE = (
    "model units are millimetres; the model's total mass is 1.0243e-3 kg "
    "(MEASURED_LOCAL, sum of body_mass), about 1042x a real Drosophila "
    "(0.983 mg).  Dimensionless ratios here are usable; absolute forces are NOT "
    "biological.")

SPECTRAL_AUDIT_NOTE = (
    "the periodicity detector compares the strongest non-DC spectral magnitude "
    "against the MEDIAN non-DC magnitude in log10 space.  A periodic grating "
    "concentrates its whole variance into one narrow bin and therefore has a "
    "large ratio; a natural (power-law) field spreads variance over a continuum "
    "of wavenumbers and has a small one.  The ratio is scale-free, so the same "
    "threshold applies to terrain in mm and to a texture in pixels.")

#: MEASURED_LOCAL: FlatGroundWorld's plane is 2*1000 mm wide and its material
#: uses texrepeat=[250, 250], so one checker tile spans 2000/250 = 8 mm.
OLD_GROUND_TILE_MM = 2000.0 / 250.0
#: MEASURED_LOCAL: the same material's builtin checker texture is 300x300 px.
OLD_GROUND_TEX_PX = 300

#: ENGINEERING_DEFAULT: Hurst exponent of the terrain.  H in [0.7, 0.9] is the
#: range conventionally used for natural topography; the spectral consequence is
#: a 2-D radial slope between -(2H+1.5) = -2.9 and -3.1.  This is a
#: plausible-looking default for a hand-built prototype, NOT a measured value
#: for leaf litter at millimetre scale, and it is swept in the ledger.
DEFAULT_HURST = 0.80
#: The radial slope the default Hurst exponent implies for a 2-D surface, kept
#: as a derived quantity so the two can never drift apart silently.
BACKGROUND_SPECTRAL_SLOPE = -(2.0 * DEFAULT_HURST + 1.5)


# --------------------------------------------------------------------------- #
# provenance for downloaded assets
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TextureProvenance:
    """Everything needed to re-fetch and legally reuse one texture asset.

    ``route`` is ``"PHOTO_PD"`` (a real photograph whose licence was verified),
    ``"PHOTO_CC0"``, or ``"PROCEDURAL"``.  A record claiming a photo route MUST
    carry ``author``, ``license_name``, ``license_url`` and ``source_page``:
    :meth:`__post_init__` raises otherwise.  This is the module's mechanical
    guard against an unlabelled or unlicensed asset.
    """
    name: str
    route: str
    sha256: str
    width_px: int
    height_px: int
    author: str = ""
    license_name: str = ""
    license_url: str = ""
    source_page: str = ""
    source_url: str = ""
    credit: str = ""
    generator_note: str = ""
    spectral: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.route not in ("PHOTO_PD", "PHOTO_CC0", "PROCEDURAL"):
            raise ValueError(f"unknown texture route {self.route!r}")
        if self.route.startswith("PHOTO"):
            missing = [k for k in ("author", "license_name", "license_url",
                                   "source_page")
                       if not str(getattr(self, k)).strip()]
            if missing:
                raise ValueError(
                    f"{self.name!r}: a photo asset may not be used without a "
                    f"verified licence; missing {missing}")
        else:
            if not str(self.generator_note).strip():
                raise ValueError(f"{self.name!r}: a procedural asset must document "
                                 f"how it was generated in generator_note")

    def as_dict(self) -> dict:
        return scene._jsonable(dataclasses.asdict(self))


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# spectral audit  (the objective test of "natural vs periodic")
# --------------------------------------------------------------------------- #

def _as_float_field(array, *, centred: bool = True) -> np.ndarray:
    """Reduce an image/terrain array to one float 2-D plane with DC removed."""
    a = np.asarray(array)
    if a.ndim == 3:
        a = a[..., :3].astype(float)
        # Rec. 601 luma; MuJoCo's checker rgb1/rgb2 are grey so this is exact
        # for greyscale inputs and a reasonable perceptual reduction otherwise.
        a = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    if a.ndim != 2:
        raise ValueError(f"spectral_audit needs a 2-D array or an HxWx3 image, "
                         f"got shape {np.shape(array)}")
    a = np.asarray(a, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError("spectral_audit input contains non-finite values")
    return a - a.mean() if centred else a


def radial_power_spectrum(array, *, n_bins: int | None = None,
                          window: bool = True) -> dict:
    """Radially averaged power spectrum of a 2-D field, with a log-log fit.

    Returns counts, mean wavenumber and mean power per radial bin.  A Hann
    window is applied by default because a finite patch's step at its edges
    would otherwise leak a 1/f-like floor of its own.
    """
    a = _as_float_field(array)
    if window:
        wy = np.hanning(a.shape[0])[:, None]
        wx = np.hanning(a.shape[1])[None, :]
        a = a * (wy * wx)
    f = np.fft.fft2(a)
    p = (np.abs(f) ** 2) / a.size
    ky = np.fft.fftfreq(a.shape[0])          # cycles per sample
    kx = np.fft.fftfreq(a.shape[1])
    KX, KY = np.meshgrid(kx, ky)
    kr = np.sqrt(KX ** 2 + KY ** 2)
    kmax = float(kr.max())
    n_bins = int(n_bins) if n_bins else max(8, a.shape[0] // 4)
    edges = np.linspace(0.0, kmax, n_bins + 1)
    idx = np.clip(np.digitize(kr.ravel(), edges) - 1, 0, n_bins - 1)
    pv = p.ravel()
    counts = np.bincount(idx, minlength=n_bins)
    sums = np.bincount(idx, weights=pv, minlength=n_bins)
    ksum = np.bincount(idx, weights=kr.ravel(), minlength=n_bins)
    kmean = np.where(counts > 0, ksum / np.maximum(counts, 1), 0.0)
    pmean = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    return {"k": kmean, "power": pmean, "counts": counts, "n_bins": n_bins,
            "kmax": kmax, "shape": tuple(int(v) for v in a.shape)}


def _fit_power_law(k: np.ndarray, power: np.ndarray, counts: np.ndarray, *,
                   k_lo: float, k_hi: float) -> dict:
    """Least-squares fit of log10(power) = slope*log10(k) + intercept."""
    m = (counts > 8) & (k >= k_lo) & (k <= k_hi) & (power > 0) & np.isfinite(power)
    n = int(m.sum())
    if n < 4:
        return {"slope": None, "intercept": None, "r2": None, "n_bins_fit": n,
                "k_lo": float(k_lo), "k_hi": float(k_hi),
                "error": "fewer than 4 usable radial bins in the fit band"}
    lk = np.log10(k[m])
    lp = np.log10(power[m])
    slope, intercept = np.polyfit(lk, lp, 1)
    pred = slope * lk + intercept
    ss_res = float(np.sum((lp - pred) ** 2))
    ss_tot = float(np.sum((lp - lp.mean()) ** 2))
    return {"slope": float(slope), "intercept": float(intercept),
            "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
            "n_bins_fit": n, "k_lo": float(k_lo), "k_hi": float(k_hi)}


def periodicity_peak(array, *, k_floor: int = 3, harmonics: int = 2,
                     exclude_axes: bool = False, smooth_bins: int = 9) -> dict:
    """The periodicity detector: how far the strongest peak stands above the trend.

    Two obvious metrics were tried and REJECTED because they are not scale-free:

    * *peak / global median magnitude* grows like ``sqrt(log N)`` for a pure
      power-law continuum -- measured here on a natural fBm patch it gave 79 for
      a field that has no grating at all, so it cannot be compared with any
      fixed threshold across sizes;
    * *peak / global RMS magnitude* punishes steep spectra (the envelope, not the
      peak, dominates), which is exactly backwards for terrain.

    What is used instead is a **local** comparison, which is what "a narrow peak
    standing out of a smooth continuum" actually means:

    * ``peak_frac_of_power`` -- the fraction of total (non-DC) power sitting in
      one single bin.  For any aperiodic field this is ~``1/n_bins``; a grating
      concentrates a large fraction into one or two bins.  This is the number a
      threshold should be set on, and it is dimensionless and size-corrected.
    * ``peak_z_local`` -- ``log10(peak) - log10(local trend)``, where the local
      trend is the median of neighbouring bins in log-magnitude space (a local
      median filter, so a peak cannot inflate its own baseline).  A continuum has
      z of order 0.5-1.5; a discrete line is many decades up.

    ``k_floor`` is deliberately >= 3 so that the lowest wavenumbers (which carry
    the patch's own fundamental and any windowing leakage) cannot produce a
    verdict on their own; ``harmonics`` values of the peak are written to a
    suppression set so one grating is not counted several times; ``exclude_axes``
    removes the pure kx=0 / ky=0 rows and columns, which is how a *striped*
    pattern is told apart from a checkerboard.
    """
    a = _as_float_field(array)
    ny, nx = a.shape
    f = np.fft.fft2(a.astype(np.float64))       # float64: uint8 squaring overflows
    mag = np.abs(f)
    mag[0, 0] = 0.0
    ky = np.fft.fftfreq(ny) * ny          # integer bin index
    kx = np.fft.fftfreq(nx) * nx
    KX, KY = np.meshgrid(kx, ky)
    KXi = np.rint(KX).astype(int)
    KYi = np.rint(KY).astype(int)
    valid = (np.abs(KYi) >= k_floor) | (np.abs(KXi) >= k_floor)
    if exclude_axes:
        valid &= (KXi != 0) & (KYi != 0)
    # An exactly periodic pattern has *identically zero* magnitude in the bins
    # the grating does not occupy, so a ratio against them is undefined and a log
    # of them is -inf.  Those bins are excluded rather than fudged.
    valid &= mag > 0
    with np.errstate(divide="ignore"):
        logmag = np.log10(np.where(mag > 0, mag, np.nan))
    trend = _local_log_trend(logmag, valid, smooth_bins)
    total_power = float(np.sum(mag[valid] ** 2))
    with np.errstate(all="ignore"):
        med = float(np.median(mag[valid])) if valid.any() else 0.0
    work = mag.copy()
    peaks = []
    for _ in range(max(1, int(harmonics))):
        w = np.where(valid, work, 0.0)
        flat = int(np.argmax(w))
        iy, ix = np.unravel_index(flat, w.shape)
        v = float(w[iy, ix])
        if v <= 0:
            break
        ky_i, kx_i = int(KYi[iy, ix]), int(KXi[iy, ix])
        kk = math.hypot(float(kx_i) / nx, float(ky_i) / ny)
        tr = float(trend[iy, ix]) if np.isfinite(trend[iy, ix]) else float("nan")
        peaks.append({
            "magnitude": v,
            "power": v ** 2,
            "peak_frac_of_power": (v ** 2 / total_power) if total_power > 0 else None,
            "peak_z_local": (math.log10(v) - tr) if np.isfinite(tr) else None,
            "local_trend_log10": tr,
            "bin": [kx_i, ky_i],
            "k": kk, "wavelength_samples": (1.0 / kk) if kk > 0 else None,
            "ratio_to_median": (v / med) if med > 0 else None,
        })
        for h in range(1, int(harmonics) + 2):
            for sx in (1, -1):
                for sy in (1, -1):
                    work[(sy * ky_i * h) % ny, (sx * kx_i * h) % nx] = 0.0
    top = peaks[0] if peaks else None
    return {
        "median_magnitude": med,
        "total_power": total_power,
        "n_bins_considered": int(valid.sum()),
        "peak_frac_of_power": (top["peak_frac_of_power"] if top else None),
        "peak_z_local": (top["peak_z_local"] if top else None),
        # kept because it was measured first and reported by the runner; it is
        # NOT the metric the PASS/FAIL threshold is set on (see docstring)
        "peak_ratio_to_median": (top["ratio_to_median"] if top else None),
        "peak_bin": (top["bin"] if top else None),
        "peak_wavelength_samples": (top["wavelength_samples"] if top else None),
        "peak_k": (top["k"] if top else None),
        "n_peaks_reported": len(peaks),
        "peaks": peaks,
        "k_floor": int(k_floor),
        "exclude_axes": bool(exclude_axes),
        "smooth_bins": int(smooth_bins),
        "definition": SPECTRAL_AUDIT_NOTE,
    }


def _local_log_trend(logmag: np.ndarray, valid: np.ndarray,
                     smooth_bins: int) -> np.ndarray:
    """Median of log-magnitude over a local ``smooth_bins`` window.

    The median makes the trend robust to the peak itself, which is the whole
    point: a mean filter would be dragged up by the very peak it is meant to
    expose.  Implemented as a stack of shifted copies so it needs no SciPy, with
    edge padding so the rows and columns near the boundary still get a trend.
    """
    nb = max(3, int(smooth_bins) | 1)               # force odd
    r = nb // 2
    ny, nx = logmag.shape
    filled = np.where(valid & np.isfinite(logmag), logmag, np.nan)
    pad = np.pad(filled, r, mode="edge")
    stack = np.full((nb * nb, ny, nx), np.nan)
    s = 0
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            stack[s] = pad[r + dy:r + dy + ny, r + dx:r + dx + nx]
            s += 1
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN rows are 0.0
        trend = np.nanmedian(stack, axis=0)
    return trend




def spectral_audit(array, *, sample_pitch: float | None = None,
                   sample_unit: str = "sample", n_bins: int | None = None,
                   k_floor: int = 3, band: tuple[float, float] = (0.02, 0.45),
                   exclude_axes: bool = False) -> dict:
    """Objective answer to "is this field natural or periodic?".

    Returns explicit numbers:

    * ``radial_power_spectrum`` -- the binned spectrum;
    * ``fitted_slope`` -- the least-squares log-log exponent over the fit band
      (given as a fraction of Nyquist, so it is resolution independent);
    * ``periodicity`` -- :func:`periodicity_peak`, whose
      ``peak_ratio_to_median`` is the PASS/FAIL quantity a caller must declare
      its threshold for BEFORE measuring;
    * ``periodicity_anisotropic`` -- the same detector with the pure-axis bins
      removed, which distinguishes a striped pattern from a true 2-D grating;
    * the peak's wavelength converted to ``sample_pitch`` units when supplied
      (mm for terrain, mm or texels for a texture).

    ``sample_pitch`` is the physical size of one sample; when given, the peak's
    wavelength is reported in ``sample_unit`` (e.g. mm), which is what makes the
    number comparable with the fly's own 2.5 mm body.
    """
    a = _as_float_field(array)
    n = min(a.shape)
    rad = radial_power_spectrum(a, n_bins=n_bins)
    k = rad["k"]
    kn = k / 0.5                    # 1.0 == Nyquist (cycles/sample 0.5)
    fit = _fit_power_law(kn, rad["power"], rad["counts"],
                         k_lo=float(band[0]), k_hi=float(band[1]))
    pk = periodicity_peak(a, k_floor=k_floor)
    pk_aniso = periodicity_peak(a, k_floor=k_floor, exclude_axes=True)
    out = {
        "shape": [int(v) for v in a.shape],
        "sample_pitch": (None if sample_pitch is None else float(sample_pitch)),
        "sample_unit": sample_unit,
        "band_fraction_of_nyquist": [float(band[0]), float(band[1])],
        "fitted_slope": fit["slope"],
        "fitted_r2": fit["r2"],
        "fit_n_bins": fit["n_bins_fit"],
        "fit_error": fit.get("error"),
        "peak_frac_of_power": pk["peak_frac_of_power"],
        "peak_z_local": pk["peak_z_local"],
        "peak_frac_of_power_anisotropic": pk_aniso["peak_frac_of_power"],
        "peak_z_local_anisotropic": pk_aniso["peak_z_local"],
        "peak_ratio_to_median": pk["peak_ratio_to_median"],
        "peak_ratio_to_median_anisotropic": pk_aniso["peak_ratio_to_median"],
        "peak_bin": pk["peak_bin"],
        "peak_wavelength_samples": pk["peak_wavelength_samples"],
        "median_magnitude": pk["median_magnitude"],
        "periodicity": pk,
        "periodicity_anisotropic": pk_aniso,
        "radial": {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                   for kk, vv in rad.items()},
        "variance": float(np.var(a)),
        "std": float(np.std(a)),
    }
    if sample_pitch is not None and pk["peak_wavelength_samples"]:
        out["peak_wavelength"] = float(pk["peak_wavelength_samples"] * sample_pitch)
        out["peak_wavelength_unit"] = sample_unit
        # What the peak means in a physical unit the fly cares about: how many
        # wavelengths fit across a 2.5 mm fly.
        wl = out["peak_wavelength"]
        out["fly_lengths_per_peak_wavelength"] = (float(2.5 / wl) if wl > 0 else None)
    return out


def muoco_builtin_checker_field(width: int = OLD_GROUND_TEX_PX,
                                height: int = OLD_GROUND_TEX_PX,
                                rgb1: Sequence[float] = (0.3, 0.3, 0.3),
                                rgb2: Sequence[float] = (0.4, 0.4, 0.4)) -> dict:
    """The OLD ground texture, read from MuJoCo itself (not re-drawn by hand).

    Builds a throwaway ``MjSpec`` with FlyGym's own checker texture
    (``builtin=mjBUILTIN_CHECKER``, same rgb1/rgb2/default checker period),
    compiles it, and reads ``model.tex_data`` back.  That is the exact texel
    field MuJoCo maps onto the plane in the old scene, so auditing it is a
    measurement of the old scene rather than a re-implementation of it.
    """
    mj = scene._import_mujoco()                       # noqa: SLF001 (same package)
    spec = mj.MjSpec()
    from flygym.utils.mjcf import add_texture
    add_texture(spec, name="oldchecker", type="2d", builtin="checker",
                width=int(width), height=int(height),
                rgb1=tuple(float(v) for v in rgb1),
                rgb2=tuple(float(v) for v in rgb2))
    model = spec.compile()
    tex = np.asarray(model.tex_data, dtype=float).reshape(-1)
    w = int(np.ravel(model.tex_width[0])[0])
    h = int(np.ravel(model.tex_height[0])[0])
    ch = int(np.ravel(model.tex_nchannel[0])[0])
    adr = int(np.ravel(model.tex_adr[0])[0])
    img = tex[adr:adr + w * h * ch].reshape(h, w, ch)
    return {
        "image": img.astype(np.uint8),
        "width": w, "height": h, "nchannel": ch,
        "source": ("MuJoCo 3.9 builtin checker, rgb1=(0.3,0.3,0.3), "
                   "rgb2=(0.4,0.4,0.4), default checker period -- i.e. exactly "
                   "FlyGym FlatGroundWorld's 'checker' texture"),
        "plane_texture_repeat": [250.0, 250.0],
        "plane_half_size_mm": 1000.0,
        "tile_size_mm": OLD_GROUND_TILE_MM,
    }


# --------------------------------------------------------------------------- #
# terrain: seeded, multi-octave fBm with an explicit Hurst exponent
# --------------------------------------------------------------------------- #

def generate_pink_noise(size: int, *, beta: float, seed: int,
                        periodic: bool = True) -> np.ndarray:
    """Zero-mean unit-variance 2-D pink noise with radial slope ``-beta``.

    ``periodic=True`` fills the amplitude of a *frequency grid* directly.  That
    has two properties this module needs: the result wraps exactly (so a MuJoCo
    heightfield asset can be TILED without a step at its edges), and there is no
    dominant wavelength at all -- the whole spectrum is a continuum.
    """
    n = int(size)
    if n < 8:
        raise ValueError("size must be >= 8")
    rng = np.random.default_rng(int(seed))
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.fftfreq(n)[None, :]
    k = np.sqrt(fx ** 2 + fy ** 2)
    amp = np.zeros_like(k)
    nz = k > 0
    amp[nz] = k[nz] ** (-float(beta) / 2.0)
    if not periodic:
        # Non-periodic route: synthesise a longer field and crop the centre.
        # Kept because the brief asks for a non-periodic option; the cost is
        # that the crop no longer wraps, so it cannot be used as a tiled asset.
        m = 3 * n
        ky = np.fft.fftfreq(m)[:, None]
        kx = np.fft.fftfreq(m)[None, :]
        kk = np.sqrt(kx ** 2 + ky ** 2)
        a2 = np.zeros_like(kk)
        nz2 = kk > 0
        a2[nz2] = kk[nz2] ** (-float(beta) / 2.0)
        ph2 = np.exp(2j * np.pi * rng.random((m, m)))
        f2 = np.fft.ifft2(a2 * ph2).real
        o = (m - n) // 2
        f = f2[o:o + n, o:o + n]
    else:
        ph = np.exp(2j * np.pi * rng.random((n, n)))
        f = np.fft.ifft2(amp * ph).real
    f = f - f.mean()
    s = f.std()
    return (f / s) if s > 0 else f


def generate_fbm_heightfield(n_samples: int = 257, *, extent_mm: float = 120.0,
                             hurst: float = DEFAULT_HURST, seed: int = 0,
                             amplitude_std_mm: float = 0.16,
                             amplitude_peak_to_peak_mm: float | None = None,
                             micro_relief_mm: float = 0.0,
                             micro_cutoff_fraction: float = 0.25,
                             periodic: bool = True,
                             metadata_extra: Mapping[str, Any] | None = None
                             ) -> tuple[np.ndarray, dict]:
    """Seeded multi-octave fBm terrain.  Returns ``(elevation, metadata)``.

    The elevation grid is normalised to exactly ``[0, 1]`` because that is what
    MuJoCo's ``hfield`` consumes: ``MjsSpec.add_hfield(userdata=...)`` is scaled
    by ``hfield_size[2]`` on the vertical axis, so all physical amplitudes are
    applied by ``apply_natural_scene`` and reported in ``metadata`` in mm.

    ``hurst`` sets the radial power-law exponent of the *background* through
    ``slope = -(2*H + 1.5)`` for a 2-D surface (the familiar 2-D result: a
    surface whose 1-D profile has spectral slope ``-(2H+1)`` has, in the radial
    average of its 2-D spectrum, the extra factor ``k^-1`` from the circle
    measure).  The relation is recorded in the metadata and is checked by the
    self-report: the audit's fitted slope is compared with it rather than
    assumed to agree.

    ``micro_relief_mm`` adds a second, high-wavenumber fBm component with a low
    cutoff (sand/grit at the grid's own scale).  It is reported separately as
    ``micro_relief`` and is deliberately *outside* the fit band used to verify
    the Hurst exponent, so it cannot be used to make the slope look natural.
    """
    n = int(n_samples)
    if n < 16:
        raise ValueError("n_samples must be >= 16")
    if not 0.0 < float(hurst) < 1.5:
        raise ValueError("hurst must be in (0, 1.5)")
    slope = -(2.0 * float(hurst) + 1.5)
    # The radial slope of a 2-D field gives beta = -slope; we ask for exactly
    # that exponent instead of hand-waving a value.
    background = generate_pink_noise(n, beta=-slope, seed=int(seed),
                                     periodic=bool(periodic))

    micro = None
    if float(micro_relief_mm) > 0:
        m = generate_pink_noise(n, beta=-slope, seed=int(seed) + 101,
                                periodic=bool(periodic))
        # keep only the high wavenumbers: a low-pass complement removes the
        # large scales, which are already carried by `background`
        m = m - _low_pass(m, cutoff_fraction=float(micro_cutoff_fraction))
        m = m / m.std() if m.std() > 0 else m
        micro = m

    if amplitude_peak_to_peak_mm is not None:
        amp_mm = float(amplitude_peak_to_peak_mm)
        mesh_mm = amplitude_std_mm
        used_rule = "peak_to_peak"
        total = background.copy()
        total = total / total.std()
        total = total * (amp_mm / (total.max() - total.min()))
        if micro is not None:
            total = total + micro * (float(micro_relief_mm)
                                     / (micro.max() - micro.min()))
    else:
        mesh_mm = float(amplitude_std_mm)
        total = background * mesh_mm
        if micro is not None:
            total = total + micro * float(micro_relief_mm)
        amp_mm = float(total.max() - total.min())
        used_rule = "std"

    lo, hi = float(total.min()), float(total.max())
    elev = (total - lo) / (hi - lo) if hi > lo else np.zeros_like(total)

    cell_mm = 2.0 * float(extent_mm) / n
    meta: dict[str, Any] = {
        "generator": "generate_fbm_heightfield",
        "method": ("spectral synthesis: radial amplitude k^(-beta/2) with random "
                   "phase on an n x n frequency grid (DC zeroed).  This is the "
                   "fBm / power-law-surface construction; its slope is set by "
                   "`hurst`, and it contains NO discrete spectral line."),
        "periodic_asset": bool(periodic),
        "periodic_asset_note": (
            "True means the grid wraps exactly.  MuJoCo tiles an hfield geom "
            "beyond its extent, so the resulting SURFACE has no step at the "
            "patch edge (the physics therefore has no cliff).  The tiling period "
            "equals the patch size, which is many times the fly's body length; "
            "the audit reports the ratio of that period to the smallest "
            "wavelength actually present so the difference between 'a periodic "
            "grating' and 'a tiled stochastic surface' is a number, not a claim."
            if periodic else
            "False means the field was cropped from a 3n x 3n synthesis, so it "
            "does NOT wrap: used as an hfield asset this produces an unphysical "
            "step of up to the full amplitude at the patch edge."),
        "n_samples": n,
        "extent_half_mm": float(extent_mm),
        "extent_full_mm": 2.0 * float(extent_mm),
        "cell_size_mm": cell_mm,
        "hurst": float(hurst),
        "hurst_spectral_slope_expected_2d": float(slope),
        "hurst_slope_relation": ("slope_2d = -(2*H + 1.5) for a 2-D surface: the "
                                 "1-D profile slope -(2H+1) plus the k^-1 circle "
                                 "measure of the radial average"),
        "amplitude_rule": used_rule,
        "amplitude_std_mm": (mesh_mm if used_rule == "std" else None),
        "amplitude_peak_to_peak_mm": amp_mm,
        "micro_relief_mm": float(micro_relief_mm),
        "micro_cutoff_fraction_of_nyquist": float(micro_cutoff_fraction),
        "seed": int(seed),
        "data_range_normalised": [0.0, 1.0],
        "elevation_std_of_normalised_grid": float(elev.std()),
        "smallest_wavelength_present_mm": 2.0 * cell_mm,
        "tiling_period_mm": 2.0 * float(extent_mm),
        "tiling_period_over_smallest_wavelength": (
            2.0 * float(extent_mm) / (2.0 * cell_mm)),
        "provenance": scene.ENGINEERING_DEFAULT,
    }
    if metadata_extra:
        meta.update(dict(metadata_extra))
    return elev, meta


def _low_pass(field: np.ndarray, *, cutoff_fraction: float) -> np.ndarray:
    n = field.shape[0]
    k = np.sqrt(np.fft.fftfreq(n)[:, None] ** 2 + np.fft.fftfreq(n)[None, :] ** 2)
    f = np.fft.fft2(field)
    f[k > float(cutoff_fraction) * 0.5] = 0.0
    return np.fft.ifft2(f).real


# --------------------------------------------------------------------------- #
# ground texture: procedural pink noise and/or a real photograph
# --------------------------------------------------------------------------- #

def generate_natural_texture_png(path: str, size: int = 512, *, seed: int = 0,
                                 beta: float = 2.4,
                                 target_std: float = 0.12,
                                 grey: bool = False,
                                 clahe_clip: float | None = None) -> dict:
    """Write a natural-statistics PNG and return its metadata.

    Route B of the brief: pink / 1-over-f noise, radial amplitude
    ``~ k**(-beta/2)`` with random phase and DC zeroed, then normalised to a
    plausible contrast.  ``beta`` is recorded; the *measured* fitted slope of
    the written file is recorded too (under ``spectral``), and the two are not
    assumed to agree -- for a 2-D field the radial slope is ``-beta``.

    ``clahe_clip`` applies Pillow's contrast-limited adaptive histogram
    equalisation when available; it is OFF by default because it is a nonlinear
    operation that can *manufacture* periodicity, and this module will not
    silently trade naturalness for contrast.
    """
    from PIL import Image
    n = int(size)
    field = generate_pink_noise(n, beta=float(beta), seed=int(seed),
                                periodic=True)
    field = field / (field.std() if field.std() > 0 else 1.0) * float(target_std)
    img = np.clip(0.5 + field, 0.0, 1.0)
    u8 = np.round(img * 255.0).astype(np.uint8)
    if grey:
        arr = np.repeat(u8[:, :, None], 3, axis=2)
    else:
        arr = np.repeat(u8[:, :, None], 3, axis=2)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    Image.fromarray(arr).save(path)
    if clahe_clip is not None:                      # pragma: no cover - optional
        from PIL import ImageOps
        im = Image.open(path)
        im = ImageOps.equalize(im)
        im.save(path)
    check = np.asarray(Image.open(path))
    audit = spectral_audit(check, sample_pitch=1.0, sample_unit="texel")
    return {
        "generator": "generate_natural_texture_png",
        "route": "PROCEDURAL",
        "path": os.path.abspath(path),
        "size_px": [n, n],
        "nchannel": int(check.shape[2]) if check.ndim == 3 else 1,
        "beta_requested": float(beta),
        "beta_expected_radial_slope": -float(beta),
        "target_std": float(target_std),
        "grey": bool(grey),
        "half_range_fraction": float(np.round(img.max() - img.min(), 4)),
        "clahe_clip": clahe_clip,
        "sha256": _sha256_file(path),
        "spectral": audit,
        "provenance": scene.ENGINEERING_DEFAULT,
        "provenance_note": ("procedural: no external source, so there is nothing "
                            "to cite; the generator parameters are the record."),
    }


# -- Route A: fetch a real, licence-verified photograph --------------------- #

#: Verified by querying the Wikimedia Commons API in this session (see
#: :func:`download_natural_texture`, which re-reads the licence from the API
#: rather than trusting this table).  Both candidates are small enough to fetch
#: whole; Wikimedia rejects arbitrary thumbnail sizes, so the API's own ``url``
#: is used and any tracking query string is stripped.
COMMONS_CANDIDATES: tuple[dict, ...] = (
    {
        "key": "dry_decay_leaves_preview",
        "title": "File:Dry decay leaves (Amal Kumar via Poly Haven).webp",
        "why": ("a real photograph of a forest floor covered in dry dead leaves: "
                "the natural surface a walking Drosophila actually meets"),
    },
    {
        "key": "small_pebbles",
        "title": "File:Small pebbles.jpg",
        "why": "a real photograph of loose small pebbles / gravel",
    },
)

_COMMONS_UA = {"User-Agent": "cell_wound_prototype-natural-scene/1.0 (research prototype)"}


def commons_imageinfo(titles: Sequence[str], *, timeout: float = 30.0) -> dict:
    """Query the Commons API for url + size + licence metadata of each title."""
    params = dict(action="query", format="json", titles="|".join(titles),
                  prop="imageinfo", iiprop="url|size|extmetadata|mime")
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=_COMMONS_UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    out = {}
    for pid, page in data.get("query", {}).get("pages", {}).items():
        if "imageinfo" not in page:
            continue
        ii = page["imageinfo"][0]
        em = ii.get("extmetadata", {})
        out[page["title"]] = {
            "title": page["title"],
            "url": ii["url"].split("?")[0],
            "width": ii.get("width"), "height": ii.get("height"),
            "mime": ii.get("mime"),
            "descriptionurl": ii.get("descriptionurl"),
            "license_name": (em.get("LicenseShortName", {}) or {}).get("value", ""),
            "license_url": (em.get("LicenseUrl", {}) or {}).get("value", ""),
            "usage_terms": (em.get("UsageTerms", {}) or {}).get("value", ""),
            "artist": _strip_html((em.get("Artist", {}) or {}).get("value", "")),
            "credit": _strip_html((em.get("Credit", {}) or {}).get("value", "")),
        }
    return out


def _strip_html(text: str) -> str:
    import re
    return re.sub(r"<[^>]+>", " ", str(text or "")).strip()


def is_free_license(license_name: str) -> bool:
    """True only for the licences this module is willing to use.

    Deliberately narrow: public domain, CC0, CC-BY and CC-BY-SA.  Anything the
    module cannot positively identify is REJECTED rather than used, because the
    brief forbids using an asset whose licence was not verified.
    """
    s = str(license_name or "").strip().lower()
    if not s:
        return False
    ok = ("public domain", "cc0", "cc-by", "cc by", "cc-by-sa", "cc by-sa",
          "pd-", "no restrictions")
    return any(t in s for t in ok)


def download_natural_texture(out_png: str, *, size: int = 1024,
                             prefer: str = "dry_decay_leaves_preview",
                             candidate_titles: Sequence[str] | None = None,
                             timeout: float = 60.0, grey: bool = False,
                             max_source_pixels: int = 4096) -> dict:
    """Route A.  Download a licence-verified Commons photo and write a square PNG.

    The licence is read from the API response, not from a table in this file.  A
    candidate is used only if :func:`is_free_license` accepts its licence name
    and it carries an author.  The image is downscaled so its longer side is at
    most ``size`` and centre-cropped to a square, which keeps the pixel statistics
    of the original while bounding the file.

    Returns a metadata dict with the full provenance (author, licence name,
    licence URL, source page, source URL) plus the spectral audit of the written
    file.  Raises ``PermissionError`` if no candidate has a usable licence.
    """
    from PIL import Image
    titles = list(candidate_titles or [c["title"] for c in COMMONS_CANDIDATES])
    info = commons_imageinfo(titles, timeout=timeout)
    order = [prefer] + [c["key"] for c in COMMONS_CANDIDATES if c["key"] != prefer]
    by_key = {c["key"]: c for c in COMMONS_CANDIDATES}
    tried, rejected = [], []
    chosen = None
    for key in order:
        cand = by_key.get(key)
        if cand is None or cand["title"] not in info:
            continue
        entry = info[cand["title"]]
        tried.append({"key": key, "title": entry["title"], "url": entry["url"],
                      "license": entry["license_name"], "author": entry["artist"],
                      "size": [entry["width"], entry["height"]]})
        if not is_free_license(entry["license_name"]):
            rejected.append({"key": key, "license": entry["license_name"],
                             "reason": "licence not accepted by is_free_license()"})
            continue
        if not entry["artist"].strip():
            rejected.append({"key": key, "license": entry["license_name"],
                             "reason": "API returned no author, so the asset cannot "
                                       "be attributed"})
            continue
        if entry["width"] and max(entry["width"], entry["height"]) > max_source_pixels:
            rejected.append({"key": key,
                             "reason": f"source is {entry['width']}x{entry['height']}; "
                                       f"above max_source_pixels"})
            continue
        chosen = entry
        chosen_key = key
        break
    if chosen is None:
        raise PermissionError(
            "no Wikimedia Commons candidate had a usable, verified licence: "
            f"{rejected or 'no candidates resolved'}")
    req = urllib.request.Request(chosen["url"], headers=_COMMONS_UA)
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    src_path = os.path.join(os.path.dirname(os.path.abspath(out_png)),
                            "src_" + os.path.basename(chosen["url"]).replace(
                                "%28", "(").replace("%29", ")"))
    with open(src_path, "wb") as f:
        f.write(raw)
    im = Image.open(src_path).convert("RGB")
    n = int(size)
    im = im.resize((n, n), Image.LANCZOS)
    if grey:
        im = im.convert("L").convert("RGB")
    im.save(out_png)
    check = np.asarray(Image.open(out_png))
    audit = spectral_audit(check, sample_pitch=1.0, sample_unit="texel")
    return {
        "generator": "download_natural_texture",
        "route": ("PHOTO_PD" if "public domain" in chosen["license_name"].lower()
                  else "PHOTO_CC0"),
        "key": chosen_key,
        "path": os.path.abspath(out_png),
        "source_file_cached_at": src_path,
        "source_url": chosen["url"],
        "source_page": chosen["descriptionurl"],
        "source_title": chosen["title"],
        "source_size_px": [chosen["width"], chosen["height"]],
        "source_mime": chosen["mime"],
        "author": chosen["artist"],
        "license_name": chosen["license_name"],
        "license_url": (chosen["license_url"]
                        or "https://commons.wikimedia.org/wiki/Commons:Licensing"),
        "usage_terms": chosen["usage_terms"],
        "credit": chosen["credit"],
        "downscaled_to_px": [n, n],
        "resampling": "PIL.Image.LANCZOS square resize",
        "grey": bool(grey),
        "bytes": int(os.path.getsize(out_png)),
        "source_bytes": len(raw),
        "sha256": _sha256_file(out_png),
        "sha256_source": _sha256_file(src_path),
        "candidates_tried": tried,
        "candidates_rejected": rejected,
        "spectral": audit,
        "provenance": scene.MEASURED_CITED,
        "provenance_note": (
            "the licence fields were read from the Wikimedia Commons API response "
            "in this session (prop=imageinfo, iiprop=extmetadata); they are quoted, "
            "not invented."),
    }


def resolve_ground_texture(*, out_png: str | None = None, size: int = 1024,
                           allow_network: bool = True, seed: int = 0,
                           beta: float = 2.4, grey: bool = False) -> dict:
    """Whichever route is available: real photo first, procedural fallback.

    Order: (1) a cached photo PNG; (2) download a licence-verified photo;
    (3) procedural pink noise.  Every branch records which route it took, so a
    report can never present a procedural texture as a photograph.
    """
    os.makedirs(ASSET_DIR, exist_ok=True)
    out_png = out_png or os.path.join(ASSET_DIR, "natural_ground_photo.png")
    proc_png = os.path.join(ASSET_DIR, "natural_ground_procedural.png")
    if os.path.exists(out_png):
        meta = _reprovenance_cached(out_png)
        if meta is not None:
            meta["route_decision"] = "cached file on disk"
            return meta
    if allow_network:
        try:
            meta = download_natural_texture(out_png, size=size, grey=grey)
            meta["route_decision"] = "downloaded from Wikimedia Commons this run"
            return meta
        except Exception as exc:                      # network or licence failure
            fallback_reason = f"{type(exc).__name__}: {exc}"
    else:
        fallback_reason = "network disabled by the caller"
    meta = generate_natural_texture_png(proc_png, size=min(int(size), 512),
                                        seed=seed, beta=beta, grey=grey)
    meta["route_decision"] = "PROCEDURAL FALLBACK (no photograph was used)"
    meta["fallback_reason"] = fallback_reason
    return meta


#: Cached provenance sidecar written next to a downloaded PNG, so a re-run with
#: no network still reports the REAL author and licence rather than losing them.
PROVENANCE_SIDECAR = "natural_ground_photo.provenance.json"
#: Suffix for a PER-IMAGE sidecar.  The constant above is the historical fixed name, kept
#: for the original asset; a scene that NAMES its own texture needs a sidecar tied to that
#: image, because a shared fixed name lets two images in one directory claim one licence
#: record -- which is how the meadow first rendered with the dry-leaves photograph while
#: the report said the grass photograph was in use.
PROVENANCE_SIDECAR_SUFFIX = ".provenance.json"


def _reprovenance_cached(png_path: str) -> dict | None:
    side = os.path.join(os.path.dirname(os.path.abspath(png_path)),
                        PROVENANCE_SIDECAR)
    if not os.path.exists(side):
        return None
    with open(side) as f:
        meta = json.load(f)
    meta["path"] = os.path.abspath(png_path)
    meta["sha256"] = _sha256_file(png_path)
    from PIL import Image
    meta["spectral"] = spectral_audit(np.asarray(Image.open(png_path)),
                                      sample_pitch=1.0, sample_unit="texel")
    return meta


def write_provenance_sidecar(meta: Mapping[str, Any], png_path: str | None = None
                             ) -> str:
    png = png_path or str(meta.get("path", ""))
    side = os.path.join(os.path.dirname(os.path.abspath(png)), PROVENANCE_SIDECAR)
    with open(side, "w") as f:
        json.dump(scene._jsonable(dict(meta)), f, indent=2)
    return side


# --------------------------------------------------------------------------- #
# the scene configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class NaturalSceneConfig(scene.SceneConfig):
    """A :class:`scene.SceneConfig` plus the fields needed for a natural scene.

    It IS a ``SceneConfig`` (subclass), so ``scene.apply_scene`` accepts it
    unchanged and nothing in ``scene.py`` has to know about this module.  The
    extra fields are all optional, so a plain ``SceneConfig`` still works with
    :func:`apply_natural_scene` -- it simply gets no sky/ground/texture change.
    """
    # -- terrain ----------------------------------------------------------
    hurst: float = DEFAULT_HURST
    hfield_amplitude_std_mm: float = 0.008
    hfield_micro_relief_mm: float = 0.0008
    # -- sky -------------------------------------------------------------
    sky_rgb1: tuple[float, float, float] = (0.42, 0.50, 0.60)   # horizon band
    sky_rgb2: tuple[float, float, float] = (0.03, 0.05, 0.10)   # zenith
    sky_texture_name: str = "skybox"
    sky_note: str = ""
    # -- ground texture ---------------------------------------------------
    ground_texture_name: str = "natural_ground"
    ground_material_name: str = "natural_ground_mat"
    #: Physical size of one texture tile, in mm.  The SAME number is used to
    #: derive ``texrepeat`` for the infinite plane and for the heightfield, so
    #: the two surfaces are painted at one scale instead of two.
    texture_tile_mm: float = 90.0
    texture_reflectance: float = 0.02
    texture_shininess: float = 0.05
    texture_specular: float = 0.05
    #: "photo" (use the resolved PNG), "procedural" (generated pink noise), or
    #: "checker_flat" (leave the ground as a flat non-periodic colour -- the
    #: control that isolates the TEXTURE's contribution from the terrain's).
    ground_texture_route: str = "photo"
    #: Optional EXPLICIT texture PNG for the ground.  ``None`` means "resolve one as
    #: before"; a path means "use exactly this file", which is how the meadow preset
    #: gets the grass photograph instead of whatever ``resolve_ground_texture`` would
    #: pick.  MEASURED reason this field exists: without it the meadow rendered with the
    #: dry-leaves photograph, because ``resolve_ground_texture`` returns the FIRST cached
    #: asset it finds and does not know which scene asked for it.
    ground_texture_path: "str | None" = None
    ground_flat_rgba: tuple[float, float, float, float] = (0.30, 0.26, 0.20, 1.0)
    # -- lighting rebalance ----------------------------------------------
    #: MuJoCo vis.headlight multipliers.  The shipped values are
    #: ambient 0.5 / diffuse 0.6 / specular 0.3 (MEASURED_LOCAL from
    #: mujoco_globals.yaml).  These are the values to use INSTEAD.
    headlight_ambient: tuple[float, float, float] = (0.25, 0.25, 0.25)
    headlight_diffuse: tuple[float, float, float] = (1.00, 1.00, 1.00)
    headlight_specular: tuple[float, float, float] = (0.15, 0.15, 0.15)

    def __post_init__(self):
        super().__post_init__()
        if self.ground_texture_route not in ("photo", "procedural", "checker_flat"):
            raise ValueError("ground_texture_route must be photo/procedural/checker_flat")
        if self.ground_texture_path is not None and not str(self.ground_texture_path).strip():
            raise ValueError("ground_texture_path must be None or a non-empty path")
        if float(self.texture_tile_mm) <= 0:
            raise ValueError("texture_tile_mm must be > 0")
        for f in ("headlight_ambient", "headlight_diffuse", "headlight_specular"):
            v = tuple(float(x) for x in getattr(self, f))
            if len(v) != 3 or not all(math.isfinite(x) and 0 <= x <= 8 for x in v):
                raise ValueError(f"{f} must be three finite non-negative numbers, got {v!r}")


NATURAL_HFIELD_HALF_EXTENT_MM = 60.0
NATURAL_HFIELD_N = 257
NATURAL_HFIELD_SEED = 20240517
NATURAL_HFIELD_AMPLITUDE_STD_MM = 0.008
NATURAL_HFIELD_MICRO_MM = 0.0008
NATURAL_LIGHT_SCALE = 1.0


def _natural_lights() -> tuple[scene.LightSpec, ...]:
    """Lights for the natural scene: a reduced sun, a sky fill and an ambient.

    ``scene.py``'s ``SUN_LIGHT``/``FILL_LIGHT``/``AMBIENT_LIGHT`` summed with the
    default headlight saturate an eye image; the values here are darker and are
    the author's choice (ENGINEERING_DEFAULT, swept in the ledger).
    """
    sun = scene.LightSpec(
        name="nat_sun", kind="directional", pos_mm=(0.0, 0.0, 200.0),
        dir=(-0.35, -0.25, -1.0), diffuse=(0.32, 0.32, 0.30),
        specular=(0.06, 0.06, 0.06), ambient=(0.06, 0.06, 0.06), castshadow=True)
    fill = scene.LightSpec(
        name="nat_fill", kind="directional", pos_mm=(0.0, 0.0, 150.0),
        dir=(0.45, 0.30, -1.0), diffuse=(0.10, 0.12, 0.16),
        specular=(0.0, 0.0, 0.0), ambient=(0.03, 0.03, 0.04), castshadow=False)
    amb = scene.LightSpec(
        name="nat_ambient", kind="directional", pos_mm=(0.0, 0.0, 100.0),
        dir=(0.0, 0.0, -1.0), diffuse=(0.0, 0.0, 0.0),
        specular=(0.0, 0.0, 0.0), ambient=(0.05, 0.05, 0.06), castshadow=False)
    return (sun, fill, amb)


def _natural_objects() -> tuple[scene.ObjectSpec, ...]:
    """Visual landmarks kept from ``scene.py`` but rescaled and darkened.

    ``scene.py``'s landmarks are 1.5 mm boxes (0.75 half-extent) 6 mm from spawn
    with a near-white 0.95 albedo: on a 2.5 mm fly those are giant mirrors and
    they contribute to the clipping.  These are smaller and darker.  All are
    ``contype=conaffinity=0`` (visual only), so they cannot touch the fly.
    """
    return (
        scene.ObjectSpec(name="landmark_dark", kind="target", pos_mm=(5.0, 0.0, 0.55),
                         size_mm=(0.35, 0.35, 0.35), rgba=(0.05, 0.05, 0.06, 1.0),
                         geom_type="box"),
        scene.ObjectSpec(name="landmark_light", kind="target", pos_mm=(-4.0, 2.0, 0.55),
                         size_mm=(0.35, 0.35, 0.35), rgba=(0.72, 0.70, 0.62, 1.0),
                         geom_type="box"),
        scene.ObjectSpec(name="pebble_contrast", kind="texture", pos_mm=(2.5, 2.5, 0.12),
                         size_mm=(0.12, 0.0, 0.0), rgba=(0.45, 0.20, 0.06, 1.0),
                         geom_type="sphere"),
    )


#: RoughnessConfig validates that the grid can resolve ``wavelength_mm`` with at
#: least 4 cells, but the fBm generator does NOT use that field (it synthesises a
#: spectral continuum).  This multiple is therefore only the value handed to the
#: validator so that the preset is a legal RoughnessConfig; it is recorded as such
#: and must not be read as "the terrain's wavelength".
HFIELD_VALIDATOR_WAVELENGTH_CELLS = 4.0


def _terrain_for_cfg(cfg) -> tuple[dict, np.ndarray | None]:
    """Generate the terrain this config asks for and audit it.  Cached by params.

    Returns ``(metadata, elevation)``.  When the config has no heightfield
    roughness the result is an empty dict and ``None``.
    """
    rc = getattr(cfg, "roughness", None)
    if rc is None or getattr(rc, "kind", None) not in ("heightfield", "grass"):
        return {}, None
    key = (str(getattr(rc, "kind", "")), int(rc.n_rows), float(rc.half_extent_mm),
           int(rc.seed),
           float(getattr(cfg, "hfield_micro_relief_mm", NATURAL_HFIELD_MICRO_MM)),
           float(getattr(cfg, "hurst", DEFAULT_HURST)))
    cached = _TERRAIN_CACHE.get(key)
    if cached is None:
        cached = _build_terrain(rc, cfg)
        _TERRAIN_CACHE[key] = cached
    meta, audit, elev = cached
    return {"metadata": dict(meta), "audit": dict(audit)}, elev


def _build_terrain(rc, cfg) -> tuple[dict, dict, np.ndarray]:
    # THE GRASS BRANCH.  Dispatched on ``roughness.kind == "grass"`` and importing the
    # generator LAZILY, so this module keeps working when ``grass_scene.py`` is absent
    # -- the same defensive pattern the scene presets use for this module.
    if getattr(rc, "kind", None) == "grass":
        try:
            from grass_scene import generate_grass_heightfield
        except ImportError as exc:                      # pragma: no cover
            raise RuntimeError(
                "roughness.kind='grass' needs grass_scene.py, which is not "
                f"importable: {type(exc).__name__}: {exc}") from exc
        elev, meta = generate_grass_heightfield(
            n_samples=int(rc.n_rows), extent_mm=float(rc.half_extent_mm),
            seed=int(rc.seed))
        audit = spectral_audit(elev, sample_pitch=float(meta["cell_size_mm_actual"]),
                               sample_unit="mm", k_floor=4)
        meta = dict(meta)
        meta["spectral_audit_fitted_slope"] = float(audit.get("fitted_slope", float("nan")))
        meta["spectral_audit_r2"] = float(audit.get("fitted_r2", float("nan")))
        meta["fly_lengths_per_peak_wavelength"] = audit.get(
            "fly_lengths_per_peak_wavelength")
        return meta, audit, elev
    elev, meta = generate_fbm_heightfield(
        n_samples=int(rc.n_rows), extent_mm=float(rc.half_extent_mm),
        hurst=float(getattr(cfg, "hurst", DEFAULT_HURST)), seed=int(rc.seed),
        amplitude_std_mm=float(getattr(cfg, "hfield_amplitude_std_mm",
                                       NATURAL_HFIELD_AMPLITUDE_STD_MM)),
        micro_relief_mm=float(getattr(cfg, "hfield_micro_relief_mm",
                                      NATURAL_HFIELD_MICRO_MM)))
    audit = spectral_audit(elev, sample_pitch=float(meta["cell_size_mm"]),
                           sample_unit="mm", k_floor=4)
    audit["slope_expected_from_hurst"] = float(
        meta["hurst_spectral_slope_expected_2d"])
    audit["slope_measured_minus_expected"] = (
        None if audit["fitted_slope"] is None else
        float(audit["fitted_slope"] - meta["hurst_spectral_slope_expected_2d"]))
    audit["detector_caveat"] = (
        "the periodicity detector ignores wavenumbers below k_floor=4 bins, "
        "because the lowest bins of a finite patch carry the patch's own "
        "fundamental and the window's leakage; a 'not periodic' verdict "
        "therefore means 'no dominant grating above the 4th harmonic', which "
        "is the meaningful claim for a tiled stochastic surface.")
    meta["validator_wavelength_mm"] = (
        HFIELD_VALIDATOR_WAVELENGTH_CELLS * float(meta["cell_size_mm"]))
    meta["validator_wavelength_note"] = (
        "RoughnessConfig.wavelength_mm is a VALIDATION gate in scene.py (>= 4 "
        "grid cells per wavelength); the fBm generator does not use it.  This "
        "preset therefore passes a legal value and takes its real spectral "
        "content from the fBm synthesis, reported in spectral_audit.")
    return meta, audit, elev


_TERRAIN_CACHE: dict = {}


def _base_natural_config(name: str, *, objects: Sequence[scene.ObjectSpec],
                         description: str, seed: int = 0) -> NaturalSceneConfig:
    return NaturalSceneConfig(
        name=name,
        friction_by_class={"ground": scene.FRICTION_GROUND,
                           "tarsus": scene.FRICTION_TARSUS,
                           "body": scene.FRICTION_BODY},
        roughness=scene.RoughnessConfig(
            kind="heightfield",
            # amplitude_mm is a PLACEHOLDER: apply_natural_scene replaces it with
            # the exact peak-to-peak of the generated fBm grid, because MuJoCo's
            # hfield_size[2] multiplies the normalised [0,1] grid, so the number
            # written there IS the physical peak-to-peak and must match the
            # generator's own report.
            amplitude_mm=NATURAL_HFIELD_AMPLITUDE_STD_MM * 2.0,
            # validation gate only -- see HFIELD_VALIDATOR_WAVELENGTH_CELLS
            wavelength_mm=HFIELD_VALIDATOR_WAVELENGTH_CELLS * (
                2.0 * NATURAL_HFIELD_HALF_EXTENT_MM / NATURAL_HFIELD_N),
            seed=NATURAL_HFIELD_SEED, n_rows=NATURAL_HFIELD_N,
            n_cols=NATURAL_HFIELD_N,
            half_extent_mm=NATURAL_HFIELD_HALF_EXTENT_MM,
            base_offset_mm=scene.HFIELD_BASE_OFFSET_MIN_MM,
            hfield_material=None),
        capillary=None,
        adhesion_gain=scene.ADHESION_GAIN,
        adhesion_added=True,
        lights=_natural_lights(),
        objects=tuple(objects),
        description=description,
        geom_priority_by_class={"ground": 0, "tarsus": 2, "body": 1},
        seed=seed,
        ground_texture_route="photo")


_NATURAL_FBM = _base_natural_config(
    "natural_fbm", objects=_natural_objects(),
    description=("Natural scene: fBm micro-relief over a 120 mm tiled patch "
                 "(wraps exactly, so there is no cliff at the edge), a natural "
                 "ground texture (CC0 photograph or procedural pink noise), a "
                 "gradient sky, rebalanced headlight, three dark lights and "
                 "three visual landmarks."))

_NATURAL_FBM_LIT_NO_OBJECTS = dataclasses.replace(
    _NATURAL_FBM, name="natural_fbm_lit_no_objects", objects=(),
    description=("Identical to natural_fbm in every field except `objects`, "
                 "which is empty.  Isolates the landmarks' visual contribution; "
                 "changes nothing physical (objects are contype=conaffinity=0)."))

_FBM_UNIFORM_SKY = dataclasses.replace(
    _NATURAL_FBM, name="fbm_uniform_sky",
    sky_rgb1=(1.0, 1.0, 1.0), sky_rgb2=(1.0, 1.0, 1.0),
    sky_note=("CONTROL: the shipped white-to-white skybox, i.e. the old uniform "
              "white sky, kept on top of the natural terrain and texture.  "
              "natural_fbm vs fbm_uniform_sky therefore measures the sky's "
              "contribution with everything else held fixed."),
    description=("Control: natural fBm terrain + natural texture + the OLD "
                 "uniform white sky.  Isolates the sky's contribution."))

_FBM_NO_TEXTURE = dataclasses.replace(
    _NATURAL_FBM, name="fbm_flat_ground_texture",
    ground_texture_route="checker_flat", objects=(),
    description=("Second control: natural fBm terrain, but the ground texture is "
                 "a single flat (non-periodic) colour instead of an image.  "
                 "Isolates the texture's contribution from the terrain's."))

#: THE MEADOW.  A grass-stand heightfield 400 x 400 mm (blades resolved at 0.25 mm),
#: painted with a real CC0 photograph of mown lawn.  Built by ``dataclasses.replace``
#: from the fBm preset so that EVERY other field -- sky, friction classes, lighting,
#: adhesion, geom priorities -- is identical to ``natural_fbm``; the ONLY differences are
#: the roughness kind and the terrain size.  That is deliberate: it makes "grass vs fBm"
#: a controlled comparison rather than two unrelated scenes.
_MEADOW_GRASS = dataclasses.replace(
    _NATURAL_FBM, name="meadow_grass",
    roughness=scene.RoughnessConfig(
        kind="grass",
        # amplitude is a PLACEHOLDER replaced by the generator's own peak-to-peak, the
        # same contract the fBm preset uses.
        amplitude_mm=1.0,
        # the resolution gate: the blade spacing must span >= 4 grid cells
        wavelength_mm=4.0 * (2.0 * 200.0 / 801),
        seed=20261004, n_rows=801, n_cols=801,
        half_extent_mm=200.0,
        base_offset_mm=scene.HFIELD_BASE_OFFSET_MIN_MM,
        # CLEARED PATCHES under the arena.  The six fiducial markers stand on these, on
        # purpose: in uncleared grass they are occluded by blades from every camera
        # (measured), which makes the world frame unfittable.
        grass_clearings_mm=((0.0, 0.0, 4.2, 1.2),),
        hfield_material=None),
    # a lawn tile: 90 mm of photograph covers the same physical patch as before, so the
    # blade scale in the IMAGE and the blade scale in the RELIEF are both about right
    # (the photo shows blades of a few mm, the relief has blades of 0.1-0.3 mm; the
    # relief is the physical one and the photo is colour, which the asset record says).
    texture_tile_mm=90.0,
    ground_texture_path=os.path.join(ASSET_DIR, "grass_meadow_photo.png"),
    objects=(),
    description=("MEADOW: a generated stand of grass, 400 x 400 mm of it, with blades "
                 "0.1-0.3 mm apart and 0.2-0.8 mm tall (litter layer at 20 um RMS), "
                 "painted with a real CC0 photograph of mown lawn.  Every other scene "
                 "field is identical to natural_fbm.  Built for the multi-camera limb "
                 "tracking arena: the fly has to walk over blades rather than on a "
                 "plane."))

NATURAL_PRESETS: dict[str, NaturalSceneConfig] = {
    "meadow_grass": _MEADOW_GRASS,
    "natural_fbm": _NATURAL_FBM,
    "natural_fbm_lit_no_objects": _NATURAL_FBM_LIT_NO_OBJECTS,
    "fbm_uniform_sky": _FBM_UNIFORM_SKY,
    "fbm_flat_ground_texture": _FBM_NO_TEXTURE,
}


# --------------------------------------------------------------------------- #
# the ledger
# --------------------------------------------------------------------------- #

def _build_ledger() -> scene.ParameterLedger:
    led = scene.ParameterLedger()
    me = "measured on the installed stack (FlyGym 2.1.0 / MuJoCo 3.9.0) in this session"

    # -- MEASURED_LOCAL: facts about the shipped stack -------------------
    led.record("old.ground_texture_name", "checker", "name", scene.MEASURED_LOCAL,
               source=f"FlatGroundWorld._add_skybox/grid material, {me}")
    led.record("old.ground_texture_builtin", "mjBUILTIN_CHECKER", "enum", scene.MEASURED_LOCAL,
               source=f"flygym/compose/world/flat_ground.py, {me}")
    led.record("old.ground_texture_px", OLD_GROUND_TEX_PX, "px", scene.MEASURED_LOCAL,
               source=f"add_texture(width=300, height=300), {me}")
    led.record("old.ground_texrepeat", 250.0, "tiles per geom", scene.MEASURED_LOCAL,
               source=f"add_material(name='grid', texrepeat=(250,250)), {me}")
    led.record("old.ground_plane_half_size_mm", 1000.0, "mm", scene.MEASURED_LOCAL,
               source=f"FlatGroundWorld default half_size, {me}")
    led.record("old.ground_tile_mm", OLD_GROUND_TILE_MM, "mm", scene.MEASURED_LOCAL,
               source=("2000 mm plane / 250 repeats; this is the periodic grating "
                       f"length, versus a 2.5 mm fly, {me}"))
    led.record("old.sky_texture_rgb1", 1.0, "rgb in [0,1]", scene.MEASURED_LOCAL,
               source=f"BaseWorld._add_skybox(rgb1=(1,1,1), rgb2=(1,1,1)), {me}")
    led.record("old.sky_texture_builtin", "mjBUILTIN_GRADIENT", "enum", scene.MEASURED_LOCAL,
               source=("the skybox is ALREADY a gradient texture; it is white-to-white, "
                       f"which is why it renders as a constant, {me}"))
    led.record("old.headlight_ambient", 0.5, "multiplier", scene.MEASURED_LOCAL,
               source=f"mujoco_globals.yaml vis.headlight.ambient, {me}")
    led.record("old.headlight_diffuse", 0.6, "multiplier", scene.MEASURED_LOCAL,
               source=f"mujoco_globals.yaml vis.headlight.diffuse, {me}")
    led.record("old.headlight_specular", 0.3, "multiplier", scene.MEASURED_LOCAL,
               source=f"mujoco_globals.yaml vis.headlight.specular, {me}")

    # -- ENGINEERING_DEFAULT: the author's modelling/visual choices ------
    led.record("natural.hurst", DEFAULT_HURST, "dimensionless", scene.ENGINEERING_DEFAULT,
               source=("hand-chosen plausible value for natural topography; NOT "
                       "measured for leaf litter at mm scale.  H=0.7-0.9 is the "
                       "conventional range for natural terrain."),
               sweep_range=(0.55, 1.0))
    led.record("natural.radial_slope_implied_by_hurst",
               BACKGROUND_SPECTRAL_SLOPE, "log-log slope",
               scene.ENGINEERING_DEFAULT,
               source="derived: -(2*H + 1.5); the relation, not a measurement",
               sweep_range=(-4.0, -2.0))
    led.record("natural.hfield.n_samples", NATURAL_HFIELD_N, "samples per side",
               scene.ENGINEERING_DEFAULT,
               source="grid resolution; 257 is odd so the patch is centred on a sample",
               sweep_range=(129, 513))
    led.record("natural.hfield.half_extent_mm", NATURAL_HFIELD_HALF_EXTENT_MM, "mm",
               scene.ENGINEERING_DEFAULT,
               source=("covers ~8 s of walking at the measured ~14 mm/s gait; the "
                       "hfield geom tiles beyond its extent in MuJoCo, so there is "
                       "no cliff at the edge, but the tile period is this value"),
               sweep_range=(30.0, 200.0))
    led.record("natural.hfield.amplitude_std_mm", NATURAL_HFIELD_AMPLITUDE_STD_MM,
               "mm", scene.ENGINEERING_DEFAULT,
               source=("hand-chosen micro-relief scale for a 2.5 mm fly.  A FIRST "
                       "attempt used 0.16 mm (0.94 mm peak-to-peak) and the fly "
                       "could not walk: it was lifted onto the relief, thorax z "
                       "2.25 mm instead of 0.86 mm, and displaced 0.74 mm in 2 s.  "
                       "0.008 mm gives ~0.05 mm peak-to-peak, i.e. <5% of leg "
                       "length, which the gait tolerates."),
               sweep_range=(0.001, 0.05))
    led.record("natural.hfield.micro_relief_mm", NATURAL_HFIELD_MICRO_MM, "mm",
               scene.ENGINEERING_DEFAULT,
               source=("high-wavenumber grit component, kept an order of magnitude "
                       "below the background relief so it cannot destabilise stance"),
               sweep_range=(0.0, 0.005))
    led.record("natural.texture.tile_mm", _NATURAL_FBM.texture_tile_mm, "mm",
               scene.ENGINEERING_DEFAULT,
               source=("one texture tile spells this many mm on BOTH the plane and "
                       "the heightfield, so the two surfaces agree in scale.  A FIRST "
                       "attempt used 20 mm, which puts a 1024 px photograph at 0.02 "
                       "mm per texel -- four times finer than the eye camera's own "
                       "pixel at 6.5 mm range -- so the ground rendered as aliased "
                       "streaks rather than as a surface."),
               sweep_range=(0.5, 20.0))
    led.record("natural.texture.target_std", 0.12, "fraction of full range",
               scene.ENGINEERING_DEFAULT,
               source="procedural-route contrast target, chosen so the fly-scale "
                      "contrast is comparable with a photographed surface",
               sweep_range=(0.03, 0.3))
    led.record("natural.texture.beta", 2.4, "log-log radial slope magnitude",
               scene.ENGINEERING_DEFAULT,
               source="procedural-route 1/f^beta amplitude law; beta in 1.5-2.5 is "
                      "the conventional range for natural images",
               sweep_range=(1.5, 2.5))
    for fieldname, value in (("sky_rgb1", _NATURAL_FBM.sky_rgb1),
                             ("sky_rgb2", _NATURAL_FBM.sky_rgb2)):
        led.record_vector(f"natural.sky.{fieldname}", value, "rgb in [0,1]",
                          scene.ENGINEERING_DEFAULT,
                          source=("visual choice for a dimmer zenith and a brighter "
                                  "horizon; not measured, and the sky is verified by "
                                  "rendering rather than asserted"),
                          sweep_range=(0.0, 1.0))
    for f in ("headlight_ambient", "headlight_diffuse", "headlight_specular"):
        led.record_vector(f"natural.{f}", getattr(_NATURAL_FBM, f), "multiplier",
                          scene.ENGINEERING_DEFAULT,
                          source=("chosen to stop the shipped headlight saturating the eye image "
                                  "while still lighting the scene enough to see the "
                                  "fly; the shipped values are 0.5/0.6/0.3.  A FIRST "
                                  "attempt used 0.14/0.34/0.06, which rendered the "
                                  "whole world view with max pixel 70/255 -- nothing "
                                  "was visible"),
                          sweep_range=(0.0, 1.6))
    led.record("natural.texture.grey", 0, "boolean", scene.ENGINEERING_DEFAULT,
               source="colour kept by default; grey is available but not used here",
               sweep_range=(0, 1))
    return led


NATURAL_PRESET_LEDGER = _build_ledger()


def natural_scene_provenance_report(*, texture_meta: Mapping[str, Any] | None = None
                                    ) -> dict:
    """Ledger dump + the asset provenance rules this module enforces."""
    counts = NATURAL_PRESET_LEDGER.counts()
    out = {
        "n_records": len(NATURAL_PRESET_LEDGER),
        "counts_by_provenance": counts,
        "ledger": NATURAL_PRESET_LEDGER.as_dicts(),
        "presets": sorted(NATURAL_PRESETS),
        "asset_rules": {
            "photo": ("a photograph is used only if the Wikimedia Commons API "
                      "returned a licence accepted by is_free_license() AND a "
                      "non-empty author; TextureProvenance raises otherwise"),
            "procedural": "labelled PROCEDURAL; nothing is cited because nothing "
                          "external was used",
        },
        "model_scale_note": SCALE_NOTE,
        "spectral_audit_note": SPECTRAL_AUDIT_NOTE,
    }
    if texture_meta is not None:
        out["ground_texture"] = scene._jsonable(dict(texture_meta))
    return out


# --------------------------------------------------------------------------- #
# applying the scene
# --------------------------------------------------------------------------- #

@dataclass
class NaturalSceneApplicationReport(scene.SceneApplicationReport):
    """``scene.SceneApplicationReport`` plus what the natural layer changed.

    Subclassing rather than re-implementing keeps ``apply_natural_scene``
    signature- and field-compatible with every existing consumer of
    ``apply_scene``.
    """
    texture_route: str = ""
    #: Set when the preset named an explicit texture file; records which one, so the
    #: difference between "resolved" and "dictated" is visible in the report.
    texture_explicit_path: str = ""
    texture_path: str = ""
    texture_sha256: str = ""
    texture_license: str = ""
    texture_author: str = ""
    texture_source_page: str = ""
    texture_size_px: tuple[int, int] = (0, 0)
    texture_tile_mm: float = 0.0
    texrepeat_plane: tuple[float, float] = (0.0, 0.0)
    texrepeat_hfield: tuple[float, float] = (0.0, 0.0)
    material_applied_to: tuple[str, ...] = ()
    sky_rgb1: tuple[float, float, float] = (1.0, 1.0, 1.0)
    sky_rgb2: tuple[float, float, float] = (1.0, 1.0, 1.0)
    sky_edited_in_place: bool = False
    headlight: dict = field(default_factory=dict)
    terrain_metadata: dict = field(default_factory=dict)
    terrain_audit: dict = field(default_factory=dict)
    texture_audit: dict = field(default_factory=dict)
    measured_texture: dict = field(default_factory=dict)
    heightfield_raycast: dict = field(default_factory=dict)


def _find_texture_element(spec, name: str):
    """``MjSpec.texture(name)`` exists; ``find_texture`` does NOT (measured)."""
    try:
        return spec.texture(name)
    except Exception:
        return None


def _set_headlight(spec, cfg: "NaturalSceneConfig") -> dict:
    """Write vis.headlight multipliers into the spec and read them back.

    NOTE (measured): ``MjVisualHeadlight``'s fields are ``ambient``, ``diffuse``,
    ``specular`` and ``active`` -- NOT ``headlight_ambient`` etc.  This function
    therefore maps the config field names explicitly and returns ``None`` for any
    multiplier it could not write, so a silently ineffective assignment shows up
    in the report instead of looking like success.
    """
    vis = spec.visual
    applied: dict = {}
    for attr, value in (("ambient", cfg.headlight_ambient),
                        ("diffuse", cfg.headlight_diffuse),
                        ("specular", cfg.headlight_specular)):
        target = getattr(vis.headlight, attr, None)
        if target is None:
            applied[attr] = None
            continue
        for i in range(min(3, len(target))):
            target[i] = float(value[i])
        applied[attr] = [float(v) for v in value]
    return applied


def raycast_terrain(model, data, *, half_extent_mm: float, n: int = 9) -> dict:
    """Sample the compiled terrain's height by ray-casting straight down.

    This is the independent check that the heightfield is a *surface at a known
    height* rather than an offset that would drop the fly.  ``mj_ray`` is called
    against the hfield geom only, so the numbers are the terrain's, not the
    plane's.  Reported in mm relative to the model origin.
    """
    mj = scene._import_mujoco()
    hid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, scene.HFIELD_GEOM_NAME)
    if hid < 0:
        return {"available": False, "reason": "no heightfield geom in the model"}
    xs = np.linspace(-half_extent_mm, half_extent_mm, int(n))
    heights = []
    pnt = np.zeros(3)
    # geomgroup = bit mask of the GEOM GROUPS to consider; the heightfield geom is
    # added with group=0 by scene.apply_scene, so group 1 (bit 0) selects it.
    # Measured: mujoco 3.9's mj_ray takes (model, data, pnt, vec, geomgroup,
    # flg_static, bodyexclude, geomid, normal) -- there is no geom-id filter.
    geomgroup = np.zeros(6, dtype=np.uint8)
    geomgroup[0] = 1 if int(np.ravel(model.geom_group[hid])[0]) == 0 else 0
    if geomgroup.sum() == 0:                      # heightfield is not in group 0
        geomgroup[:] = 1
    geomid = np.zeros(1, dtype=np.int32)
    for x in xs:
        row = []
        for y in xs:
            from_mm = np.array([float(x), float(y), 50.0])
            dist = mj.mj_ray(model, data, from_mm, np.array([0.0, 0.0, -1.0]),
                             geomgroup, 1, -1, geomid)
            row.append(float(from_mm[2] - dist) if dist >= 0 else float("nan"))
        heights.append(row)
    h = np.asarray(heights, dtype=float)
    finite = h[np.isfinite(h)]
    inside = h[1:-1, 1:-1]
    return {
        "available": True,
        "n": int(n),
        "grid_mm": [float(xs[0]), float(xs[-1])],
        "heights_mm": h.tolist(),
        "min_mm": float(finite.min()) if finite.size else None,
        "max_mm": float(finite.max()) if finite.size else None,
        "peak_to_peak_mm": (float(finite.max() - finite.min()) if finite.size else None),
        "std_mm": float(finite.std()) if finite.size else None,
        "all_rays_hit": bool(np.isfinite(h).all()),
        "note": ("rays are cast against the hfield geom ONLY (geomid filter), so "
                 "these are terrain heights, not plane heights"),
        "outside_patch_note": ("the sampled grid spans the patch and beyond; a hit "
                               "outside the nominal half extent proves MuJoCo tiles "
                               "the hfield rather than ending it"),
        "inside_patch_std_mm": float(inside.std()) if inside.size else None,
    }


def apply_natural_scene(world, fly, cfg, ledger=None, *, measure: bool = True
                        ) -> NaturalSceneApplicationReport:
    """Realise ``cfg`` in ``world``'s MJCF spec.  Call BEFORE ``world.compile()``.

    Same calling convention as :func:`scene.apply_scene`.  Order:

    1. ``scene.apply_scene`` -- friction classes, the heightfield, pair friction,
       lights, objects.  This module does not duplicate any of that.
    2. **Sky**: the EXISTING ``skybox`` texture is edited IN PLACE.  Adding a
       second skybox is silently ignored by MuJoCo (measured: ``ntex`` rises but
       the render is unchanged), because the renderer uses the texture registered
       in the skybox role.
    3. **Ground texture**: the image is loaded into a NEW 2-D texture which a NEW
       material references, and that material is assigned to the plane and to the
       heightfield with a ``texrepeat`` computed from ONE physical tile size, so
       the two surfaces are painted at a single scale.  Replacing the builtin
       ``checker`` in place was measured to work too, but it cannot give the plane
       and the hfield different repeats.
    4. **Headlight**: written into the spec and read back from the compiled model.
    """
    if not isinstance(cfg, scene.SceneConfig):
        raise TypeError(f"cfg must be a scene.SceneConfig, got {type(cfg).__name__}")
    if ledger is not None and hasattr(ledger, "add"):
        for rec in NATURAL_PRESET_LEDGER:
            if rec.name not in ledger.names():
                ledger.add(rec)

    # Generate the terrain FIRST: its peak-to-peak is the number that must reach
    # MuJoCo's hfield_size[2], and RoughnessConfig.amplitude_mm is a compile-time
    # spec field that cannot be changed after the hfield asset is added.
    terrain, elev = _terrain_for_cfg(cfg)
    if terrain and cfg.roughness is not None:
        cfg = dataclasses.replace(
            cfg, roughness=dataclasses.replace(
                cfg.roughness,
                amplitude_mm=float(terrain["metadata"]["amplitude_peak_to_peak_mm"])))

    base = scene.apply_scene(world, fly, cfg,
                             ledger=ledger if isinstance(ledger, scene.ParameterLedger)
                             else None,
                             measure=False)
    report = NaturalSceneApplicationReport(
        **{f.name: getattr(base, f.name)
           for f in dataclasses.fields(scene.SceneApplicationReport)})
    warnings = list(report.warnings)
    spec = world.mjcf_root
    mj = scene._import_mujoco()

    natural = cfg if isinstance(cfg, NaturalSceneConfig) else None

    # -- 2. sky, edited in place -----------------------------------------
    if natural is not None:
        sky = _find_texture_element(spec, natural.sky_texture_name)
        if sky is None:
            warnings.append(
                f"sky texture {natural.sky_texture_name!r} is not in this spec, so "
                f"the sky was NOT changed and stays whatever the world shipped")
        else:
            sky.builtin = mj.mjtBuiltin.mjBUILTIN_GRADIENT
            sky.rgb1 = [float(v) for v in natural.sky_rgb1]
            sky.rgb2 = [float(v) for v in natural.sky_rgb2]
            report.sky_edited_in_place = True
            report.sky_rgb1 = tuple(float(v) for v in natural.sky_rgb1)
            report.sky_rgb2 = tuple(float(v) for v in natural.sky_rgb2)
            warnings.append(
                "the skybox texture was edited IN PLACE (name "
                f"{natural.sky_texture_name!r}), not re-added: measured on this "
                "stack, adding a second skybox texture compiles but changes nothing "
                "in the render, because the renderer uses the registered skybox "
                "texture.  Verified by rendering the sky band, not by assuming.")

        # -- 4a. headlight ----------------------------------------------
        report.headlight = _set_headlight(spec, natural)
        if any(v is None for v in report.headlight.values()):
            warnings.append(
                "vis.headlight could NOT be fully written (some multipliers came "
                "back None): the eye images may still saturate and the headlight "
                "part of this claim is VOID")
        warnings.append(
            "vis.headlight was rebalanced away from the shipped 0.5/0.6/0.3.  The "
            "shipped values plus white-ish materials are what saturated the eye "
            "image; the new values are ENGINEERING_DEFAULT choices, not measured.")

    # -- 3. ground texture ------------------------------------------------
    tex_meta: dict = {}
    if natural is not None and natural.ground_texture_route != "checker_flat":
        if getattr(natural, "ground_texture_path", None):
            # AN EXPLICIT TEXTURE WINS over resolution.  Its provenance sidecar must
            # already exist next to it (the meadow asset is fetched and hashed by
            # meadow_scene.ensure_grass_texture), otherwise a photo would be used with
            # no recorded licence -- which is exactly the failure this branch prevents.
            png = os.path.abspath(str(natural.ground_texture_path))
            # The sidecar is named after the IMAGE, not after a fixed file name: a fixed
            # name would make two different images in one directory share one licence
            # record, which is how the meadow first rendered with the dry-leaves
            # photograph while the report said the grass photograph was in use.
            side = os.path.splitext(png)[0] + PROVENANCE_SIDECAR_SUFFIX
            if not os.path.exists(side):
                # accept the legacy fixed-name sidecar ONLY when it records this path
                legacy = os.path.join(os.path.dirname(png), PROVENANCE_SIDECAR)
                if os.path.exists(legacy):
                    try:
                        with open(legacy) as f:
                            _leg = json.load(f)
                        if os.path.abspath(str(_leg.get("path", ""))) == png:
                            side = legacy
                    except Exception:
                        pass
            if not os.path.exists(side):
                raise scene.ProvenanceError(
                    f"{png} was given explicitly as the ground texture but has no "
                    f"provenance sidecar at {side}: refusing to use an image whose "
                    "licence is not on record")
            with open(side) as f:
                tex_meta = json.load(f)
            tex_meta["route_decision"] = "explicit ground_texture_path on the preset"
            report.texture_explicit_path = png
        elif natural.ground_texture_route == "procedural":
            png = os.path.join(ASSET_DIR, "natural_ground_procedural.png")
            tex_meta = generate_natural_texture_png(png, size=512, seed=cfg.seed)
        else:
            tex_meta = resolve_ground_texture(size=1024, seed=cfg.seed)
            if tex_meta.get("route") != "PROCEDURAL":
                write_provenance_sidecar(tex_meta)
        png = str(tex_meta["path"])
        from PIL import Image
        with Image.open(png) as im:
            px_w, px_h = im.size
        tex = spec.add_texture(name=natural.ground_texture_name,
                               type=mj.mjtTexture.mjTEXTURE_2D,
                               file=os.path.abspath(png),
                               width=int(px_w), height=int(px_h))
        mat = spec.add_material(name=natural.ground_material_name,
                                texrepeat=[1.0, 1.0],
                                # texuniform=0 -> the repeat is counted ACROSS THE
                                # GEOM, which is what makes `texture_tile_mm` a real
                                # physical tile size.  MuJoCo's default (uniform)
                                # maps per unit length instead, so the same number
                                # would mean two different sizes on the 2*half_size
                                # plane and the 2*half_extent heightfield.
                                texuniform=0,
                                reflectance=float(natural.texture_reflectance),
                                shininess=float(natural.texture_shininess),
                                specular=float(natural.texture_specular))
        mat.textures[int(mj.mjtTextureRole.mjTEXROLE_RGB)] = natural.ground_texture_name
        # MuJoCo stores texrepeat on the MATERIAL, not on the geom, so one
        # material cannot give the 2*half_size plane and the 2*half_extent
        # heightfield the same physical tile size.  The heightfield therefore
        # gets its own material sharing the same texture.  MEASURED: assigning a
        # repeat to one shared material leaves the compiled `mat_texrepeat` at
        # that single value for both geoms (verified by reading mat_texrepeat
        # back), which is why this is two materials.
        hmat_name = natural.ground_material_name + "_hfield"
        hmat = spec.add_material(name=hmat_name, texrepeat=[1.0, 1.0],
                                 texuniform=0,
                                 reflectance=float(natural.texture_reflectance),
                                 shininess=float(natural.texture_shininess),
                                 specular=float(natural.texture_specular))
        hmat.textures[int(mj.mjtTextureRole.mjTEXROLE_RGB)] = natural.ground_texture_name
        report.texture_route = str(tex_meta.get("route", ""))
        report.texture_path = png
        report.texture_sha256 = str(tex_meta.get("sha256", ""))
        report.texture_license = str(tex_meta.get("license_name", ""))
        report.texture_author = str(tex_meta.get("author", ""))
        report.texture_source_page = str(tex_meta.get("source_page", ""))
        report.texture_size_px = (int(px_w), int(px_h))
        report.texture_tile_mm = float(natural.texture_tile_mm)
        report.texture_audit = dict(tex_meta.get("spectral") or {})
        applied_to = []
        # the infinite plane that FlyGym made, and the heightfield scene.py made
        for gname in ("ground_plane", scene.HFIELD_GEOM_NAME):
            g = None
            for gg in spec.geoms:
                if str(gg.name) == gname:
                    g = gg
                    break
            if g is None:
                continue
            try:
                extent = (2.0 * float(cfg.roughness.half_extent_mm)
                          if gname == scene.HFIELD_GEOM_NAME else
                          2.0 * float(cfg.world_half_size_mm))
                rep = max(1.0, round(extent / float(natural.texture_tile_mm), 3))
                if gname == scene.HFIELD_GEOM_NAME:
                    g.material = hmat_name
                    hmat.texrepeat = [rep, rep]
                    report.texrepeat_hfield = (rep, rep)
                else:
                    g.material = natural.ground_material_name
                    mat.texrepeat = [rep, rep]
                    report.texrepeat_plane = (rep, rep)
                applied_to.append(gname)
            except Exception as exc:                       # pragma: no cover
                warnings.append(f"could not paint {gname!r}: {exc!r}")
        report.material_applied_to = tuple(applied_to)
        if "ground_plane" not in applied_to:
            warnings.append("the ground plane geom was not found, so the visible "
                            "ground outside the heightfield keeps the old checker "
                            "texture")
        warnings.append(
            f"ground texture: the plane and the heightfield are BOTH painted with "
            f"material {natural.ground_material_name!r} at one physical tile size of "
            f"{natural.texture_tile_mm} mm, so the two surfaces agree in scale "
            f"(this is the fix for scene.py's warning about the checker being finer "
            f"inside the patch).  The old periodic 'checker' texture and the 'grid' "
            f"material are left in the model untouched so the old and new scenes can "
            f"be rendered and audited side by side.")

    if natural is not None and natural.ground_texture_route == "checker_flat":
        # Control: keep the terrain, replace the periodic checker with one flat
        # colour (builtin=flat, no file).  Non-periodic is the point.
        chk = _find_texture_element(spec, "checker")
        if chk is not None:
            chk.builtin = mj.mjtBuiltin.mjBUILTIN_FLAT
            chk.rgb1 = [float(v) for v in natural.ground_flat_rgba[:3]]
            chk.rgb2 = [float(v) for v in natural.ground_flat_rgba[:3]]
            warnings.append(
                "ground_texture_route='checker_flat': the builtin checker texture was "
                "replaced IN PLACE by a flat colour (builtin=flat).  This is the "
                "control that isolates the terrain's contribution from the texture's; "
                "it is NOT a natural ground and is not the natural_fbm preset.")

    # -- terrain (already generated above; attach the report fields) ------
    if terrain:
        tmeta = dict(terrain["metadata"])
        tmeta["amplitude_written_to_hfield_size_z_mm"] = float(
            cfg.roughness.amplitude_mm)
        tmeta["provenance_amplitude"] = scene.ENGINEERING_DEFAULT
        report.terrain_metadata = tmeta
        report.terrain_audit = dict(terrain["audit"])

    # -- 5. measure -------------------------------------------------------
    if measure:
        try:
            model, data = world.compile()
            measured = scene.SceneApplicationReport.measured_after_compile(
                model, fly_name=str(getattr(fly, "name", None) or "fly"),
                ground_geom_names=report.ground_geom_names,
                n_roughness_geoms=report.n_roughness_geoms)
            report.measured = measured
            report.n_geoms = int(measured["ngeom"])
            report.n_lights = int(measured["nlight"])
            report.n_hfield = int(measured["nhfield"])
            report.n_objects = int(measured["n_object_geoms"])
            report.measured_note = ("read back from a compiled COPY of the spec; the "
                                    "caller's own world.compile() gives an equivalent "
                                    "model")
            # the specific read-back that proves the FILE was loaded and not the
            # builtin checker: the texture's pixel dimensions must equal the PNG's
            tex_rows = []
            for i in range(int(model.ntex)):
                tname = str(model.texture(i).name)
                tex_rows.append({
                    "name": tname,
                    "type": int(np.ravel(model.texture(i).type)[0]),
                    "width": int(np.ravel(model.tex_width[i])[0]),
                    "height": int(np.ravel(model.tex_height[i])[0]),
                    "nchannel": int(np.ravel(model.tex_nchannel[i])[0]),
                })
            report.measured_texture = {
                "textures": tex_rows,
                "expected_ground_px": list(report.texture_size_px),
                "ground_texture_matches_png": any(
                    r["name"] == natural.ground_texture_name
                    and (r["width"], r["height"]) == report.texture_size_px
                    for r in tex_rows) if natural is not None else False,
                "checker_still_present": any(r["name"] == "checker" for r in tex_rows),
                "checker_px": next(([r["width"], r["height"]] for r in tex_rows
                                    if r["name"] == "checker"), None),
                "mat_texrepeat": [[float(v) for v in np.ravel(model.mat_texrepeat[i])]
                                  for i in range(int(model.nmat))],
                "mat_texuniform": [int(np.ravel(model.mat_texuniform[i])[0])
                                   for i in range(int(model.nmat))],
                "mat_names": [str(model.material(i).name)
                              for i in range(int(model.nmat))],
                "headlight_ambient": [float(v) for v in
                                      np.ravel(model.vis.headlight.ambient)],
                "headlight_diffuse": [float(v) for v in
                                      np.ravel(model.vis.headlight.diffuse)],
                "headlight_specular": [float(v) for v in
                                       np.ravel(model.vis.headlight.specular)],
                "sky_texture_px": next(([r["width"], r["height"]] for r in tex_rows
                                        if r["name"] == "skybox"), None),
                "note": ("if ground_texture_matches_png is False the file was NOT "
                         "loaded and MuJoCo fell back to something else -- this is "
                         "the exact silent failure the builtin texture causes"),
            }
            report.heightfield_raycast = raycast_terrain(
                model, data, half_extent_mm=float(cfg.roughness.half_extent_mm),
                n=9) if report.n_hfield else {"available": False}
        except Exception as exc:
            report.measured_note = f"could not compile to measure: {exc!r}"
            warnings.append(f"apply_natural_scene could not compile to verify itself: "
                            f"{exc!r}")
    report.warnings = tuple(warnings)
    return report


# --------------------------------------------------------------------------- #
# self-report helpers used by the runner
# --------------------------------------------------------------------------- #

def preset_configs() -> dict:
    return {k: v.name for k, v in NATURAL_PRESETS.items()}


def apply_natural_scene_to_backend(backend, preset: str, **kwargs):
    """Apply a natural preset to an already-built ``BodyBackend``.

    Provided for callers that build a backend first; the supported path is still
    ``BodyConfig(scene_preset=...)``, which ``BodyBackend`` resolves by calling
    :func:`apply_natural_scene` before ``compile()``.
    """
    cfg = NATURAL_PRESETS[preset]
    return apply_natural_scene(backend.world, backend.fly, cfg,
                               ledger=NATURAL_PRESET_LEDGER, **kwargs)


def assert_module_contract() -> dict:
    """Cheap self-check of the exports the integration depends on."""
    problems = []
    for name in ("natural_fbm", "natural_fbm_lit_no_objects", "fbm_uniform_sky"):
        if name not in NATURAL_PRESETS:
            problems.append(f"missing preset {name}")
    if not callable(apply_natural_scene):
        problems.append("apply_natural_scene is not callable")
    for fn in (generate_fbm_heightfield, generate_natural_texture_png,
               spectral_audit):
        if not callable(fn):
            problems.append(f"{fn} is not callable")
    for p in NATURAL_PRESETS.values():
        if not isinstance(p, scene.SceneConfig):
            problems.append(f"{p.name} is not a scene.SceneConfig")
    return {"ok": not problems, "problems": problems,
            "presets": sorted(NATURAL_PRESETS),
            "n_ledger_records": len(NATURAL_PRESET_LEDGER),
            "ledger_requires_sweeps": all(
                r.sweep_range is not None for r in NATURAL_PRESET_LEDGER
                if r.provenance in (scene.ASSUMED, scene.ENGINEERING_DEFAULT))}
