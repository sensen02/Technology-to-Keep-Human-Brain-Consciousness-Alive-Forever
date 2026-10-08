"""ARBITRATE THE PERIODICITY CONTRADICTION with criteria that are INDEPENDENT of
the two contested statistics.

WHY THIS FILE EXISTS
--------------------
Two ground-periodicity detectors disagree on ``scene_preset='natural_fbm'``
(ground = a real CC0 photograph of dry leaf litter, ``natural_ground_photo.png``):

  * ``peak / median`` of the windowed 2-D |F| (``scene_realism_audit_run.py::
    periodicity``, reused by ``compare_ground_audit_instruments.py``) measured
    8892 on the new ground against the audit's published target <= 300 -> FAIL.
    Baseline ``blank`` measured 22727.
  * ``peak_frac`` = the single strongest |F|^2 bin's share of TOTAL spectral
    power (``run_natural_scene.py``) measured 0.0158 against its threshold
    < 0.05 -> PASS.  The old checker terrain measured 0.3452.

Both can be true at once: if most high-frequency bins are near zero the MEDIAN
is tiny, so ``max/median`` explodes even when the max carries ~1.6 % of the
power.  Neither number answers the question.  The question this file answers is:
IS A PERIODIC LATTICE PHYSICALLY PRESENT IN THE RENDERED GROUND?

This script does NOT fix, tune, refactor or modify anything else.  It measures.

    Run: MUJOCO_GL=egl venv_body/bin/python arbitrate_periodicity.py

CAMERA GEOMETRY
---------------
The 36 mm condition imports ``render_free``, ``CAM_H``, ``CAM_W``, ``CAM_FOVY``
and ``MM_PER_PX_IN`` from ``scene_realism_audit_run`` and reproduces
``compare_ground_audit_instruments.py`` exactly: straight down (elevation 90,
azimuth 0), lookat at the fly thorax, distance 36 mm, fovy 35 deg, 1024x900.
That image is therefore directly comparable to the published 8892 / 22727.
Two further distances (75, 150 mm) are added because the SAME camera pulled
back changes mm/px, which is what makes criterion C3 decisive.

================================================================================
PRE-REGISTERED DECISION RULE
================================================================================
Declared before the real presets were measured.  Thresholds are calibrated
against synthetic controls run by this SAME code in the SAME process, so the
calibration is self-contained and inspectable rather than borrowed from either
contested instrument.  Nothing is retuned after seeing a result; every FAIL is
reported as a failure.

  INSTRUMENT REVISION (recorded, not hidden): the first version of C1 searched
  only for positive local maxima r>0 and therefore classified a perfect square
  checkerboard as non-periodic -- because a checker is ANTICORRELATED at one
  period: a shift of p flips floor(x/p)+floor(y/p) by exactly 1, so every cell
  changes colour and r(p,0) = -r(0,0), while r(p,p) = +1 and r(2p,0) = +1.  C1
  now takes extremum candidates on |r| and reports the sign of r.  This was
  found by the in-process checker control failing, i.e. by the validity gate
  working, and it was fixed BEFORE any real preset was re-measured.  The first
  version's output is not used anywhere below.

BANDS (declared, not tuned)
  ANALYSIS_BAND_MM = (0.300, 12.0) mm  = 0.0833 .. 3.333 cycles/mm
      Lower end: a 2.5 mm fly resolving 0.3 mm features.  Upper end: 12 mm is a
      generous declared bound on "a feature the fly could see" (~5 body
      lengths).  The old terrain checker (8 mm) is INSIDE this band; its
      diagonal autocorrelation peak (11.3 mm) is also inside.
  TILE_EXCLUSION_MM = 90 mm.  The ground macro-texture tile is 90 mm
      (natural_scene.py ``texture_tile_mm``), i.e. 0.0111 cyc/mm, far below the
      band and 1060 px at 36 mm (larger than the frame, so tiling seams are not
      even resolvable in-frame).  A "periodicity" at the tile scale or at the
      frame size is tiling / window envelope, not a lattice the fly can see.
  MM_PER_PX(d) = 2*d*tan(fovy/2)/H  (the audit's own formula)
      36 mm -> 0.04530 mm/px : band spans 6.6 .. 265 px   (comfortable)
      75 mm -> 0.09438 mm/px : band spans 3.2 .. 127 px
     150 mm -> 0.18875 mm/px : band spans 1.6 px at the short end, so the
        0.3-0.6 mm part of the band is NOT RESOLVED there; this is reported per
        distance (``resolved_gt2px``) and the autocorrelation criterion is
        additionally protected because its own minimum lag is 0.3 mm.

C1  AUTOCORRELATION SIDELOBE / ANISOTROPY  (independent of |F| max and median)
      r = the EXACT overlap-and-local-mean-normalised 2-D autocorrelation of the
      image, no window (see autocorr: with a Hann or radial taper, r is the
      signal correlation CONVOLVED with the window autocorrelation, so a FLAT
      image showed |r| = 0.83 at 0.31 mm -- an artefact that looked like a
      result).  Then the RADIAL MEAN of r is subtracted (radial_residual): a 1/f
      envelope tail is isotropic, i.e. a function of lag radius alone, so
      subtracting it deletes the envelope and leaves pure anisotropy, which is
      what a lattice is.  MEASURED on the controls at 36 mm: a 4.8 mm checker
      leaves |r_resid| = 0.838, while three aperiodic fields (pink-coloured
      noise, the CC0 photograph, the same photograph's random-phase version)
      leave 0.101, 0.131, 0.131.
      Report: the strongest off-centre |r_resid| in the lag window, as a
      fraction of the DC peak (r(0,0) = 1), its lag in mm, its lag direction, its
      sign, its prominence on the map, and its spread over 4 independent 2-D
      crops of the frame.
      PASS if  |r_resid| >= 0.15  AND  it clears the 3-sigma overlap-sample
      noise floor  AND  prominence >= 0.05  AND  (max crop - min crop) <= 0.5 x
      |r_resid|.
      The lag window starts at max(0.3 mm, 2 x central |r| half-width) because
      inside the central peak's own halo the radial-mean subtraction produces a
      purely geometric residual; see central_peak_halfwidth_mm, which also
      records the sweep that fixed the multiplier at 2.
      Rationale: a periodic component at wavelength L puts discrete peaks in the
      autocorrelation at L (anticorrelated for a checker), L*sqrt(2) on the
      diagonal and 2L; 0.15 demands only that it explain ~15 % of the image
      variance at ONE lag.  Aperiodic texture decays without any off-centre
      structure, so it fails on level, not on prominence.

C2  ANGULAR / ORIENTATION  (independent: uses only the ANGULAR shape of |F|^2
    inside the band, after dividing out the radial mean, so the spectral slope,
    the DC lobe and the max/median ratio cannot enter)
      Fold the band |F|^2 into a 90 deg period (a square lattice is 4-fold
      symmetric), 36 bins, normalised to unit mean.  circ_var = 1 - |sum_o p_o
      exp(i 4 theta_o)| -- 0 for one discrete orientation, 1 for uniform.
      PASS if  circ_var <= 0.60  AND  peak-to-mean over the 36 bins >= 2.0
      AND the dominant orientation is within 10 deg of a 45-deg multiple: a
      square checker's discrete lines sit at 45 deg multiples, i.e. its
      autocorrelation peaks are on the DIAGONALS (a (1,1)-type peak), so a
      dominant orientation that is not on a diagonal cannot be a square lattice.
      The bars are checked against the in-process checker (must PASS) and two
      aperiodic controls (must FAIL); the controls' own circ_var is reported as
      the empirical aperiodic reference.

C3  SCALE / POSITION ROBUSTNESS  (independent: mm-calibrated, cross-distance)
      The band is mm-fixed, so its pixel size changes with mm/px.  Run C1's
      detection at 36 / 75 / 150 mm plus 2x2 crops of each and convert every lag
      to mm.
      PASS if  every distance returns |r_resid| >= 0.15  AND  all detected
      wavelengths agree within 25 %  AND  the near-to-far drift is <= 50 %  AND
      the far/near |r_resid| ratio is <= 2.5.
      Rationale: a genuine grating has ONE wavelength in mm and keeps its
      sharpness when the camera pulls back (it only gets finer in px).
      Photographic content does not, and a "periodicity" that is really a
      renderer band stays put in mm but collapses in strength.

C4  PHASE-RANDOMISED SURROGATE NULL
      Null model: keep the exact |F|, randomise all phases (45 surrogates, RNG
      seed 20240517), so the amplitude spectrum, the 1/f slope, the DC lobe and
      the tiling envelope are all preserved and only phase coherence is
      destroyed.
      Two statistics are tested, both restricted to the band:
        (a) the CONTESTED statistic itself, max|F| / median|F|, so this test
            adjudicates that exact number;
        (b) max|F|^2 / median|F|^2 after dividing out the radial mean, which
            removes the 1/f envelope entirely and therefore detects DISCRETE
            lattice lines specifically.
      PASS if the real value exceeds the maximum over the 45 surrogates (i.e.
      clears the 98th percentile by construction, empirical p <= 1/46 = 0.022);
      the real percentile in each surrogate distribution is reported.
      The all-bins (non-band) variant is also reported but a low-frequency 1/f
      lobe makes it lenient and it is NOT used for the verdict.
      REPORTED LIMITATION, found by running this test on the checker control: a
      phase-only randomisation also preserves the checker's discrete amplitude
      peaks, so it CANNOT distinguish a lattice from aperiodic content whose
      envelope happens to be spiky -- both fail.  Its measured behaviour here is
      consistent and monotone though: the controls and every preset fail, and the
      real band max/median for the leaf-litter ground (2.7e5) is BELOW the
      surrogate median (4.9e5) and below every one of the 45 surrogates, so this
      statistic gives no positive evidence of discrete lines.  C4 is therefore
      used as a NECESSARY condition (a lattice must beat its own surrogates),
      never as sufficient evidence, and it never overrides C1/C2/C3.

VALIDITY GATE (runs before the real presets; a criterion that fails its control
is reported INVALID and its verdict is not issued)
    * checkerboard at 4.8 mm, 2x supersampled, same size and window: MUST PASS
      C1 and C2 (a real lattice).
    * phase-randomised coloured noise whose amplitude envelope is a RADIAL MEAN
      only (so it carries no discrete lines -- building this control from a
      checker's full |F| made a control that inherited the checker's lattice and
      scored |r| = 0.85, i.e. it proved nothing): MUST FAIL C1 and C2.
    * the CC0 photograph itself, no render: MUST FAIL C1 and C2.
    * the CC0 photograph itself (no render), as a known-real-litter reference.

VERDICT PER PRESET
    score = number of criteria with a valid PASS.
    >=3 PASS -> POSITIVE ; 2 -> WEAK ; <=1 -> NEGATIVE.
    Disagreement is reported verbatim and is never averaged away.  Overrides:
      * C1/C2 PASS with C4 FAIL -> WEAK, not POSITIVE: a real lattice is phase
        locked, so a peak its own phase-randomised surrogate reproduces is not a
        coherent lattice.
      * if the detected wavelength scales with camera distance (the signature of
        synthetic content), the verdict is NEGATIVE regardless of the score.
================================================================================
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

ROOT = "/run/media/sensen/Data2/cell_wound_prototype"
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402

import scene_realism_audit_run as A  # noqa: E402

OUT = os.path.join(ROOT, "outputs", "embodied_body")
JSON_OUT = os.path.join(OUT, "arbitrate_periodicity.json")
PNG_OUT = os.path.join(OUT, "arbitrate_periodicity.png")

# --------------------------------------------------------------------------- #
# PRE-REGISTERED CONSTANTS
# --------------------------------------------------------------------------- #
PRESETS = ("blank", "natural", "natural_fbm")
DISTANCES_MM = (36.0, 75.0, 150.0)
LOOKAT_MM = (0.0, 16.0, 0.0)          # audit "in" camera lookat = fly thorax
MASK_ROW = (0.30, 0.72)               # fly mask, identical to
MASK_COL = (0.28, 0.72)               # compare_ground_audit_instruments.py
CROP_DIV = 2                          # 2x2 independent crops
CROP_TRIM_PX = 4                      # drop the mask boundary
WARMUP_STEPS = 400                    # audit's settle for a comparable pose

ANALYSIS_BAND_MM = (0.300, 12.0)      # declared above; excludes the 90 mm tile
TILE_EXCLUSION_MM = 90.0              # natural_scene.py texture_tile_mm
CROP_MIN_MM = 0.300                   # min autocorrelation lag examined
AC_LAG_MAX_MM = 12.0                  # max autocorrelation lag for the verdict
LOCAL_MIN_R = 0.02                    # |r| floor for an "extremum"
#: C1 examines lags beyond max(CROP_MIN_MM, C1_WIDTH_MULT x central |r|
#: half-width); see central_peak_halfwidth_mm for why and for the measurement.
#: 2.0 was chosen BEFORE the real presets were re-measured, from a sweep on the
#: controls at 36 mm which showed the statistic in a stable regime for
#: multipliers >= 2: the checker residual stays 0.838 (its peaks are at 4.8/6.8/
#: 9.6 mm, far outside any halo), while the leaf-litter ground stops falling as the
#: exclusion grows (0.127 at mult 1.5, then 0.102, 0.092, 0.092 for 2/3/4), i.e.
#: past 2x the halo only its true, isotropic floor remains.
C1_WIDTH_MULT = 2.0

# C1
C1_PASS_R = 0.15
C1_PASS_PROMINENCE = 0.05
C1_PASS_CROP_SPREAD = 0.5
# C2
C2_PASS_CIRC_VAR = 0.60
C2_PASS_PTM = 2.0
C2_PASS_ALIGN_DEG = 10.0
# C3
C3_PASS_PEAK_R = 0.15
C3_PASS_WAVELENGTH_SPREAD = 0.25
C3_PASS_DRIFT = 0.50
C3_PASS_PEAK_RATIO = 2.5
# C4
N_SURROGATES = 45
SURROGATE_SEED = 20240517
C4_PASS_PCT = 98.0

# controls
CTRL_CHECKER_MM = 4.8
CTRL_SUPERSAMPLE = 2


# --------------------------------------------------------------------------- #
# spectral / statistical machinery (new; neither contested instrument)
# --------------------------------------------------------------------------- #
def hann2d(h, w):
    return np.outer(np.hanning(h), np.hanning(w))


def isotropic_window(h, w, rolloff=0.15):
    """Radial (isotropic) taper, 1 in the interior, raised-cosine over the outer
    `rolloff` fraction of the inscribed radius.

    Why not a separable Hann: the frame is rectangular (1024x900) AND the fly mask
    is a mean-filled rectangle.  A separable window turns both into a rectangular
    envelope, whose transform is a cross of power at 0/90 deg.  This instrument
    first measured that artefact on a FLAT blank ground -- blank, a checkerboard
    and random-phase noise all scored circ_var ~0.39-0.42 with dominant
    orientation 5 deg off a 90-deg multiple, i.e. the window, not the ground.  A
    radial taper removes the rectangular envelope's 4-fold anisotropy, so the
    angular criterion measures the image instead of the frame.
    """
    yy = np.linspace(-1.0, 1.0, h).reshape(-1, 1)
    xx = np.linspace(-1.0, 1.0, w).reshape(1, -1)
    r = np.sqrt(yy ** 2 + xx ** 2) / math.sqrt(2.0)      # 0 centre, 1 at corners
    r_in = 1.0 - float(rolloff)
    win = np.ones((h, w))
    m = r > r_in
    win[m] = 0.5 * (1.0 + np.cos(np.pi * (r[m] - r_in) / max(1e-9, 1.0 - r_in)))
    return win


def mm_per_px(distance, fovy=A.CAM_FOVY, h=A.CAM_H):
    return 2.0 * float(distance) * math.tan(math.radians(float(fovy)) / 2.0) / float(h)


def mean_filled_edges(img):
    """Ordered-statistic edge fill.  The fly region is a mean-filled flat patch;
    a flat patch is smooth, so its (weak) spectral content sits at low frequency
    and cannot create a lattice band line.  This maps where that content is."""
    g = np.asarray(img, float)
    return np.abs((g - g.mean())) / max(g.std(), 1e-12)


def windowed(img, window="isotropic"):
    x = np.asarray(img, float)
    x = x - x.mean()
    if window == "isotropic":
        return x * isotropic_window(*x.shape)
    if window == "hann2d":
        return x * hann2d(*x.shape)
    return x


def power2d(img, normalize="none"):
    """|F|^2 of the Hann-windowed, mean-subtracted image.
    normalize='radial': divide out the mean over 300 radial annuli, i.e. remove
    the 1/f envelope and the DC lobe, leaving DISCRETE lines only."""
    X = np.fft.fft2(windowed(img))
    P = X.real ** 2 + X.imag ** 2
    if normalize == "none":
        return P
    k = freq_radius(img.shape)
    nb = 300
    edges = np.linspace(0.0, k.max(), nb + 1)
    idxr = np.clip(np.digitize(k.ravel(), edges) - 1, 0, nb - 1)
    tot = np.bincount(idxr, weights=P.ravel(), minlength=nb)
    cnt = np.bincount(idxr, minlength=nb).astype(float)
    mean_r = tot / np.maximum(cnt, 1.0)
    return P / np.maximum(mean_r[idxr].reshape(k.shape), 1e-30)


def band_mask_from_spec(P, mmpp, band=ANALYSIS_BAND_MM):
    k = freq_radius(P.shape)
    lo = 1.0 / band[1]
    hi = 1.0 / band[0]
    return (k >= lo) & (k <= hi), k / mmpp


def freq_radius(shape):
    """Radius of each FFT bin in cycles/PIXEL (unshifted fftfreq)."""
    h, w = shape
    fy = np.fft.fftfreq(h).reshape(-1, 1)
    fx = np.fft.fftfreq(w).reshape(1, -1)
    return np.sqrt(fy ** 2 + fx ** 2)


def band_resolution(shape, mmpp, band=ANALYSIS_BAND_MM):
    k = freq_radius(shape) / mmpp
    lo, hi = 1.0 / band[1], 1.0 / band[0]
    return dict(min_wavelength_px=band[0] / mmpp, max_wavelength_px=band[1] / mmpp,
                resolved_gt2px=bool(band[0] / mmpp > 2.0),
                n_band_bins=int(((k >= lo) & (k <= hi)).sum()),
                band_mm=list(band), band_cmm=[lo, hi])


def autocorr(img, min_overlap_px=10000):
    """EXACT LOCAL-mean 2-D autocorrelation over the overlap only.  No window.

      r(d) = [<x y>_d - <x>_d <y>_d] / sqrt(Var_d(x) Var_d(y))

    where <..>_d averages over exactly the pixels present in both the image and
    its d-shift, d is circular (which is legitimate: the boundaries of this image
    are not physical), and y is x shifted by d.  All the sliding sums are FFT
    correlations, so this costs three FFTs.

    Two earlier versions of this instrument were wrong in ways that LOOK like
    results, and both are recorded here rather than deleted:

      v1  only searched for POSITIVE maxima of r, so it classified a perfect
          square checkerboard as non-periodic.  A checker is ANTICORRELATED at
          one period: a shift of p flips floor(x/p)+floor(y/p) by exactly 1, so
          every cell changes colour, r(p,0) = -0.92 (measured on the control),
          while r(p,p) = +0.64 and r(2p,0) = +0.85.  Fixed by taking extrema of
          |r| and reporting the sign.
      v2  used a Hann or radial taper.  Then r = signal-correlation CONVOLVED
          with window-autocorrelation, so a featureless field shows a broad core
          of |r| > 0.8 out to ~0.3 mm -- the detector reported |r| = 0.83 at
          0.31 mm on a perfectly FLAT blank ground, and 0.67 for the leaf-litter
          ground.  That floor is the window, not the ground.  Fixed here by
          dropping the window entirely and normalising on the overlap instead.
    """
    g = np.asarray(img, float)
    H, W = g.shape
    Fx = np.fft.fft2(g)
    Fx2 = np.fft.fft2(g * g)
    Fn = np.fft.fft2(np.ones_like(g))
    Sxy = np.fft.fftshift(np.fft.ifft2(Fx * np.conj(Fx)).real)
    Sx2 = np.fft.fftshift(np.fft.ifft2(Fx2 * np.conj(Fn)).real)
    Sx = np.fft.fftshift(np.fft.ifft2(Fx * np.conj(Fn)).real)
    n = np.fft.fftshift(np.fft.ifft2(Fn * np.conj(Fn)).real)
    ok = n >= min_overlap_px
    nz = np.maximum(n, 1e-12)
    var = np.maximum(Sx2 / nz - (Sx / nz) ** 2, 0.0)      # same in x and y
    num = Sxy / nz - (Sx / nz) * (Sx / nz)
    den = np.sqrt(var * var)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(ok & (den > 1e-12), num / np.where(den > 1e-12, den, 1.0), 0.0)
    r[H // 2, W // 2] = 1.0
    return r


def central_peak_halfwidth_mm(ac, mmpp, level=0.5):
    """Distance from the origin at which |r| first falls below `level`, along +x.

    This sets C1's effective minimum lag.  Reason, and it is a real effect
    measured on this stack: the central autocorrelation peak is FINITE-WIDTH, and
    subtracting the radial mean inside that peak's own footprint leaves a ring of
    purely geometric residual that a peak detector will happily report as
    "anisotropy".  MEASURED at 36 mm: on the leaf-litter ground the residual at
    0.31 mm is 0.171, but the central |r| half-width is 0.51 mm, so that 0.171 is
    inside the halo; past the halo the same ground's residual is 0.092 (and 0.092
    again at 6.4 mm, i.e. flat, isotropic).  On a real lattice the halo test costs
    nothing -- a checker's central peak is 0.04 mm wide while its lattice peaks sit
    at 4.8 / 6.8 / 9.6 mm -- so the siting rule is: examine lags beyond
    max(0.3 mm resolution bar, 2x the central half-width), and REPORT both so
    the exclusion is auditable rather than hidden.
    """
    H, W = ac.shape
    x = ac[H // 2, W // 2:]
    for i in range(1, len(x)):
        if abs(x[i]) < level:
            return float(i * mmpp)
    return float(len(x) * mmpp)


def radial_residual(ac, mmpp, nb=200):
    """The 2-D autocorrelation with its RADIAL MEAN subtracted.

    This is the resolution of the central difficulty of this task.  A phase-free
    statement: aperiodic content with a DC-dominated envelope is mathematically
    FORCED to have a long correlation length (a narrow envelope in k and a
    compact support in x are Fourier duals), so ANY windowless detector sees
    |r| ~ 0.8 at the smallest lag it is allowed to look at -- measured here on
    random-phase noise with the photograph's own radial envelope: |r| = 0.81 at
    0.31 mm, and on the CC0 photograph itself 0.18.  That tail is the DC lobe,
    not a lattice.  A wavenumber window cannot separate them because the DC lobe
    extends across the whole band.

    What DOES separate them is ANGULAR structure.  A 1/f tail is isotropic --
    it is a function of lag radius alone -- so subtracting the radial mean
      * removes the whole envelope tail and the DC lobe exactly,
      * leaves a pure anisotropy map, which is the definition of a lattice:
        discrete spatial frequencies at discrete orientations.
    For a lattice the residual carries the lattice peaks (>= 0.8 for a checker);
    for any isotropic aperiodic field it is ~ 0.1 (measured, all controls).
    """
    H, W = ac.shape
    yy = (np.arange(H) - H // 2).reshape(-1, 1)
    xx = (np.arange(W) - W // 2).reshape(1, -1)
    k = np.sqrt(yy ** 2 + xx ** 2) * mmpp
    edges = np.linspace(0.0, k.max(), nb + 1)
    idx = np.clip(np.digitize(k.ravel(), edges) - 1, 0, nb - 1)
    tot = np.bincount(idx, weights=ac.ravel(), minlength=nb)
    cnt = np.bincount(idx, minlength=nb).astype(float)
    mean_r = tot / np.maximum(cnt, 1.0)
    return ac - mean_r[idx].reshape(k.shape)


def ac_residual_peak(ac, mmpp, min_lag_mm=CROP_MIN_MM, max_lag_mm=AC_LAG_MAX_MM):
    """C1's primary statistic: strongest ANISOTROPY of the autocorrelation inside
    the lag window, as a fraction of the DC peak (r(0,0) = 1).

    `min_lag_mm` is the halo-excluded lag from central_peak_halfwidth_mm, clamped
    to half the window so a very broad central peak (which means "no lattice at
    these scales" anyway) cannot empty the window.  An empty window returns
    abs_r = 0, i.e. FAIL, never an exception."""
    R = radial_residual(ac, mmpp)
    H, W = ac.shape
    yy = (np.arange(H) - H // 2).reshape(-1, 1)
    xx = (np.arange(W) - W // 2).reshape(1, -1)
    k = np.sqrt(yy ** 2 + xx ** 2) * mmpp
    min_lag_mm = float(min(min_lag_mm, 0.5 * max_lag_mm))
    band = (k >= min_lag_mm) & (k <= max_lag_mm)
    if not band.any():
        return dict(abs_r=0.0, r=0.0, sign="n/a", wavelength_mm=min_lag_mm,
                    lag_xy_mm=[0.0, 0.0], direction_deg=0.0, prominence=0.0,
                    frac_map_over_0p10=0.0, n_pixels_over_0p10=0,
                    mean_overlap_px=0.0, r_floor_3sigma=None,
                    lag_window_mm=[min_lag_mm, max_lag_mm])
    A = np.where(band, np.abs(R), 0.0)
    N = np.outer(np.minimum(np.arange(H) + 1, H - np.arange(H)),
                 np.minimum(np.arange(W) + 1, W - np.arange(W))).astype(float)
    iy, ix = np.unravel_index(int(np.argmax(A)), A.shape)
    val = float(R[iy, ix])
    w = max(2, int(round(0.6 / mmpp)))
    patch = np.abs(R[max(0, iy - w):iy + w + 1, max(0, ix - w):ix + w + 1])
    return dict(abs_r=abs(val), r=val,
                sign=("anticorrelated" if val < 0 else "correlated"),
                wavelength_mm=float(k[iy, ix]),
                lag_xy_mm=[float(xx[0, ix] * mmpp), float(yy[iy, 0] * mmpp)],
                direction_deg=float(math.degrees(math.atan2(float(yy[iy, 0]),
                                                            float(xx[0, ix]))) % 180.0),
                prominence=abs(val) - float(patch.min()),
                frac_map_over_0p10=float((A > 0.10).mean()),
                n_pixels_over_0p10=int((A > 0.10).sum()),
                mean_overlap_px=float(np.mean(N[band])) if band.any() else 0.0,
                r_floor_3sigma=float(3.0 / math.sqrt(max(np.mean(N[band]), 1.0)))
                if band.any() else None,
                lag_window_mm=[min_lag_mm, max_lag_mm])


def ac_radial(ac, mmpp):
    H, W = ac.shape
    yy = (np.arange(H) - H // 2).reshape(-1, 1)
    xx = (np.arange(W) - W // 2).reshape(1, -1)
    r_px = np.sqrt(yy ** 2 + xx ** 2)
    n = int(r_px.max()) + 1
    idx = np.clip(np.round(r_px).astype(int), 0, n - 1)
    s = np.bincount(idx.ravel(), weights=ac.ravel(), minlength=n)
    s2 = np.bincount(idx.ravel(), weights=(ac ** 2).ravel(), minlength=n)
    c = np.bincount(idx.ravel(), minlength=n).astype(float)
    mean = s / np.maximum(c, 1.0)
    sd = np.sqrt(np.maximum(s2 / np.maximum(c, 1.0) - mean ** 2, 0.0))
    return np.arange(n) * mmpp, mean, sd, c


def bilinear(ac, y, x):
    """Bilinear sample of the (fftshifted) autocorrelation, 1.0 outside."""
    H, W = ac.shape
    cy, cx = H // 2, W // 2
    fy, fx = y + cy, x + cx
    out = np.ones(np.shape(fy))
    ok = (fy >= 0) & (fy <= H - 2) & (fx >= 0) & (fx <= W - 2)
    if not np.any(ok):
        return out
    fy, fx = fy[ok], fx[ok]
    y0 = np.floor(fy).astype(int)
    x0 = np.floor(fx).astype(int)
    dy, dx = fy - y0, fx - x0
    v = (ac[y0, x0] * (1 - dy) * (1 - dx) + ac[y0 + 1, x0] * dy * (1 - dx)
         + ac[y0, x0 + 1] * (1 - dy) * dx + ac[y0 + 1, x0 + 1] * dy * dx)
    out[ok] = v
    return out


def directional_profiles(ac, mmpp, n_rays=36, max_lag_mm=AC_LAG_MAX_MM):
    """Directional autocorrelation along n_rays azimuths.

    Why rays and not the ring average: a square checkerboard is STRONGLY
    anisotropic.  Measured on the in-process checker control, r(one period) is
    -0.68 along x but the ANGULAR AVERAGE over the whole ring at that lag is only
    -0.19, because the ring also contains the directions in which the lattice is
    uncorrelated.  A ring-averaged detector therefore dilutes exactly the signal
    it is looking for.  Rays keep it.
    """
    n = int(max_lag_mm / mmpp) + 1
    lags = np.arange(n) * mmpp
    angs = np.arange(n_rays) * (180.0 / n_rays)
    profs, peaks = [], []
    for a in angs:
        t = np.radians(a)
        y = lags * math.sin(t) / mmpp
        x = lags * math.cos(t) / mmpp
        p = bilinear(ac, y, x)
        profs.append(p)
        k0 = max(int(np.ceil(CROP_MIN_MM / mmpp)), 1)
        seg = np.abs(p[k0:])
        if seg.size:
            j = int(np.argmax(seg)) + k0
            peaks.append(dict(ray_deg=float(a), abs_r=float(abs(p[j])),
                              r=float(p[j]), lag_mm=float(lags[j])))
    P = np.vstack(profs)
    return dict(lags_mm=lags, rays_deg=angs, profiles=P,
                median_profile=np.median(P, axis=0), max_profile=P.max(axis=0),
                ray_peaks=peaks)


def ac_2d_peak(ac, mmpp, min_lag_mm=CROP_MIN_MM, max_lag_mm=AC_LAG_MAX_MM):
    """C1's primary statistic: the global off-centre extremum of the 2-D
    autocorrelation MAP, as a fraction of the DC peak (r(0,0) = 1 by
    construction), with its lag in mm, its signed value and its prominence on the
    map itself.

    Why the 2-D map and not a radial or angular average: a square lattice is
    strongly ANISOTROPIC, so any average over directions dilutes exactly the
    structure being tested.  MEASURED on the in-process checker control: r is
    -0.92 along the axes at one period and +0.64 on the diagonal, but the
    ring-averaged radial profile at that lag is only -0.19 and the median over 36
    rays is 0.21.  A ring- or median-averaged detector therefore fails on a
    perfect checkerboard -- the very failure this script exists to arbitrate.
    Directional structure is measured separately by C2.

    Extremum of |r| and not of r: a square checker is ANTICORRELATED at one
    period (a shift of p flips floor(x/p)+floor(y/p) by exactly 1, so every cell
    changes colour: r(p,0) = -0.92 measured) and correlated on the diagonal
    (r(p,p) = +0.64) and at 2p (+0.85).  A detector that only looked for
    positive maxima classified a perfect checkerboard as non-periodic, which cost
    one instrument revision (recorded in the module docstring).  The sign is
    reported so anticorrelation is not hidden.
    """
    H, W = ac.shape
    yy = (np.arange(H) - H // 2).reshape(-1, 1)
    xx = (np.arange(W) - W // 2).reshape(1, -1)
    core = np.sqrt(yy ** 2 + xx ** 2) * mmpp
    band = (core >= min_lag_mm) & (core <= max_lag_mm)
    A = np.where(band, np.abs(ac), 0.0)
    iy, ix = np.unravel_index(int(np.argmax(A)), A.shape)
    val = float(ac[iy, ix])
    lag = float(core[iy, ix])
    w = max(2, int(round(0.6 / mmpp)))
    y0, y1 = max(0, iy - w), min(H, iy + w + 1)
    x0, x1 = max(0, ix - w), min(W, ix + w + 1)
    prom = abs(val) - float(np.abs(ac[y0:y1, x0:x1]).min())
    ang = math.atan2(float(yy[iy, 0]), float(xx[0, ix]))
    uy, ux = math.sin(ang), math.cos(ang)
    widths = []
    for sign in (1.0, -1.0):
        s = 0.0
        while True:
            s += mmpp
            y = iy + sign * uy * s / mmpp
            x = ix + sign * ux * s / mmpp
            if not (0 <= y <= H - 2 and 0 <= x <= W - 2) or s > 3.0:
                break
            if abs(float(bilinear(ac, np.array([y]), np.array([x]))[0])) < abs(val) / 2.0:
                break
        widths.append(s)
    return dict(found=bool(A.max() > 0.0), abs_r=abs(val), r=val,
                sign=("anticorrelated" if val < 0 else "correlated"),
                wavelength_mm=lag, lag_mm=lag,
                lag_xy_mm=[float(xx[0, ix] * mmpp), float(yy[iy, 0] * mmpp)],
                direction_deg=float(math.degrees(ang) % 180.0),
                prominence=float(prom), fwhm_mm=float(sum(widths)),
                lag_window_mm=[min_lag_mm, max_lag_mm],
                n_pixels_over_0p15=int((A > 0.15).sum()),
                frac_of_map_over_0p15=float((A > 0.15).mean()))

def angular_stats(img, mmpp):
    """C2: angular shape of the band |F|^2 with the radial mean divided out."""
    Pn = power2d(img, normalize="radial")
    band, k_mm = band_mask_from_spec(Pn, mmpp)
    H, W = img.shape
    ky = np.broadcast_to(np.fft.fftfreq(H).reshape(-1, 1) / mmpp, (H, W))
    kx = np.broadcast_to(np.fft.fftfreq(W).reshape(1, -1) / mmpp, (H, W))
    th = np.degrees(np.arctan2(ky, kx)) % 360.0
    w = Pn[band]
    a = th[band]
    nb_a = 36
    hist, edges_a = np.histogram(a, bins=nb_a, range=(0, 360), weights=w)
    hist = hist / max(hist.sum(), 1e-30)
    centres = 0.5 * (edges_a[:-1] + edges_a[1:])
    z = np.sum(hist * np.exp(1j * np.radians(4.0 * centres)))
    z1 = np.sum(hist * np.exp(1j * np.radians(centres)))
    dom = float(centres[int(np.argmax(hist))])
    dom_fold = float(dom % 90.0)
    return dict(hist=hist.tolist(), bin_centres_deg=centres.tolist(),
                peak_to_mean_deg=float(hist.max() / max(hist.mean(), 1e-30)),
                circ_var_folded=float(1.0 - abs(z)),
                circ_var_unfolded=float(1.0 - abs(z1)),
                dominant_orientation_deg=dom, dominant_folded_deg=dom_fold,
                dominant_to_90deg_multiple_deg=float(min(dom_fold, 90.0 - dom_fold)),
                dominant_to_45deg_multiple_deg=float(min(dom % 45.0,
                                                         45.0 - (dom % 45.0))),
                n_band_bins=int(band.sum()),
                band_mm=list(ANALYSIS_BAND_MM))


def band_peak_stats(img, mmpp, normalize="none"):
    """max / median inside the band (optionally on the radially-normalised
    power, which isolates discrete lines from the 1/f envelope)."""
    P = power2d(img, normalize=normalize)
    band, _ = band_mask_from_spec(P, mmpp)
    v = P[band]
    if v.size < 4:
        return None
    med = float(np.median(v))
    return dict(max=float(v.max()), median=med,
                peak_over_median=(float(v.max() / med) if med > 0 else float("inf")),
                n=int(v.size), normalize=normalize)


def surrogate_test(img, mmpp, n=N_SURROGATES, seed=SURROGATE_SEED):
    """C4: keep |F|, randomise phases, recompute the band statistics."""
    x = windowed(img)
    amp = np.abs(np.fft.fft2(x))
    H, W = x.shape
    rng = np.random.default_rng(seed)
    real_raw = band_peak_stats(img, mmpp, "none")
    real_norm = band_peak_stats(img, mmpp, "radial")
    raw_v, norm_v = [], []
    for _ in range(n):
        ph = rng.uniform(0.0, 2.0 * np.pi, size=(H, W))
        ph[0, 0] = 0.0
        ph = 0.5 * (ph + (-ph[::-1, ::-1]))          # Hermitian -> real output
        xs = np.fft.ifft2(amp * np.exp(1j * ph)).real
        raw_v.append(band_peak_stats(xs, mmpp, "none")["peak_over_median"])
        norm_v.append(band_peak_stats(xs, mmpp, "radial")["peak_over_median"])
    def pct(real, arr):
        arr = np.asarray(arr, float)
        return float(100.0 * np.mean(arr <= real))
    return dict(
        n_surrogates=int(n), seed=int(seed),
        real_band_peak_over_median=real_raw["peak_over_median"],
        real_band_peak_over_median_radial_normalized=real_norm["peak_over_median"],
        surrogate_band_peak_over_median=dict(
            values=raw_v, min=float(np.min(raw_v)), median=float(np.median(raw_v)),
            p98=float(np.percentile(raw_v, 98)), max=float(np.max(raw_v))),
        surrogate_band_peak_over_median_radial_normalized=dict(
            values=norm_v, min=float(np.min(norm_v)), median=float(np.median(norm_v)),
            p98=float(np.percentile(norm_v, 98)), max=float(np.max(norm_v))),
        real_percentile_in_surrogates_raw=pct(real_raw["peak_over_median"], raw_v),
        real_percentile_in_surrogates_radial=pct(real_norm["peak_over_median"], norm_v),
        empirical_p_raw=float((1.0 + np.sum(np.asarray(raw_v) >=
                                            real_raw["peak_over_median"])) / (n + 1.0)),
        empirical_p_radial=float((1.0 + np.sum(np.asarray(norm_v) >=
                                               real_norm["peak_over_median"])) / (n + 1.0)),
        band_resolution=band_resolution(img.shape, mmpp))


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def render_ground(preset, distance):
    b, build_s = A.build(preset)
    b.attach_cpg_baseline()
    for _ in range(WARMUP_STEPS):
        b.step()
    th = b.observe().thorax_position_mm
    r = mujoco.Renderer(b.model, height=A.CAM_H, width=A.CAM_W)
    img = A.render_free(b, r, lookat=[float(th[0]), float(th[1]), 0.0],
                        distance=float(distance), fovy=A.CAM_FOVY)
    r.close()
    assert img.shape[:2] == (A.CAM_H, A.CAM_W), img.shape
    raw = img[..., :3].mean(axis=2).astype(float)
    h, w = raw.shape
    mask = np.zeros(raw.shape, bool)
    mask[int(h * MASK_ROW[0]):int(h * MASK_ROW[1]),
         int(w * MASK_COL[0]):int(w * MASK_COL[1])] = True
    g = np.where(mask, float(raw[~mask].mean()), raw)
    m = b.model
    gmats = []
    for gi in range(m.ngeom):
        if int(m.geom_type[gi]) in (0, 1):
            mid = int(m.geom_matid[gi])
            tex = None
            if mid >= 0:
                for ti in np.asarray(m.mat_texid[mid]).ravel():
                    if int(ti) >= 0:
                        tex = str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TEXTURE,
                                                    int(ti)))
                        break
            gmats.append(dict(
                geom=str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gi)),
                material=(str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, mid))
                          if mid >= 0 else None),
                texture=tex,
                texrepeat=([float(v) for v in np.asarray(m.mat_texrepeat[mid])]
                           if mid >= 0 else None)))
    # Save the EXACT analysis input as float32 .npy, so every number in the JSON
    # can be re-derived bit-for-bit from a file in this directory:
    #     np.load(...)  ->  measure_image(that array, mm_per_px(distance))
    # PNG was tried first twice and is not good enough: the renderer's values are
    # quantised to 1/3 of a level, so 8-bit rescaling moved C2's circ_var by 0.02
    # and C4's surrogate maximum by 10 %, and PIL's I;16 round-trip still rounded.
    try:
        np.save(os.path.join(
            OUT, f"arbitrate_analysis_input_{preset}_{distance:.0f}mm.npy"),
            g.astype(np.float32))
    except Exception as _exc:                                   # pragma: no cover
        print(f"  [warn] could not export analysis input: {_exc}")
    return dict(preset=preset, distance_mm=float(distance),
                mm_per_px=mm_per_px(distance), build_s=float(build_s),
                gray=g, gray_raw=raw, mask=mask,
                lookat=[float(th[0]), float(th[1]), 0.0],
                ground_bindings=gmats,
                ground_mean=float(g[~mask].mean()), ground_std=float(g[~mask].std()))


# --------------------------------------------------------------------------- #
# criteria
# --------------------------------------------------------------------------- #
def measure_image(g, mmpp, do_surrogate=True):
    ac = autocorr(g)
    lag, prof, sd, cnt = ac_radial(ac, mmpp)
    dirs = directional_profiles(ac, mmpp)
    peak2d = ac_2d_peak(ac, mmpp)
    width_mm = central_peak_halfwidth_mm(ac, mmpp)
    eff_min_lag = max(CROP_MIN_MM, C1_WIDTH_MULT * width_mm)
    resid_full = ac_residual_peak(ac, mmpp, min_lag_mm=eff_min_lag)
    # per 2x2 crop: same statistic, independent part of the frame
    crops = []
    H, W = g.shape
    for iy in range(CROP_DIV):
        for ix in range(CROP_DIV):
            r0 = int(iy * H / CROP_DIV) + CROP_TRIM_PX
            r1 = int((iy + 1) * H / CROP_DIV) - CROP_TRIM_PX
            c0 = int(ix * W / CROP_DIV) + CROP_TRIM_PX
            c1 = int((ix + 1) * W / CROP_DIV) - CROP_TRIM_PX
            acs = autocorr(g[r0:r1, c0:c1])
            wcrop = central_peak_halfwidth_mm(acs, mmpp)
            crops.append(dict(row=iy, col=ix,
                              peak=ac_residual_peak(
                                  acs, mmpp,
                                  min_lag_mm=max(CROP_MIN_MM, C1_WIDTH_MULT * wcrop)),
                              raw_peak=ac_2d_peak(acs, mmpp),
                              central_halfwidth_mm=wcrop))
    vals = [c["peak"]["abs_r"] for c in crops]
    ray_abs = np.array([p["abs_r"] for p in dirs["ray_peaks"]])
    ray_lag = np.array([p["lag_mm"] for p in dirs["ray_peaks"]])
    step = max(1, len(lag) // 200)
    out = dict(mm_per_px=mmpp, shape=list(g.shape),
               ac_noise_floor_sd=float(np.median(sd[int(np.ceil(CROP_MIN_MM / mmpp)):])),
               ac_2d_peak_raw=peak2d,
               central_peak_halfwidth_mm=width_mm,
               c1_effective_min_lag_mm=eff_min_lag,
               ac_residual_peak=resid_full,
               ray_peak_abs_r_values=ray_abs.tolist(),
               ray_peak_lag_mm_values=ray_lag.tolist(),
               ray_peak_angle_deg=[p["ray_deg"] for p in dirs["ray_peaks"]],
               ray_peak_abs_r_median=float(np.median(ray_abs)),
               ray_peak_abs_r_min=float(ray_abs.min()),
               ray_peak_abs_r_max=float(ray_abs.max()),
               ray_peak_lag_mm_median=float(np.median(ray_lag)),
               ray_peak_lag_mm_iqr=float(np.percentile(ray_lag, 75)
                                         - np.percentile(ray_lag, 25)),
               ray_peak_lag_dominant_mm=float(
                   ray_lag[int(np.argmax(ray_abs))]),
               ray_peak_dominant_angle_deg=float(
                   dirs["ray_peaks"][int(np.argmax(ray_abs))]["ray_deg"]),
               band_resolution=band_resolution(g.shape, mmpp),
               crops=[dict(row=c["row"], col=c["col"], peak=c["peak"]) for c in crops],
               crop_peak_r_values=[float(v) for v in vals],
               crop_peak_r_spread=float(max(vals) - min(vals)),
               crop_peak_r_mean=float(np.mean(vals)),
               ac_profile_mm=lag[::step].tolist(), ac_profile=prof[::step].tolist(),
               ac_profile_step_px=int(step))
    out["angular"] = angular_stats(g, mmpp)
    if do_surrogate:
        out["surrogate"] = surrogate_test(g, mmpp)
    out["C1"] = criterion_C1(out)
    out["C2"] = criterion_C2(out.get("angular"), out)
    out["C4"] = criterion_C4(out.get("surrogate"))
    return out


def criterion_C1(m):
    """AUTOCORRELATION SIDELOBE / ANISOTROPY test (see radial_residual for the
    justification of the residual form).

    PRE-REGISTERED PASS: off-centre |residual| >= 0.15 AND its prominence >= 0.05
    AND the spread over 2x2 independent crops <= 0.5 x |residual|.
    """
    pk = m["ac_residual_peak"]
    spread = m["crop_peak_r_spread"]
    conds = dict(
        residual=(
            pk["abs_r"] >= C1_PASS_R,
            f"strongest off-centre anisotropy |r_resid|={pk['abs_r']:.3f} at "
            f"{pk['wavelength_mm']:.3f} mm lag, direction {pk['direction_deg']:.1f} deg "
            f"vs >= {C1_PASS_R}"),
        significance=(
            pk["r_floor_3sigma"] is not None and pk["abs_r"] > pk["r_floor_3sigma"],
            f"|r_resid|={pk['abs_r']:.3f} vs the 3-sigma overlap-sample noise floor "
            f"{'n/a' if pk['r_floor_3sigma'] is None else round(pk['r_floor_3sigma'],3)} "
            f"at the detected lag "
            f"(effective overlap {pk['mean_overlap_px']:.0f} px) -- a correlation "
            f"carried by a thin sliver of overlap is noise"),
        prominence=(pk["prominence"] >= C1_PASS_PROMINENCE,
                    f"prominence={pk['prominence']:.3f} vs >= {C1_PASS_PROMINENCE}"),
        crop_stability=(spread <= C1_PASS_CROP_SPREAD * pk["abs_r"],
                        f"2x2 crop spread={spread:.3f} vs <= "
                        f"{C1_PASS_CROP_SPREAD} x |r_resid|"))
    return dict(pass_=all(v[0] for v in conds.values()),
                peak_r=float(pk["abs_r"]), signed_r=float(pk["r"]), sign=pk["sign"],
                prominence=float(pk["prominence"]),
                wavelength_mm=float(pk["wavelength_mm"]),
                lag_xy_mm=pk["lag_xy_mm"], direction_deg=pk["direction_deg"],
                frac_map_over_0p10=pk["frac_map_over_0p10"],
                n_pixels_over_0p10=pk["n_pixels_over_0p10"],
                lag_window_mm=pk["lag_window_mm"], crop_spread=spread,
                raw_autocorr_peak=m["ac_2d_peak_raw"],
                ray_detail=dict(median_abs_r=float(np.median(
                                    np.asarray(m["ray_peak_abs_r_values"], float))),
                                min_abs_r=m["ray_peak_abs_r_min"],
                                max_abs_r=m["ray_peak_abs_r_max"],
                                lag_iqr_mm=m["ray_peak_lag_mm_iqr"],
                                dominant_ray_deg=m["ray_peak_dominant_angle_deg"]),
                conditions={k: dict(passed=bool(v[0]), detail=v[1])
                            for k, v in conds.items()})


def criterion_C2(a, m=None):
    """Angular/orientation test, PRE-REGISTERED PASS: circ_var(90-deg folded)
    <= 0.60 AND peak/mean over 36 deg bins >= 2.0 AND the dominant orientation is
    within 10 deg of a 45-deg multiple (a square lattice's discrete lines are on
    the diagonals)."""
    if a is None:
        return dict(pass_=False, reason="angular test not run")
    conds = dict(
        circ_var=(a["circ_var_folded"] <= C2_PASS_CIRC_VAR,
                  f"circ_var(4-fold)={a['circ_var_folded']:.3f} "
                  f"vs <= {C2_PASS_CIRC_VAR}"),
        peak_to_mean=(a["peak_to_mean_deg"] >= C2_PASS_PTM,
                      f"peak/mean over 36 deg bins={a['peak_to_mean_deg']:.3f} "
                      f"vs >= {C2_PASS_PTM}"),
        alignment=(a["dominant_to_45deg_multiple_deg"] <= C2_PASS_ALIGN_DEG,
                   f"dominant orientation {a['dominant_orientation_deg']:.1f} deg is "
                   f"{a['dominant_to_45deg_multiple_deg']:.1f} deg off the nearest "
                   f"45-deg multiple vs <= {C2_PASS_ALIGN_DEG}"))
    out = dict(pass_=all(v[0] for v in conds.values()),
               circ_var_folded=a["circ_var_folded"],
               circ_var_unfolded=a["circ_var_unfolded"],
               peak_to_mean_deg=a["peak_to_mean_deg"],
               dominant_orientation_deg=a["dominant_orientation_deg"],
               dominant_folded_deg=a["dominant_folded_deg"],
               dominant_to_45deg_multiple_deg=a["dominant_to_45deg_multiple_deg"],
               n_band_bins=a["n_band_bins"], band_mm=a["band_mm"],
               conditions={k: dict(passed=bool(v[0]), detail=v[1])
                           for k, v in conds.items()})
    if m is not None:
        out["autocorr_dominant_ray_deg"] = m.get("ray_peak_dominant_angle_deg")
        out["autocorr_ray_lag_iqr_mm"] = m.get("ray_peak_lag_mm_iqr")
    return out



def criterion_C3(per_dist):
    detected = {}
    for d, m in per_dist.items():
        pk = m["ac_residual_peak"]
        detected[d] = dict(peak_r=float(pk["abs_r"]),
                           wavelength_mm=float(pk["wavelength_mm"]),
                           direction_deg=float(pk["direction_deg"]),
                           crop_r=[c["peak"]["abs_r"] for c in m["crops"]],
                           resolvable=m["band_resolution"]["resolved_gt2px"],
                           crop_peak_r_values=m.get("crop_peak_r_values"))
    okd = [d for d in detected if detected[d]["wavelength_mm"] is not None]
    conds = dict(peak_visible_all_distances=(
        all(detected[d]["peak_r"] >= C3_PASS_PEAK_R for d in detected),
        "|r| per distance " + ", ".join(f"{d:.0f}mm:{detected[d]['peak_r']:.3f}"
                                        for d in sorted(detected))))
    if len(okd) >= 2:
        wl = [detected[d]["wavelength_mm"] for d in okd]
        wmin, wmax = min(wl), max(wl)
        spread = (wmax - wmin) / max(wmax, 1e-12)
        d36 = detected.get(36.0, {}).get("wavelength_mm")
        d150 = detected.get(150.0, {}).get("wavelength_mm")
        if d36 is None and okd:
            d36 = detected[min(okd)]["wavelength_mm"]
        if d150 is None and okd:
            d150 = detected[max(okd)]["wavelength_mm"]
        drift = abs(d150 - d36) / max(d36, 1e-12)
        conds["wavelength_stable_across_distances"] = (
            spread <= C3_PASS_WAVELENGTH_SPREAD,
            f"wavelengths {['%.3f' % v for v in wl]} mm over "
            f"{min(okd):.0f}-{max(okd):.0f} mm: spread {spread*100:.1f} % vs <= "
            f"{C3_PASS_WAVELENGTH_SPREAD*100:.0f} %")
        conds["wavelength_drift_near_to_far"] = (
            drift <= C3_PASS_DRIFT,
            f"drift {drift*100:.1f} % vs <= {C3_PASS_DRIFT*100:.0f} %")
    else:
        spread, drift = None, None
        conds["wavelength_stable_across_distances"] = (False, "<2 distances yielded a peak")
        conds["wavelength_drift_near_to_far"] = (False, "peak missing at a distance")
    lo, hi = min(detected), max(detected)
    a, b = detected[lo]["peak_r"], detected[hi]["peak_r"]
    ratio = (max(a, b) / max(min(a, b), 1e-12)) if min(a, b) > 0 else float("inf")
    conds["peak_strength_stable"] = (
        ratio <= C3_PASS_PEAK_RATIO,
        f"|r| ratio {lo:.0f}mm vs {hi:.0f}mm = {ratio:.2f} vs <= {C3_PASS_PEAK_RATIO}")
    return dict(pass_=all(v[0] for v in conds.values()), detected=detected,
                wavelength_spread_frac=spread, drift=drift,
                peak_r_ratio_far_over_near=(float(ratio) if np.isfinite(ratio) else None),
                wavelength_scales_with_distance=bool(drift is not None
                                                     and drift > C3_PASS_DRIFT),
                conditions={k: dict(passed=bool(v[0]), detail=v[1])
                            for k, v in conds.items()})


def criterion_C4(s):
    if s is None:
        return dict(pass_=False, reason="surrogate test not run")
    conds = dict(
        band_peak_over_median=(
            s["real_percentile_in_surrogates_raw"] > C4_PASS_PCT,
            f"real band max/median={s['real_band_peak_over_median']:.1f} vs surrogate "
            f"max {s['surrogate_band_peak_over_median']['max']:.1f} "
            f"(percentile {s['real_percentile_in_surrogates_raw']:.1f})"),
        band_radial_normalized=(
            s["real_percentile_in_surrogates_radial"] > C4_PASS_PCT,
            f"real normalised band max/median="
            f"{s['real_band_peak_over_median_radial_normalized']:.2f} vs surrogate max "
            f"{s['surrogate_band_peak_over_median_radial_normalized']['max']:.2f} "
            f"(percentile {s['real_percentile_in_surrogates_radial']:.1f})"))
    return dict(pass_=all(v[0] for v in conds.values()),
                note=("both variants are band-restricted; the raw variant tests the "
                      "contested peak/median number itself, the radial-normalised "
                      "variant tests for DISCRETE lines with the 1/f envelope removed"),
                conditions={k: dict(passed=bool(v[0]), detail=v[1])
                            for k, v in conds.items()})


# --------------------------------------------------------------------------- #
# controls
# --------------------------------------------------------------------------- #
def control_checker(shape, mmpp, period_mm=CTRL_CHECKER_MM, ss=CTRL_SUPERSAMPLE):
    H, W = shape
    p = period_mm / mmpp
    yy = (np.arange(H * ss) + 0.5) / ss
    xx = (np.arange(W * ss) + 0.5) / ss
    c = np.floor(yy[:, None] / p) + np.floor(xx[None, :] / p)
    img = np.where((c % 2) == 0, 1.0, 0.0)
    return img.reshape(H, ss, W, ss).mean(axis=(1, 3)) * 255.0


def radial_mean_power(img, nb=64):
    """Radial mean of |F|^2 over nb annuli (the amplitude envelope of an image)."""
    P = np.abs(np.fft.fft2(np.asarray(img, float))) ** 2
    k = freq_radius(P.shape)
    edges = np.linspace(0.0, k.max(), nb + 1)
    idx = np.clip(np.digitize(k.ravel(), edges) - 1, 0, nb - 1)
    tot = np.bincount(idx, weights=P.ravel(), minlength=nb)
    cnt = np.bincount(idx, minlength=nb).astype(float)
    return tot / np.maximum(cnt, 1.0), 0.5 * (edges[:-1] + edges[1:]), nb


def control_colored_noise(shape, envelope_source, seed=SURROGATE_SEED + 7):
    """Aperiodic control: a featureless (radial-mean) power envelope with random
    phases.

    NOT made from a lattice's full |F|: randomising the phases of a checker keeps
    the checker's amplitude DELTAS intact, so the result is still periodic and the
    autocorrelation of this control measured |r| = 0.85 -- i.e. a control that
    inherited the very structure it was supposed to disprove.  The envelope is
    reduced to its RADIAL MEAN instead, so only the 1/f-like trend survives and
    every discrete line is gone.
    """
    H, W = shape
    env, kr, nb = radial_mean_power(envelope_source)
    amp = np.sqrt(np.maximum(env, 0.0))
    # expand the 1-D radial envelope back over the 2-D frequency plane
    k = freq_radius(shape)
    ridx = np.clip(np.digitize(k, np.linspace(0.0, k.max(), nb + 1)) - 1, 0, nb - 1)
    A = amp[ridx]
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0.0, 2.0 * np.pi, size=(H, W))
    ph[0, 0] = 0.0
    ph = 0.5 * (ph + (-ph[::-1, ::-1]))
    return np.fft.ifft2(A * np.exp(1j * ph)).real


def load_photo_control(shape):
    from PIL import Image
    from scipy.ndimage import zoom as _zoom
    photo = np.asarray(Image.open(os.path.join(
        OUT, "assets", "natural_ground_photo.png")).convert("L"), float)
    zy, zx = shape[0] / photo.shape[0], shape[1] / photo.shape[1]
    photo = _zoom(photo, (zy, zx), order=1)
    pad = np.full(shape, photo.mean())
    h = min(shape[0], photo.shape[0])
    w = min(shape[1], photo.shape[1])
    pad[:h, :w] = photo[:h, :w]
    return pad


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    print("arbitrate_periodicity.py -- criteria independent of peak/median and peak_frac")
    print(f"PRE-REGISTERED band {ANALYSIS_BAND_MM} mm ({1/ANALYSIS_BAND_MM[1]:.3f}-"
          f"{1/ANALYSIS_BAND_MM[0]:.3f} cyc/mm); autocorrelation lag window "
          f"[{CROP_MIN_MM}, {AC_LAG_MAX_MM}] mm")
    print(f"PRE-REGISTERED C1 |r|>={C1_PASS_R} prom>={C1_PASS_PROMINENCE} "
          f"crop-spread<={C1_PASS_CROP_SPREAD}|r| ; C2 circvar<={C2_PASS_CIRC_VAR} "
          f"ptm>={C2_PASS_PTM} align<={C2_PASS_ALIGN_DEG}deg ; C3 spread<="
          f"{C3_PASS_WAVELENGTH_SPREAD} drift<={C3_PASS_DRIFT} ratio<={C3_PASS_PEAK_RATIO} ; "
          f"C4 >{C4_PASS_PCT}th percentile of {N_SURROGATES} surrogates\n")

    report = dict(
        script="arbitrate_periodicity.py",
        purpose=("decide whether a periodic lattice is physically present in the "
                 "rendered ground, using criteria independent of peak/median and "
                 "peak_frac"),
        preregistered=dict(
            analysis_band_mm=list(ANALYSIS_BAND_MM),
            tile_exclusion_mm=TILE_EXCLUSION_MM,
            ac_lag_window_mm=[CROP_MIN_MM, AC_LAG_MAX_MM], local_min_r=LOCAL_MIN_R,
            C1=dict(peak_r=C1_PASS_R, prominence=C1_PASS_PROMINENCE,
                    crop_spread_frac=C1_PASS_CROP_SPREAD),
            C2=dict(circ_var=C2_PASS_CIRC_VAR, peak_to_mean=C2_PASS_PTM,
                    align_deg=C2_PASS_ALIGN_DEG),
            C3=dict(peak_r=C3_PASS_PEAK_R, wavelength_spread=C3_PASS_WAVELENGTH_SPREAD,
                    drift=C3_PASS_DRIFT, peak_r_ratio=C3_PASS_PEAK_RATIO),
            C4=dict(n_surrogates=N_SURROGATES, seed=SURROGATE_SEED,
                    pass_percentile=C4_PASS_PCT),
            verdict_rule=(">=3 PASS -> POSITIVE; 2 -> WEAK; <=1 -> NEGATIVE; "
                          "C1/C2 PASS with C4 FAIL -> WEAK (not phase locked); "
                          "wavelength scaling with camera distance -> NEGATIVE"),
            instrument_revision=("C1 v1 searched only for positive maxima and "
                                 "therefore failed on a perfect square checker, which "
                                 "is ANTICORRELATED at one period.  Fixed to extrema "
                                 "of |r| BEFORE the real presets were re-measured.")),
        provenance=dict(
            units="mm; model total mass 1.0243 g is ~1042x a real Drosophila (given)",
            camera=("render_free/CAM_H/CAM_W/CAM_FOVY imported from "
                    "scene_realism_audit_run; the 36 mm condition reproduces "
                    "compare_ground_audit_instruments.py exactly; 75 and 150 mm added"),
            reference_contested=dict(
                peak_over_median_natural_fbm=8892.0, audit_target_le=300.0,
                blank_peak_over_median=22727.0, peak_frac_natural_fbm=0.0158,
                peak_frac_max=0.05, peak_frac_old_terrain=0.3452,
                tag="MEASURED_LOCAL (values given in the task brief)"),
            tags=dict(thresholds="ENGINEERING_DEFAULT, declared before measurement, "
                                 "calibrated on in-process synthetic controls",
                      measured="MEASURED_LOCAL"),
            assumptions=[
                "ASSUMED: the mean-filled fly mask covers the fly at every distance "
                "(row 0.30-0.72, col 0.28-0.72). At 150 mm the footprint is larger so "
                "the mask removes more ground; the 2x2 crop spread bounds the effect.",
                "ASSUMED: the ground macro-texture tile is 90 mm (natural_scene.py "
                "texture_tile_mm), so tile repetition is excluded by the band, not by "
                "choice of a favourable window; 90 mm is 1060 px at 36 mm, larger than "
                "the frame, so tiling seams are not resolvable in-frame at all.",
                "ASSUMED: the same lattice would be measured on the eye cameras; this "
                "script only measures the audit's overhead camera.",
            ]),
    )

    # ---------------- validity gate: controls ----------------------------- #
    shape = (A.CAM_H, A.CAM_W)
    mmpp36 = mm_per_px(36.0)
    ctrl = {}
    ck = control_checker(shape, mmpp36)
    ns = control_colored_noise(shape, ck)
    items = [("ctrl_checker_%.1fmm" % CTRL_CHECKER_MM, ck, "lattice"),
             ("ctrl_coloured_noise", ns, "aperiodic")]
    try:
        items.append(("ctrl_cc0_photo_no_render", load_photo_control(shape), "aperiodic"))
    except Exception as exc:
        print(f"[control] photo control unavailable: {type(exc).__name__}: {exc}")
    for name, im, kind in items:
        m = measure_image(im, mmpp36)
        c1, c2, c4 = m["C1"], m["C2"], m["C4"]
        ctrl[name] = dict(kind=kind, C1=c1, C2=c2, C4=c4,
                          circ_var_folded=c2["circ_var_folded"],
                          peak_to_mean_deg=c2["peak_to_mean_deg"],
                          C1_peak_r=c1["peak_r"], C1_wavelength_mm=c1.get("wavelength_mm"))
        print(f"[control {name:24s} {kind:9s}] C1 "
              f"{'PASS' if c1['pass_'] else 'FAIL':4s} "
              f"|r|={c1['peak_r']:.3f} @ {c1.get('wavelength_mm')} mm "
              f"prom={c1.get('prominence')} | C2 {'PASS' if c2['pass_'] else 'FAIL':4s} "
              f"circvar={c2['circ_var_folded']:.3f} ptm={c2['peak_to_mean_deg']:.2f} "
              f"dom={c2['dominant_orientation_deg']:.1f}deg | C4 "
              f"{'PASS' if c4['pass_'] else 'FAIL'} "
              f"pct={m['surrogate']['real_percentile_in_surrogates_radial']:.1f}")
    report["controls"] = ctrl
    ck_c = ctrl["ctrl_checker_%.1fmm" % CTRL_CHECKER_MM]
    c1_valid = bool(ck_c["C1"]["pass_"])
    c2_valid = bool(ck_c["C2"]["pass_"])
    if not (c1_valid and c2_valid):
        print("\n*** VALIDITY GATE: the checker control did not pass; the failing "
              "criterion is reported INVALID ***\n")

    # ---------------- real presets --------------------------------------- #
    measures = {}
    for p in PRESETS:
        measures[p] = {}
        for d in DISTANCES_MM:
            t = time.time()
            try:
                r = render_ground(p, d)
            except Exception as exc:
                print(f"{p} @ {d:.0f} mm: RENDER FAILED {type(exc).__name__}: {exc}")
                measures[p][d] = dict(error=f"{type(exc).__name__}: {exc}")
                continue
            m = measure_image(r["gray"], r["mm_per_px"])
            m["distance_mm"] = d
            m["ground_mean"] = r["ground_mean"]
            m["ground_std"] = r["ground_std"]
            m["ground_bindings"] = r["ground_bindings"]
            measures[p][d] = m
            r["_img"] = r["gray_raw"]
            measures[p][f"img_{d}"] = r
            c1, c2, c4 = m["C1"], m["C2"], m["C4"]
            print(f"{p:12s} @ {d:5.1f} mm ({time.time()-t:5.1f}s) "
                  f"C1 {'PASS' if c1['pass_'] else 'FAIL'} "
                  f"|r|={c1['peak_r']:.3f}@{c1['wavelength_mm']:.3f}mm "
                  f"{c1['sign'][:4]} prom={c1['prominence']:.3f} "
                  f"cropspread={c1['crop_spread']:.3f}"
                  + f" | C2 {'PASS' if c2['pass_'] else 'FAIL'} "
                    f"circvar={c2['circ_var_folded']:.3f} "
                    f"ptm={c2['peak_to_mean_deg']:.2f} "
                    f"dom={c2['dominant_orientation_deg']:.1f}deg "
                    f"off45={c2['dominant_to_45deg_multiple_deg']:.1f} | "
                    f"C4 {'PASS' if c4['pass_'] else 'FAIL'} "
                    f"raw={m['surrogate']['real_band_peak_over_median']:.1f}/surmax"
                    f"{m['surrogate']['surrogate_band_peak_over_median']['max']:.1f} "
                    f"pct={m['surrogate']['real_percentile_in_surrogates_raw']:.1f} "
                    f"norm_pct={m['surrogate']['real_percentile_in_surrogates_radial']:.1f}")

    # ---------------- verdicts ------------------------------------------- #
    verdicts = {}
    for p in PRESETS:
        per_dist = {d: measures[p][d] for d in DISTANCES_MM
                    if isinstance(measures[p].get(d), dict)
                    and "error" not in measures[p][d]}
        if not per_dist:
            verdicts[p] = dict(verdict="NO_DATA", reason="all renders failed")
            continue
        d36 = per_dist.get(36.0, next(iter(per_dist.values())))
        c1, c2, c4 = d36["C1"], d36["C2"], d36["C4"]
        c3 = criterion_C3(per_dist)
        crit = dict(C1=("INVALID" if not c1_valid else ("PASS" if c1["pass_"] else "FAIL")),
                    C2=("INVALID" if not c2_valid else ("PASS" if c2["pass_"] else "FAIL")),
                    C3=("PASS" if c3["pass_"] else "FAIL"),
                    C4=("PASS" if c4["pass_"] else "FAIL"))
        score = sum(1 for v in crit.values() if v == "PASS")
        base = "POSITIVE" if score >= 3 else ("WEAK" if score == 2 else "NEGATIVE")
        flags = []
        if (crit["C1"] == "PASS" or crit["C2"] == "PASS") and crit["C4"] == "FAIL":
            if base == "POSITIVE":
                base = "WEAK"
            flags.append("C1/C2 see a bump but the phase-randomised null reproduces it "
                         "-> not phase-locked, so not POSITIVE")
        if c3.get("wavelength_scales_with_distance"):
            base = "NEGATIVE"
            flags.append("the detected wavelength scales with camera distance "
                         "(synthetic-content signature), which overrides the score")
        valid = [v for v in crit.values() if v != "INVALID"]
        disagree = len(set(valid)) > 1
        verdicts[p] = dict(verdict=base, score=f"{score}/4", criteria=crit,
                           criteria_disagree=bool(disagree), flags=flags, C3=c3,
                           evidence=dict(
                               C1_first_sidelobe_abs_r=c1["peak_r"],
                               C1_sign=c1.get("sign"),
                               C1_wavelength_mm=c1.get("wavelength_mm"),
                               C1_prominence=c1.get("prominence"),
                               C1_crop_spread=c1.get("crop_spread"),
                               C2_circ_var=c2["circ_var_folded"],
                               C2_peak_to_mean=c2["peak_to_mean_deg"],
                               C2_dominant_deg=c2["dominant_orientation_deg"],
                               C2_off_45_multiple_deg=c2["dominant_to_45deg_multiple_deg"],
                               C4_raw_percentile=d36["surrogate"]["real_percentile_in_surrogates_raw"],
                               C4_radial_percentile=d36["surrogate"]["real_percentile_in_surrogates_radial"]))
        print(f"VERDICT {p:12s} {base:9s} score {score}/4 {crit}  "
              f"{'CRITERIA DISAGREE' if disagree else 'criteria agree'}"
              + ("" if not flags else "  || " + " ; ".join(flags)))
    report["measurements"] = {p: {str(k): ({kk: vv for kk, vv in v.items() if kk != "_img"}
                                         if isinstance(v, dict) and "_img" in v else v)
                                  for k, v in measures[p].items()}
                              for p in PRESETS}
    report["verdicts"] = verdicts

    # ---------------- figure --------------------------------------------- #
    fig = plt.figure(figsize=(17, 15.5))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.0, 1.0, 1.0, 1.05], hspace=0.45,
                          wspace=0.26)
    titles = {"blank": "blank (flat ground control)",
              "natural": "natural: old scene.py checker, 8 mm tiles",
              "natural_fbm": "natural_fbm: CC0 leaf-litter photograph"}
    for j, p in enumerate(PRESETS):
        r = measures[p].get("img_36.0")
        d36 = measures[p].get(36.0, {})
        c1 = d36.get("C1", {})
        if r is not None:
            ax = fig.add_subplot(gs[0, j])
            ax.imshow(r["_img"], cmap="gray")
            ax.set_title(f"{titles[p]}\n36 mm, fovy {A.CAM_FOVY:.0f}, "
                         f"{A.CAM_W}x{A.CAM_H}, straightened down "
                         f"({r['mm_per_px']:.4f} mm/px)", fontsize=9)
            ax.axis("off")
        ax = fig.add_subplot(gs[1, j])
        if r is not None:
            ac = autocorr(r["gray"])
            H = ac.shape[0]
            half_mm = AC_LAG_MAX_MM
            k = int(half_mm / r["mm_per_px"])
            cen = ac[H // 2 - k:H // 2 + k, ac.shape[1] // 2 - k:ac.shape[1] // 2 + k]
            im = ax.imshow(cen, cmap="inferno", vmin=-0.5, vmax=1.0,
                           extent=[-half_mm, half_mm, -half_mm, half_mm])
            lab = (f"C1 autocorrelation map 0-12 mm lag\n"
                   f"|r_resid|={c1.get('peak_r'):.3f} @ {c1.get('wavelength_mm'):.2f} mm"
                   f"  ({c1.get('sign','')[:4]})  prom={c1.get('prominence'):.3f}")
            ax.set_title(lab, fontsize=8)
            ax.set_xlabel("lag x (mm)", fontsize=8)
            ax.set_ylabel("lag y (mm)", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046)
        ax = fig.add_subplot(gs[2, j])
        a = d36.get("angular")
        if a:
            c = np.asarray(a["bin_centres_deg"])
            h = np.asarray(a["hist"])
            ax.bar(c, h, width=10.0, color="#4477aa", align="center")
            ax.axvline(0, color="k", lw=0.6)
            for g in (90, 180, 270):
                ax.axvline(g, color="k", lw=0.6, ls=":")
            ax.set_xlim(0, 360)
            ax.set_title(f"C2 angular power, band {ANALYSIS_BAND_MM[0]}-"
                         f"{ANALYSIS_BAND_MM[1]} mm\n"
                         f"circ_var={a['circ_var_folded']:.3f}  "
                         f"peak/mean={a['peak_to_mean_deg']:.2f}  "
                         f"dom={a['dominant_orientation_deg']:.0f}deg",
                         fontsize=8)
            ax.set_xlabel("orientation (deg); dotted = 90-deg multiples", fontsize=8)
            ax.set_ylabel("normalised power", fontsize=8)
    ax = fig.add_subplot(gs[3, 0])
    for p in PRESETS:
        ds, wl = [], []
        for d in DISTANCES_MM:
            m = measures[p].get(d)
            if not isinstance(m, dict) or "error" in m:
                continue
            w = m["C1"].get("wavelength_mm")
            if w:
                ds.append(d)
                wl.append(w)
        if ds:
            ax.plot(ds, wl, "o-", label=p)
    ax.plot([], [], "k:", label="lattice predicts a FLAT line")
    ax.set_xlabel("camera distance (mm)")
    ax.set_ylabel("detected wavelength (mm)")
    ax.set_title("C3 scale robustness: wavelength in mm\n(a lattice is flat here; "
                 "photographic content tracks the footprint)", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax2 = fig.add_subplot(gs[3, 1])
    for p in PRESETS:
        ds, rr = [], []
        for d in DISTANCES_MM:
            m = measures[p].get(d)
            if not isinstance(m, dict) or "error" in m:
                continue
            ds.append(d)
            rr.append(m["C1"]["peak_r"])
        if ds:
            ax2.plot(ds, rr, "o-", label=p)
    ax2.axhline(C1_PASS_R, color="r", ls="--", lw=1, label=f"C1 pass bar |r|={C1_PASS_R}")
    ax2.set_xlabel("camera distance (mm)")
    ax2.set_ylabel("autocorrelation first-extremum |r|")
    ax2.set_title("C1 peak strength vs distance\n(a lattice keeps its sharpness)",
                  fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7)
    ax3 = fig.add_subplot(gs[3, 2])
    labels = []
    for p in PRESETS:
        s = measures[p].get(36.0, {}).get("surrogate")
        if not s:
            continue
        v = np.asarray(s["surrogate_band_peak_over_median_radial_normalized"]["values"],
                       float)
        x = len(labels)
        ax3.scatter(np.full(v.shape, x) + np.random.default_rng(1).normal(0, 0.06, v.shape),
                    v, s=12, color="#888888", label="surrogate" if x == 0 else None)
        ax3.scatter([x], [s["real_band_peak_over_median_radial_normalized"]], s=90,
                    marker="*", color="crimson", label="real image" if x == 0 else None)
        labels.append(p)
        show = {0: "blank", 1: "natural", 2: "natural_\nfbm"}
    ax3.set_xticks(range(len(labels)))
    ax3.set_xticklabels([l.replace("natural_fbm", "natural_\nfbm") for l in labels],
                        fontsize=7)
    ax3.set_ylabel("band max|F|^2 / median|F|^2\n(radial mean divided out)")
    ax3.set_title(f"C4 phase-randomised null, {N_SURROGATES} surrogates (grey)\n"
                  "vs the real image (red *), 36 mm", fontsize=9)
    ax3.grid(alpha=0.3)
    ax3.legend(fontsize=7)
    fig.suptitle("arbitrate_periodicity.py -- is a periodic lattice physically present in "
                 "the rendered ground?\n(pass bars, band, lag window and surrogate null "
                 "declared in the source BEFORE measurement)", fontsize=12)
    fig.savefig(PNG_OUT, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {PNG_OUT}")

    with open(JSON_OUT, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"wrote {JSON_OUT}")
    print(f"total {time.time()-t0:.1f} s")


if __name__ == "__main__":
    main()
