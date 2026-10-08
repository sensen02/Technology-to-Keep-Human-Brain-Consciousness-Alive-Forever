# 400-cell Drosophila epithelial wound calcium prototype

## Preregistered scope (before simulation results)
Primary source: Stevens et al., Molecular Biology of the Cell 2023, https://pmc.ncbi.nlm.nih.gov/articles/PMC10208100/ DOI 10.1091/mbc.E22-08-0361.
Source implementation/data: https://github.com/mshutson/wound-calcium-LRCa (MIT).

This is an independently implemented reduced Python port, not a full reproduction of the original Mathematica simulator and not a virtual animal. Target: early membrane-microtear-induced calcium signaling in ~400 epithelial cells. Fixed hexagonal cells, cytosol/ER calcium, IP3, receptor inactivation, gap-junction transfer; no explicit deformable mechanics, ATP metabolism, chemical erosion or long-term wound repair. Prescribed injury is not a computed mechanical injury.

## Scope and comparison
- Real cell area retained at approximately 43 square micrometres; do not enlarge cells to claim coverage of late remote waves.
- Simulate only first 25 seconds, before the delayed ligand-driven distal response. Extracellular ligand/receptor occupancy set to zero: an explicit model truncation, not a measured absence of signaling.
- Compare experimental controlCaRadData.m for 2.14 <= t <= 23.54 s. t=0 is reported separately due to acquisition/injury alignment and the explicit simulator onset.
- Primary observable: fluorescence annular profile outer half-height radius, approximating original fig2.nb measurement. No fitting of biological parameters or observation threshold to improve the comparison.
- Report MAE/RMSE/bias in micrometres, residual curves, and fraction of points within one equivalent-area cell diameter. These are descriptive metrics, not a percent-realism score.
- Original publication parameters partly derive from this experimental context. This is a retrospective reference comparison, NOT independent biological validation. The available control trace is one wound; repeated simulation seeds are not biological replicates.
- Hypothesis-level ablations: reducing GJ transfer should suppress early expansion; PLC reduction should be less disruptive to this initial direct calcium influx. No numeric knockdown raw data is claimed.

## Numerical controls
No-wound stability, finite/nonnegative concentrations, receptor bounds, area-weighted conservative GJ exchange, tighter solver tolerance, and larger physical domain at unchanged cell size. Use shared cell heterogeneity when comparing domain size where possible. Distinguish numerical agreement from biological validity. Save raw state trajectories, exact parameters, random seeds and source commit.

## Post-run amendment and audit
The initially specified 400 cells censored most late early-wave observations (only 3/11 reference points available). Added 576 cells as primary display and 784 as boundary reference without changing cell size or biological parameters. Added tighter solver and two additional seeds for 576. This is a disclosed numerical-domain correction, not an originally preregistered choice.

The fluorescence operator is approximate: raw CtoF times GCaMP multiplier with black ablated core and pixel annuli, not the original video's affine brightness rescaling/clipping before heterogeneous multiplication. Absolute error can include that observation difference. No-response gate rejects static heterogeneity (<1e-6 fluorescence change); it is a numerical gate, not experimentally calibrated detection sensitivity. Censor flags include no response and missing crossing. Radius errors for interventions are descriptive distances to CONTROL data, NOT intervention validation. Calcium predictions near damaged cells reach hundreds of micromolar; GCaMP saturation prevents validating those values from radius data. Raster-resolution comparison currently retains a nine-bin exclusion and is only a sensitivity check, not strict resolution-invariant convergence.

## Delivery
Runnable code, experimental CSV with provenance, simulation arrays, metrics JSON, figures, animation and Chinese report. Inspect generated images directly. Primary workspace: /run/media/sensen/Data2/cell_wound_prototype . Deliver convenient report/figures to /home/sensen/Desktop/cell_wound_prototype . CPU implementation first; no claimed H20/RX9070XT benchmark without running it.
