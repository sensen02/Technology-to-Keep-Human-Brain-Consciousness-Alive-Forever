#!/bin/bash
# Deliver the results of this workstream to the local Desktop.
set -u
SRC=/run/media/sensen/Data2/cell_wound_prototype
DST=/home/sensen/Desktop/cell_wound_prototype
mkdir -p "$DST"

# ---- code written/used in this workstream ----
cp -v "$SRC/cpm.py"                 "$DST/" 2>/dev/null
cp -v "$SRC/viz.py"                 "$DST/" 2>/dev/null
cp -v "$SRC/vasculogenesis.py"      "$DST/" 2>/dev/null
cp -v "$SRC/wound_healing.py"       "$DST/" 2>/dev/null
cp -v "$SRC/epidermis.py"           "$DST/" 2>/dev/null
cp -v "$SRC/run_cpm_selftest.py"    "$DST/" 2>/dev/null
cp -v "$SRC/run_p1_vasculogenesis.py" "$DST/" 2>/dev/null
cp -v "$SRC/run_p3_wound.py"        "$DST/" 2>/dev/null
cp -v "$SRC/run_p4_epidermis.py"    "$DST/" 2>/dev/null
cp -v "$SRC/make_report_phases.py"  "$DST/" 2>/dev/null
cp -v "$SRC/regen_panels.py"        "$DST/" 2>/dev/null
cp -v "$SRC/merks.py"               "$DST/" 2>/dev/null
cp -v "$SRC/run_p1b_merks.py"       "$DST/" 2>/dev/null
cp -v "$SRC/run_p1_network.py"      "$DST/" 2>/dev/null
cp -v "$SRC/run_p1c_phase.py"       "$DST/" 2>/dev/null
cp -v "$SRC/run_p1d_fine.py"        "$DST/" 2>/dev/null
cp -v "$SRC/run_p1d_combine.py"     "$DST/" 2>/dev/null
cp -v "$SRC/MEASURED_ANCHORS.md"    "$DST/" 2>/dev/null

# ---- evidence: metrics JSON ----
for f in metrics_cpm_selftest.json metrics_p1.json p3_wound_metrics.json \
         p4_epidermis_metrics.json metrics_physicell_vasculogenesis.json \
         p1_network_metrics.json p1b_merks_metrics.json p1d_merks_metrics.json \
         PHYSICELL_NOTES.md; do
  [ -f "$SRC/outputs/$f" ] && cp -v "$SRC/outputs/$f" "$DST/"
done

# ---- figures ----
for f in p1_sprouting_panels.png p1_contact_inhibition.png p1_metrics.png \
         p3_wound_panels.png p3_wound_curves.png p3_wound.gif \
         p4_epidermis_panels.png p4_epidermis_curves.png \
         p4_epidermis_gradient_sweep.png p4_epidermis.gif \
         physicell_vasculogenesis.png \
         p1_network_timeseries.png p1_network_control.png p1_network.gif \
         p1b_merks_clusters.png p1b_merks_transition.png p1b_merks_lambda.png \
         p1b_merks_denovo.png p1d_merks_transition_fine.png \
         p1d_merks_phaseplane.png p1d_merks_morphology.png; do
  [ -f "$SRC/outputs/$f" ] && cp -v "$SRC/outputs/$f" "$DST/"
done

# ---- report ----
[ -f "$SRC/outputs/REPORT_P1P4.zh-CN.md" ] && cp -v "$SRC/outputs/REPORT_P1P4.zh-CN.md" "$DST/"

echo
echo "delivered to $DST"
ls -la "$DST" | tail -30
