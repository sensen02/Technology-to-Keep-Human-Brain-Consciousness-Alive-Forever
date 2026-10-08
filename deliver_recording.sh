#!/bin/bash
# Deliver the END-TO-END RECORDING PIPELINE results to the local Desktop.
#
# What this delivers, and why each piece is there:
#   * the three pipeline scripts, so the run is reproducible from the desktop copy;
#   * the electrode payload layer and the body-backend change it hooks into;
#   * comparison.json and recording_index.json -- every number;
#   * the FIGURES, because a long run has to be looked at, not only tabulated;
#   * one episode's camera frames in both views, so the raw material is visible
#     without digging into outputs/;
#   * the Chinese report.
set -u
SRC=/run/media/sensen/Data2/cell_wound_prototype
DST=/home/sensen/Desktop/cell_wound_prototype
mkdir -p "$DST/recording_pipeline" "$DST/recording_pipeline/figures" \
         "$DST/recording_pipeline/frames_example_worldcam" \
         "$DST/recording_pipeline/frames_example_detailcam"

# scripts
cp -v "$SRC"/electrode_payload.py "$SRC"/tools_measure_body_mass.py \
      "$SRC"/tools_rebuild_recording_index.py "$SRC"/tools_recording_report.py \
      "$SRC"/tools_test_electrode_payload.py \
      "$SRC"/run_recording_pipeline.py "$SRC"/run_recording_vision.py \
      "$SRC"/run_recording_verify.py "$SRC"/run_recording_figures.py \
      "$DST/recording_pipeline/"
cp -v "$SRC"/engine/embodied/body_backend.py "$DST/recording_pipeline/body_backend.py"

# numbers
for f in recording_index.json comparison.json; do
  [ -f "$SRC/outputs/electrode_payload/$f" ] && \
    cp -v "$SRC/outputs/electrode_payload/$f" "$DST/recording_pipeline/"
done
[ -f "$SRC/outputs/electrode_payload/body_mass.json" ] && \
  cp -v "$SRC/outputs/electrode_payload/body_mass.json" "$DST/recording_pipeline/"

# figures
for f in "$SRC"/outputs/electrode_payload/figures/*.png; do
  [ -f "$f" ] && cp -v "$f" "$DST/recording_pipeline/figures/"
done

# example frames (one seed, one condition)
EP="$SRC/outputs/electrode_payload/episode_control_seed0"
if [ -d "$EP/frames" ]; then
  for i in 00000 00048 00096 00144 00192 00239; do
    f="$EP/frames/f$i.png"; [ -f "$f" ] && \
      cp -v "$f" "$DST/recording_pipeline/frames_example_worldcam/"
    f="$EP/frames_detailcam/f$i.png"; [ -f "$f" ] && \
      cp -v "$f" "$DST/recording_pipeline/frames_example_detailcam/"
  done
fi

# reports and this script
for f in RECORDING_PIPELINE.zh-CN.md RECORDING_PIPELINE_METHOD.zh-CN.md \
         RECORDING_PIPELINE_READ_ME.zh-CN.md; do
  [ -f "$SRC/$f" ] && cp -v "$SRC/$f" "$DST/recording_pipeline/"
done
cp -v "$SRC/deliver_recording.sh" "$DST/recording_pipeline/"

echo
echo "交付到: $DST/recording_pipeline"
du -sh "$DST/recording_pipeline"
