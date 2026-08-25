#!/bin/bash

# Create videos from plot and plot_emulation directories
# Requires ffmpeg to be installed

OUTPUT_DIR="."
PLOT_DIR="./plot_emulation"
FPS=10

for SCENARIO in {0..2}; do
  if [ -d "$PLOT_DIR/$SCENARIO" ]; then
    echo "[INFO] Creating video from $PLOT_DIR/$SCENARIO..."
    ffmpeg -y -r $FPS -pattern_type glob -i "$PLOT_DIR/$SCENARIO/mass_*.png" \
        -c:v libx264 -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
        -vcodec libx264 "${OUTPUT_DIR}/mixing_${SCENARIO}.mp4"
    echo "[INFO] Video saved to ${OUTPUT_DIR}/mixing_${SCENARIO}.mp4"
  else
    echo "[WARNING] $PLOT_DIR/${SCENARIO} not found, skipping."
  fi
done 

echo "[INFO] Done."
