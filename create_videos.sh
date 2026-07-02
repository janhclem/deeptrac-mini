#!/bin/bash

# Create videos from plot and plot_emulation directories
# Requires ffmpeg to be installed

OUTPUT_DIR="."
PLOT_DIR="./plot"
PLOT_EMULATION_DIR="./plot_emulation"
FPS=10

# Video for plot directory (folder 0 only)
if [ -d "$PLOT_DIR/0" ]; then
    echo "[INFO] Creating video from $PLOT_DIR/0..."
    ffmpeg -y -r $FPS -pattern_type glob -i "$PLOT_DIR/0/mass_*.png" \
        -c:v libx264 -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
        -vcodec libx264 "${OUTPUT_DIR}/mixing.mp4"
    echo "[INFO] Video saved to ${OUTPUT_DIR}/mixing.mp4"
else
    echo "[WARNING] $PLOT_DIR/0 not found, skipping."
fi

# Video for plot_emulation directory (folder 0 only)
if [ -d "$PLOT_EMULATION_DIR/0" ]; then
    echo "[INFO] Creating video from $PLOT_EMULATION_DIR/0..."
    ffmpeg -y -r $FPS -pattern_type glob -i "$PLOT_EMULATION_DIR/0/mass_*.png" \
        -c:v libx264 -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
        -vcodec libx264 "${OUTPUT_DIR}/mixing_emulation.mp4"
    echo "[INFO] Video saved to ${OUTPUT_DIR}/mixing_emulation.mp4"
else
    echo "[WARNING] $PLOT_EMULATION_DIR/0 not found, skipping."
fi

echo "[INFO] Done."
