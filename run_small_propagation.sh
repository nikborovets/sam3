#!/bin/bash

WEIGHTS="/workspace/data_mount/model_weights/sam3/sam3.pt"
INPUT_BASE_DIR="/workspace/data_mount/third_wave_tracker_rgb_input/every_n_frames/E-1023-R1-input-rgb/seq1_orbbec_every_5"
SEQ="seq1_orbbec"
ANNO_DIR="/workspace/data_mount/third_wave_tracker_rgb_input/every_n_frames/E-1023-R1-input-rgb/seq1_orbbec_export_11-03-26"
RESULTS_ROOT_DIR="/workspace/data_mount/third_wave_tracker_rgb_input/every_n_frames/E-1023-R1-input-rgb/small_results"
mkdir -p "$RESULTS_ROOT_DIR"

CURRENT_OUT_DIR="${RESULTS_ROOT_DIR}/${SEQ}_results_every_5"
OUT_OVERLAY="${CURRENT_OUT_DIR}/overlay"
OUT_MASKS="${CURRENT_OUT_DIR}/masks"

mkdir -p "$OUT_OVERLAY" "$OUT_MASKS"

echo "PROCESSING: $SEQ"
echo "INPUT_BASE_DIR: $INPUT_BASE_DIR"
echo "ANNO_DIR: $ANNO_DIR"
echo "RESULTS_ROOT_DIR: $RESULTS_ROOT_DIR"
echo "CURRENT_OUT_DIR: $CURRENT_OUT_DIR"
echo "OUT_OVERLAY: $OUT_OVERLAY"
echo "OUT_MASKS: $OUT_MASKS"

python propagate_sam3_ivan.py \
    --weights "$WEIGHTS" \
	--inputs "${INPUT_BASE_DIR}" \
	--masks "${ANNO_DIR}/SegmentationClass/${SEQ}" \
	--labelmap "${ANNO_DIR}/labelmap.txt" \
	--out-overlay "$OUT_OVERLAY" \
	--out-masks "$OUT_MASKS" \
	--device "cuda"

if [ $? -ne 0 ]; then
    echo "CRITICAL ERROR: $SEQ failed. Stopping all tasks."
    exit 1
fi

echo "SUCCESS: $SEQ finished. Results in $CURRENT_OUT_DIR"