#!/bin/bash

WEIGHTS="/workspace/data_mount/model_weights/sam3/sam3.pt"
# INPUT_BASE_DIR="/workspace/E-3023-A5-input-rgb"
# INPUT_BASE_DIR="/workspace/data_mount/third_wave_tracker_rgb_input/E-1023-R1-input-rgb"
INPUT_BASE_DIR="/workspace/data_mount/third_wave_tracker_rgb_input/every_n_frames/E-1023-R1-rgb/input_every_5"
# ANNO_BASE_DIR="/workspace/3023-cvat-annotations/1stage"
# ANNO_BASE_DIR="/workspace/3023-cvat-annotations/2stage"
ANNO_BASE_DIR="/workspace/E-1023-R1-cvat-annotations/1stage"
PYTHON_SCRIPT="propagate_sam3_ivan.py"

# RESULTS_ROOT_DIR="/workspace/3023-A5-SAM3-results/1stage"
# RESULTS_ROOT_DIR="/workspace/E-1023-R1-SAM3-results/1stage"
RESULTS_ROOT_DIR="/workspace/data_mount/third_wave_tracker_rgb_input/every_n_frames/E-1023-R1-rgb/results_every_5"
mkdir -p "$RESULTS_ROOT_DIR"

# SEQUENCES=("seq1" "seq2" "seq3" "seq4" "seq5" "seq6")
# SEQUENCES=("seq6")
# SEQUENCES=("seq1_rs" "seq1_orbbec" "seq2_rs" "seq2_orbbec" "seq3_orbbec" "seq4_orbbec" "seq5_orbbec")
SEQUENCES=("seq1_rs" "seq1_orbbec" "seq2_rs" "seq2_orbbec" "seq3_orbbec")

echo "INPUT_BASE_DIR: $INPUT_BASE_DIR"

for SEQ in "${SEQUENCES[@]}"; do
    echo "----------------------------------------"
    
    # ANNO_DIR=$(ls -d ${ANNO_BASE_DIR}/${SEQ}-job-* 2>/dev/null | head -n 1)
    ANNO_DIR=$(ls -d ${ANNO_BASE_DIR}/${SEQ}_export* 2>/dev/null | head -n 1)
    
    if [ -z "$ANNO_DIR" ] || [ ! -d "$ANNO_DIR" ]; then
        echo "SKIPPING: $SEQ (No annotation folder found in $ANNO_BASE_DIR)"
        continue
    fi

    echo "PROCESSING: $SEQ (Found: $(basename $ANNO_DIR))"
    
    SEQ_RESULT_NAME="$(basename $ANNO_DIR)_results"
    CURRENT_OUT_DIR="${RESULTS_ROOT_DIR}/${SEQ_RESULT_NAME}"
    OUT_OVERLAY="${CURRENT_OUT_DIR}/overlay"
    OUT_MASKS="${CURRENT_OUT_DIR}/masks"

    mkdir -p "$OUT_OVERLAY" "$OUT_MASKS"

    python "$PYTHON_SCRIPT" \
        --weights "$WEIGHTS" \
        --inputs "${INPUT_BASE_DIR}/${SEQ}_every_5" \
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
done

echo "========================================"
echo "All available sequences processed!"
echo "Check results in: $RESULTS_ROOT_DIR"
