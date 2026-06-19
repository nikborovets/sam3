#!/bin/bash

INPUT_BASE_DIR="/workspace/data_mount/third_wave_tracker_rgb_input/E-3023-A5-input-rgb"
ANNO_BASE_DIR="/workspace/3023-cvat-annotations/2stage"
PYTHON_SCRIPT="1_track_and_save.py"

RESULTS_ROOT_DIR="/workspace/3023-A5-SAM3-results/2stage"
mkdir -p "$RESULTS_ROOT_DIR"

SEQUENCES=("seq1" "seq2" "seq3" "seq4" "seq5")

echo "INPUT_BASE_DIR: $INPUT_BASE_DIR"

for SEQ in "${SEQUENCES[@]}"; do
    echo "----------------------------------------"
    
    ANNO_DIR=$(ls -d ${ANNO_BASE_DIR}/${SEQ}-job* 2>/dev/null | head -n 1)
    
    if [ -z "$ANNO_DIR" ] || [ ! -d "$ANNO_DIR" ]; then
        echo "SKIPPING: $SEQ (No annotation folder found in $ANNO_BASE_DIR)"
        continue
    fi

    echo "PROCESSING: $SEQ (Found: $(basename $ANNO_DIR))"
    
    SEQ_RESULT_NAME="$(basename $ANNO_DIR)_results"
    CURRENT_OUT_DIR="${RESULTS_ROOT_DIR}/${SEQ_RESULT_NAME}"
    OUT_NPZ_DIR="${CURRENT_OUT_DIR}/masks_npz"

    mkdir -p "$OUT_NPZ_DIR"

    python "$PYTHON_SCRIPT" \
        --inputs "${INPUT_BASE_DIR}/${SEQ}" \
        --masks "${ANNO_DIR}/SegmentationClass/${SEQ}" \
        --labelmap "${ANNO_DIR}/labelmap.txt" \
        --out-npz "$OUT_NPZ_DIR" \
        --device "cuda"

    if [ $? -ne 0 ]; then
        echo "CRITICAL ERROR: $SEQ failed. Stopping all tasks."
        exit 1
    fi
    
    echo "SUCCESS: $SEQ finished. Results in $OUT_NPZ_DIR"
done

echo "========================================"
echo "All available sequences processed!"
echo "Check results in: $RESULTS_ROOT_DIR"
