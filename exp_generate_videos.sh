#!/bin/bash

# Путь к папке с экспериментами
EXP_DIR="/workspace/data_mount/experiments_tmp"
# Путь куда сохранять финальные видео
OUTPUT_DIR="/workspace/data_mount/experiments_results_videos"

# Создаем папку для видео, если её нет
mkdir -p "$OUTPUT_DIR"

echo "Начинаем генерацию видео для экспериментов..."

# Проходим по всем папкам exp_*
for dir in "$EXP_DIR"/exp_*; do
    if [ -d "$dir" ]; then
        exp_name=$(basename "$dir")
        masks_dir="$dir/masks_out"
        # overlay_dir="$dir/overlay"
        
        # 1. Генерируем видео из masks_out
        if [ -d "$masks_dir" ]; then
            echo "Создаем masks видео для $exp_name..."
            ffmpeg -y -framerate 30 -pattern_type glob -i "$masks_dir/*.png" \
                -c:v libx264 -pix_fmt yuv420p \
                "$OUTPUT_DIR/${exp_name}_masks.mp4"
        else
            echo "ПРЕДУПРЕЖДЕНИЕ: Папка $masks_dir не найдена, пропускаем."
        fi
        
        # # 2. Генерируем видео из overlay (для наглядности)
        # if [ -d "$overlay_dir" ]; then
        #     echo "Создаем overlay видео для $exp_name..."
        #     ffmpeg -y -framerate 30 -pattern_type glob -i "$overlay_dir/*.png" \
        #         -c:v libx264 -pix_fmt yuv420p \
        #         "$OUTPUT_DIR/${exp_name}_overlay.mp4"
        # else
        #     echo "ПРЕДУПРЕЖДЕНИЕ: Папка $overlay_dir не найдена, пропускаем."
        # fi
    fi
done

echo "=== Готово! Видео сохранены в $OUTPUT_DIR ==="
