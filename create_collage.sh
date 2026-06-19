#!/bin/bash

# Папка с исходными видео
INPUT_DIR="/workspace/data_mount/experiments_results_videos"
# Выходной файл
OUTPUT_FILE="$INPUT_DIR/collage1.mp4"

# Проверяем наличие всех нужных файлов
missing_files=0
for f in all_rgb.mp4 exp_1_masks.mp4 exp_2_masks.mp4 exp_3_masks.mp4 exp_4_masks.mp4 exp_5_masks.mp4 exp_6_masks.mp4 exp_7_masks.mp4; do
    if [ ! -f "$INPUT_DIR/$f" ]; then
        echo "Ошибка: Файл $INPUT_DIR/$f не найден!"
        missing_files=1
    fi
done

if [ $missing_files -eq 1 ]; then
    echo "Отмена создания коллажа: отсутствуют необходимые файлы."
    exit 1
fi

echo "Начинаем создание коллажа 4x2..."

# Используем фильтр xstack для создания сетки 4x2
# Входы (-i):
# 0: all_rgb
# 1: exp_1
# 2: exp_2
# 3: exp_3
# 4: exp_4
# 5: exp_5
# 6: exp_6
# 7: exp_7

ffmpeg -y \
    -i "$INPUT_DIR/exp_1_masks.mp4" \
    -i "$INPUT_DIR/exp_2_masks.mp4" \
    -i "$INPUT_DIR/exp_5_masks.mp4" \
    -i "$INPUT_DIR/exp_7_masks.mp4" \
    -i "$INPUT_DIR/exp_3_masks.mp4" \
    -i "$INPUT_DIR/exp_4_masks.mp4" \
    -i "$INPUT_DIR/exp_6_masks.mp4" \
    -i "$INPUT_DIR/all_rgb.mp4" \
    -filter_complex "
        [0:v] drawtext=text='Exp 1 (N=150, 1 mask)':fontcolor=white:fontsize=36:box=1:boxcolor=black@0.5:boxborderw=5:x=(w-text_w)/2:y=10 [v0];
        [1:v] drawtext=text='Exp 2 (N=75, 1 mask)':fontcolor=white:fontsize=36:box=1:boxcolor=black@0.5:boxborderw=5:x=(w-text_w)/2:y=10 [v1];
        [2:v] drawtext=text='Exp 5 (N=150x2, 3 masks)':fontcolor=white:fontsize=36:box=1:boxcolor=black@0.5:boxborderw=5:x=(w-text_w)/2:y=10 [v2];
        [3:v] drawtext=text='Exp 7 (300 frames, 5 masks)':fontcolor=white:fontsize=36:box=1:boxcolor=black@0.5:boxborderw=5:x=(w-text_w)/2:y=10 [v3];
        [4:v] drawtext=text='Exp 3 (N=150, 2 masks)':fontcolor=white:fontsize=36:box=1:boxcolor=black@0.5:boxborderw=5:x=(w-text_w)/2:y=10 [v4];
        [5:v] drawtext=text='Exp 4 (N=75, 2 masks)':fontcolor=white:fontsize=36:box=1:boxcolor=black@0.5:boxborderw=5:x=(w-text_w)/2:y=10 [v5];
        [6:v] drawtext=text='Exp 6 (N=75x2, 3 masks)':fontcolor=white:fontsize=36:box=1:boxcolor=black@0.5:boxborderw=5:x=(w-text_w)/2:y=10 [v6];
        [7:v] drawtext=text='Original RGB':fontcolor=white:fontsize=36:box=1:boxcolor=black@0.5:boxborderw=5:x=(w-text_w)/2:y=10 [v7];
        [v0][v1][v2][v3][v4][v5][v6][v7]xstack=inputs=8:layout=0_0|w0_0|w0+w1_0|w0+w1+w2_0|0_h0|w0_h0|w0+w1_h0|w0+w1+w2_h0[out]
    " \
    -map "[out]" \
    -c:v libx264 -crf 23 -preset fast \
    "$OUTPUT_FILE"

echo "Коллаж успешно сохранен в: $OUTPUT_FILE"
