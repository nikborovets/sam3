#!/bin/bash
# Один ряд 1x3: RGB, 1 stage, 2 stage для 03-25-26 E-R1-1023 (сначала — seq2; поменяйте SEQ).
set -euo pipefail

SEQ=2
TAG="03-25-26_E-R1-1023"
INPUT_DIR="/workspace/data_mount/third_wave_tracker_rgb_input/videos/for_collage/1023"
OUT_NAME="collage_${TAG}_seq${SEQ}_1x3_rgb_1-2stage"
OUTPUT_FILE="${INPUT_DIR}/${OUT_NAME}.mp4"

F_RGB="${INPUT_DIR}/rgb_${TAG}_seq${SEQ}_video.mp4"
F1="${INPUT_DIR}/1stage_${TAG}_seq${SEQ}_video.mp4"
F2="${INPUT_DIR}/2stage_${TAG}_seq${SEQ}_video.mp4"

for f in "$F_RGB" "$F1" "$F2"; do
	if [ ! -f "$f" ]; then
		echo "Ошибка: нет файла: $f"
		exit 1
	fi
done

echo "Собираем коллаж 1x3 (1023) seq${SEQ} -> $OUTPUT_FILE"

W=640
H=360
FS=24

ffmpeg -y \
	-i "$F_RGB" -i "$F1" -i "$F2" \
	-filter_complex "
		[0:v] scale=${W}:${H}:force_original_aspect_ratio=decrease, pad=${W}:${H}:(ow-iw)/2:(oh-ih)/2, setsar=1, drawtext=text='E-R1-1023 seq${SEQ} — RGB':fontcolor=white:fontsize=${FS}:box=1:boxcolor=black@0.5:boxborderw=4:x=(w-text_w)/2:y=8 [v0];
		[1:v] scale=${W}:${H}:force_original_aspect_ratio=decrease, pad=${W}:${H}:(ow-iw)/2:(oh-ih)/2, setsar=1, drawtext=text='1 stage':fontcolor=white:fontsize=${FS}:box=1:boxcolor=black@0.5:boxborderw=4:x=(w-text_w)/2:y=8 [v1];
		[2:v] scale=${W}:${H}:force_original_aspect_ratio=decrease, pad=${W}:${H}:(ow-iw)/2:(oh-ih)/2, setsar=1, drawtext=text='2 stage':fontcolor=white:fontsize=${FS}:box=1:boxcolor=black@0.5:boxborderw=4:x=(w-text_w)/2:y=8 [v2];
		[v0][v1][v2] xstack=inputs=3:layout=0_0|w0_0|w0+w1_0:shortest=1 [out]
	" \
	-map "[out]" -c:v libx264 -crf 20 -preset fast \
	"$OUTPUT_FILE"

echo "Готово: $OUTPUT_FILE"
