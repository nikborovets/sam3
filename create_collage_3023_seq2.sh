#!/bin/bash
# Коллаж 2x2: RGB, 1stage, 2stage, 3stage для E-3023-A5 (сначала — seq2; поменяйте SEQ).
set -euo pipefail

SEQ=2
INPUT_DIR="/workspace/data_mount/third_wave_tracker_rgb_input/videos/for_collage/3023"
OUT_NAME="collage_E-3023-A5_seq${SEQ}_2x2_rgb_1-3stage"
OUTPUT_FILE="${INPUT_DIR}/${OUT_NAME}.mp4"

# Ожидаемые ролики (после generate_E-3023-A5_collage_videos.sh)
F_RGB="${INPUT_DIR}/rgb_E-3023-A5-seq${SEQ}_video.mp4"
F1="${INPUT_DIR}/1stage_E-3023-A5-seq${SEQ}_video.mp4"
F2="${INPUT_DIR}/2stage_E-3023-A5-seq${SEQ}_video.mp4"
F3="${INPUT_DIR}/3stage_E-3023-A5-seq${SEQ}_video.mp4"

for f in "$F_RGB" "$F1" "$F2" "$F3"; do
	if [ ! -f "$f" ]; then
		echo "Ошибка: нет файла: $f"
		exit 1
	fi
done

echo "Собираем коллаж 2x2 (3023) seq${SEQ} -> $OUTPUT_FILE"

# Панель 640x360, подписи сверху; сетка: [RGB | 1st] / [2nd | 3rd]
W=640
H=360
FS=26

ffmpeg -y \
	-i "$F_RGB" -i "$F1" -i "$F2" -i "$F3" \
	-filter_complex "
		[0:v] scale=${W}:${H}:force_original_aspect_ratio=decrease, pad=${W}:${H}:(ow-iw)/2:(oh-ih)/2, setsar=1, drawtext=text='E-3023-A5 seq${SEQ} — RGB':fontcolor=white:fontsize=${FS}:box=1:boxcolor=black@0.5:boxborderw=4:x=(w-text_w)/2:y=8 [v0];
		[1:v] scale=${W}:${H}:force_original_aspect_ratio=decrease, pad=${W}:${H}:(ow-iw)/2:(oh-ih)/2, setsar=1, drawtext=text='1 stage':fontcolor=white:fontsize=${FS}:box=1:boxcolor=black@0.5:boxborderw=4:x=(w-text_w)/2:y=8 [v1];
		[2:v] scale=${W}:${H}:force_original_aspect_ratio=decrease, pad=${W}:${H}:(ow-iw)/2:(oh-ih)/2, setsar=1, drawtext=text='2 stage':fontcolor=white:fontsize=${FS}:box=1:boxcolor=black@0.5:boxborderw=4:x=(w-text_w)/2:y=8 [v2];
		[3:v] scale=${W}:${H}:force_original_aspect_ratio=decrease, pad=${W}:${H}:(ow-iw)/2:(oh-ih)/2, setsar=1, drawtext=text='3 stage':fontcolor=white:fontsize=${FS}:box=1:boxcolor=black@0.5:boxborderw=4:x=(w-text_w)/2:y=8 [v3];
		[v0][v1][v2][v3] xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0:shortest=1 [out]
	" \
	-map "[out]" -c:v libx264 -crf 20 -preset fast \
	"$OUTPUT_FILE"

echo "Готово: $OUTPUT_FILE"
