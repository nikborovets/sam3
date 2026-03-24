import shutil
import re
from pathlib import Path

# --- НАСТРОЙКИ ---
# step = 75
# # anno_name = "seq4_aruna_anno_4iter"
# # anno_name = "02-16-26_seq6_aruna_anno_1iter"
# anno_name = "seq1-job-53_results"
# # /home/n.borovets/user/segment_experiments/3023-A5-SAM3-results/seq1-job-53_results
# input_dir = Path(f"/workspace/3023-A5-SAM3-results/{anno_name}/masks")
# output_dir = Path(f"/workspace/3023-A5-SAM3-results/{anno_name}/masks_every_{step}")

# list_file_path = Path(f"/workspace/3023-A5-SAM3-results/{anno_name}/masks_every_{step}_list.txt")
# -----------------

def copy_every_n_frames(src: Path, dst: Path, n: int):
    """Выбирает каждый n-й кадр из src и копирует в dst."""
    dst.mkdir(parents=True, exist_ok=True)
    print(f"--- Копирование кадров (каждый {n}-й) ---")
    print(f"Из: {src}")
    print(f"В: {dst}")

    files = sorted([f for f in src.iterdir() if f.is_file()])
    count = 0

    for i, file_path in enumerate(files):
        is_nth = False
        
        # Если в имени файла (без расширения) есть точка, считаем это таймстемпом (например, 1619.140700000.png).
        # В этом случае ориентируемся на порядковый индекс файла в папке.
        if "." in file_path.stem:
            if i % n == 0:
                is_nth = True
        else:
            # Старый формат: извлекаем число (например, из frame000150.png)
            match = re.search(r'(\d+)', file_path.stem)
            if match:
                frame_idx = int(match.group(1))
                if frame_idx % n == 0:
                    is_nth = True
            else:
                # На всякий случай, если чисел нет вообще
                if i % n == 0:
                    is_nth = True

        if is_nth:
            shutil.copy2(file_path, dst / file_path.name)
            count += 1
    
    print(f"Копирование завершено. Всего скопировано: {count}")

def save_stems_to_txt(folder: Path, output_txt: Path):
    """Читает все файлы в folder и записывает их имена (без расширений) в txt."""
    print(f"\n--- Генерация списка имен ---")
    print(f"Папка: {folder}")
    
    # Собираем все stem (имена без расширений) и сортируем их
    stems = sorted([f.stem for f in folder.iterdir() if f.is_file()])
    
    with output_txt.open("w") as f:
        for stem in stems:
            f.write(stem + "\n")
            
    print(f"Список сохранен: {output_txt}")
    print(f"Количество записей: {len(stems)}")

def pick_every_n_frames_input():
    step = 5
    anno_names = ["seq1_orbbec", "seq1_rs", "seq2_orbbec", "seq2_rs", "seq3_orbbec"]
    # seq1_orbbec
    # seq1_rs
    # seq2_orbbec
    # seq2_rs
    # seq3_orbbec
    for anno_name in anno_names:
        input_dir = Path(f"/workspace/data_mount/third_wave_tracker_rgb_input/E-1023-R1-input-rgb/{anno_name}")
        output_dir = Path(f"/workspace/data_mount/third_wave_tracker_rgb_input/every_n_frames/E-1023-R1-rgb/input_every_5/{anno_name}_every_{step}")

        copy_every_n_frames(input_dir, output_dir, step)

def pick_every_n_frames_masks():
    step = 15
    # anno_name = "seq4_aruna_anno_4iter"
    # anno_name = "02-16-26_seq6_aruna_anno_1iter"
    # anno_names = ["seq1-job-53_results", "seq2-job-54_results", "seq3-job-55_results", "seq4-job-56_results", "seq5-job-57_results"]
    # /home/n.borovets/user/segment_experiments/3023-A5-SAM3-results/seq1-job-53_results

    anno_names = ["seq1_rs_export_11-03-26_results", "seq2_orbbec_export_11-03-26_results", "seq2_rs_export_11-03-26_results", "seq3_orbbec_export_11-03-26_results"]
    # "seq1_orbbec_export_11-03-26_results"

    for anno_name in anno_names:
        # input_dir = Path(f"/workspace/3023-A5-SAM3-results/{anno_name}/masks")
        # output_dir = Path(f"/workspace/3023-A5-SAM3-results/{anno_name}/masks_every_{step}")
        input_dir = Path(f"/workspace/data_mount/third_wave_tracker_rgb_input/every_n_frames/E-1023-R1-rgb/results_every_5/{anno_name}/masks")
        output_dir = Path(f"/workspace/data_mount/third_wave_tracker_rgb_input/every_n_frames/E-1023-R1-rgb/result_masks_every_{step*5}/{anno_name}")

        list_file_path = Path(f"/workspace/data_mount/third_wave_tracker_rgb_input/every_n_frames/E-1023-R1-rgb/result_masks_every_{step*5}/{anno_name}_masks_every_{step*5}_list.txt")
        # 1. Выполняем выборку и копирование
        copy_every_n_frames(input_dir, output_dir, step)
        
        
        # 2. Формируем список на основе того, что реально оказалось в папке
        save_stems_to_txt(output_dir, list_file_path)

if __name__ == "__main__":
    pick_every_n_frames_masks()