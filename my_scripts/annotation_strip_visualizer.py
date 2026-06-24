"""
Визуализация полосы аннотаций: показывает, какие изображения из последовательности
были размечены вручную, в том числе по итерациям разметки.
"""
import argparse
import re
from pathlib import Path
from typing import Set, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm

def _get_cmap(name: str):
    try:
        return plt.colormaps.get_cmap(name)
    except AttributeError:
        return plt.get_cmap(name)

try:
    import cv2
except ImportError:
    cv2 = None


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

anno_name = "seq4_aruna_anno_4iter"
DEFAULT_PATH_IMAGES = "/workspace/seq4img"
DEFAULT_PATH_ANNOTATIONS = f"/workspace/{anno_name}/SegmentationClass/seq4img"
DEFAULT_PATH_OUTPUT = f"/workspace/{anno_name}_results/annotation_strip.png"


def _parse_index_from_name(name: str) -> Optional[int]:
    """
    Извлекает порядковый номер из имени файла (без расширения).
    Поддерживает: "0", "123", "frame0", "frame123", "img_0001" и т.п.
    """
    stem = Path(name).stem
    # Сначала пробуем как целое число
    if stem.isdigit():
        return int(stem)
    # Ищем число в имени (frame0, frame123, img_001 и т.д.)
    match = re.search(r"\d+", stem)
    if match:
        return int(match.group())
    return None


def get_image_indices(images_path: str | Path) -> List[int]:
    """
    Возвращает отсортированный список индексов изображений из папки.
    Индекс извлекается из имени файла (0, 1, 150, ... или frame0, frame150, ...).
    """
    p = Path(images_path)
    if not p.is_dir():
        raise FileNotFoundError(f"Папка не найдена: {images_path}")
    indices = []
    for f in p.iterdir():
        if f.suffix.lower() in IMAGE_EXTS:
            idx = _parse_index_from_name(f.name)
            if idx is not None:
                indices.append(idx)
    indices.sort()
    return indices


def get_index_to_path(images_path: str | Path) -> dict[int, Path]:
    """
    Возвращает словарь {индекс_кадра: путь_к_файлу}.
    """
    p = Path(images_path)
    if not p.is_dir():
        raise FileNotFoundError(f"Папка не найдена: {images_path}")
    out = {}
    for f in p.iterdir():
        if f.suffix.lower() in IMAGE_EXTS:
            idx = _parse_index_from_name(f.name)
            if idx is not None:
                out[idx] = f
    return out


def get_annotated_indices(anno_path: str | Path) -> Set[int]:
    """
    Возвращает множество индексов изображений, для которых есть аннотации.
    Имена файлов масок должны совпадать с именами изображений.
    """
    p = Path(anno_path)
    if not p.is_dir():
        raise FileNotFoundError(f"Папка не найдена: {anno_path}")
    indices = set()
    for f in p.iterdir():
        if f.suffix.lower() in IMAGE_EXTS:
            idx = _parse_index_from_name(f.name)
            if idx is not None:
                indices.add(idx)
    return indices


# ----- Итерации разметки (можно редактировать/добавлять) -----

def iter_1_every_nth(step: int = 150, max_idx: int = 3107) -> Set[int]:
    """
    Итерация 0: каждая step-я фотография (0, 150, 300, 450, ...).
    """
    return set(range(0, max_idx + 1, step))


def iter_2_indices() -> Set[int]:
    """
    Итерация 1: заданный список изображений.
    Заполните множеством индексов, когда список будет готов.
    """
    # Пример: return {10, 20, 100, 200}
    # return set()
    return {64, 345, 416, 690, 980, 1024, 1107, 1276, 1441, 1591, 1874, 2018, 2151, 2325, 2485, 2623, 2786, 2878, 3046, 3106}


def iter_3_indices() -> Set[int]:
    """
    Итерация 2: заданный список изображений.
    Заполните множеством индексов, когда список будет готов.
    """
    # return set()
    return {388, 408, 841, 945, 1008, 1014, 1225, 1560, 1835, 1896, 2120, 2353, 2415, 2623, 2820, 3025}


def iter_4_indices() -> Set[int]:
    """
    Итерация 3: при необходимости добавьте новые итерации.
    """
    # return set()
    return {1899, 2294, 2511, 2802}


def get_all_iteration_indices(max_idx: int) -> dict[str, Set[int]]:
    """
    Собирает индексы по всем итерациям.
    Ключи: имена итераций, значения: множества индексов.
    """
    return {
        "iter_1 (каждые 150)": iter_1_every_nth(step=150, max_idx=max_idx),
        "iter_2": iter_2_indices(),
        "iter_3": iter_3_indices(),
        "iter_4": iter_4_indices(),
    }


# ----- Визуализация -----

def _sample_by_density(
    density: np.ndarray,
    total_images: int,
    n: int,
    offset: int = 30,
    window: int = 100,
    min_distance: int = 150,
) -> List[tuple[int, int, int]]:
    """
    Возвращает троек (idx_before, idx_peak, idx_after) для режима density.
    Сканирует скользящим окном: в каждом window ищет кадр с max density.
    Ограничение min_distance: на промежутке min_distance — только один стек изображений.
    """
    step = max(1, window // 2)
    candidates: dict[int, float] = {}
    for lo in range(0, total_images - 1, step):
        hi = min(lo + window, total_images)
        segment = density[lo:hi]
        local_max = lo + int(np.argmax(segment))
        if local_max not in candidates or density[local_max] > candidates[local_max]:
            candidates[local_max] = float(density[local_max])
    sorted_peaks = sorted(candidates.keys(), key=lambda i: -candidates[i])
    selected: List[int] = []
    for peak in sorted_peaks:
        if any(abs(peak - s) < min_distance for s in selected):
            continue
        selected.append(peak)
        if len(selected) >= n:
            break
    selected.sort()
    triplets = []
    for peak in selected:
        idx_before = max(0, peak - offset)
        idx_after = min(total_images - 1, peak + offset)
        triplets.append((idx_before, peak, idx_after))
    return triplets


def compute_density(
    total_images: int,
    annotated_indices: Set[int],
    window: int = 100,
) -> np.ndarray:
    """
    Плотность аннотаций: для каждого кадра — число аннотированных в окне [i-window, i+window].
    """
    print(sorted([i for i in annotated_indices]))
    density = np.zeros(total_images)
    for i in range(total_images):
        lo, hi = max(0, i - window), min(total_images - 1, i + window)
        density[i] = sum(1 for j in annotated_indices if lo <= j <= hi)
    if density.max() > 0:
        density = density / density.max()
    return density

def stats_info(annotated_indices: Set[int]) -> str:
    """
    Возвращает статистическую информацию о аннотациях.
    """
    list_annotated_indices = sorted([i for i in annotated_indices])
    list_of_space = []
    for idx, anno_idx in enumerate(list_annotated_indices):
        if idx + 1 < len(list_annotated_indices):
            list_of_space.append(list_annotated_indices[idx+1] - anno_idx)
        else:
            break

    string = f"Среднее расстояние между аннотациями: {sum(list_of_space) / len(list_of_space)}\n"
    print(list_of_space)
    return string


def build_annotation_strip(
    total_images: int,
    annotated_from_anno_folder: Set[int],
    iteration_indices: dict[str, Set[int]],
    output_path: str | Path,
    strip_height: int = 32,
    width_px: Optional[int] = None,
    show_legend: bool = True,
    density_window: int = 100,
    density_cmap: str = "YlOrRd",
    density_alpha: float = 0.55,
    show_density: bool = True,
    show_thumbnails: bool = False,
    index_to_path: Optional[dict[int, Path]] = None,
    thumbnail_count: int = 8,
    thumbnail_height: int = 80,
    thumbnail_mode: str = "uniform",
    thumbnail_density_offset: int = 30,
    thumbnail_density_window: Optional[int] = None,
    thumbnail_density_min_distance: int = 150,
) -> np.ndarray:
    """
    Строит полосу-визуализацию: каждый столбец — одно изображение.
    Фон: colormap плотности аннотаций. Поверх — маркеры итераций.

    Args:
        total_images: общее число изображений (0..total_images-1).
        annotated_from_anno_folder: индексы, для которых есть файлы в папке аннотаций.
        iteration_indices: { "iter_0": {0,150,...}, "iter_1": {...}, ... }.
        output_path: путь для сохранения PNG.
        strip_height: высота полосы в пикселях.
        width_px: ширина в пикселях; если None — 1 пиксель на изображение.
        density_window: окно для подсчёта плотности (в кадрах).
        density_cmap: имя colormap (YlOrRd, viridis, plasma и т.д.).
        density_alpha: прозрачность colormap (0–1): меньше — светлее фон, маркеры виднее.
        show_thumbnails: показать мини-превью изображений над полосой.
        index_to_path: словарь {индекс: путь} для загрузки кадров.
        thumbnail_count: число превью.
        thumbnail_height: высота превью в пикселях.
        thumbnail_mode: "uniform" — каждые N кадров; "density" — 3 уровня: max-30, max, max+30.
        thumbnail_density_offset: смещение в кадрах для уровней ± (только density).
        thumbnail_density_window: окно для поиска пиков (если None — density_window).
        thumbnail_density_min_distance: мин. дистанция между пиками, чтобы избежать наложения.
    """
    w = width_px if width_px is not None else total_images
    all_annotated = annotated_from_anno_folder.copy()
    for s in iteration_indices.values():
        all_annotated |= s

    print(stats_info(all_annotated))

    strip = np.ones((strip_height, w, 3), dtype=np.uint8)
    gray = np.array([180, 180, 180], dtype=np.uint8)

    need_density = show_density or (show_thumbnails and thumbnail_mode == "density")
    if need_density:
        density = compute_density(total_images, all_annotated, window=density_window)
        cmap_obj = _get_cmap(density_cmap)
        white = np.array([1.0, 1.0, 1.0])
        for x in range(w):
            img_idx = int(round(x * (total_images - 1) / (w - 1))) if w > 1 else 0
            if img_idx < total_images:
                c = np.array(cmap_obj(density[img_idx])[:3])
                blend_c = density_alpha * c + (1 - density_alpha) * white
                strip[:, x] = (blend_c * 255).astype(np.uint8)
    else:
        strip[:, :] = gray

    # Цвета для итераций (RGB)
    iter_colors = [
        [65, 105, 225],   # Royal Blue — iter_0
        [220, 20, 60],    # Crimson — iter_1
        [34, 139, 34],    # Forest Green — iter_2
        [255, 140, 0],    # Dark Orange — iter_3
    ]

    # Объединённое множество всех индексов из итераций
    all_iter = set()
    for s in iteration_indices.values():
        all_iter |= s

    # Заливка: приоритет — итерации по порядку, затем папка аннотаций.
    # width_px — целевая ширина; вся последовательность масштабируется в неё.
    for x in range(w):
        img_idx = int(round(x * (total_images - 1) / (w - 1))) if w > 1 else 0
        if img_idx >= total_images:
            continue
        color = None
        for i, (name, inds) in enumerate(iteration_indices.items()):
            if img_idx in inds and inds:
                color = iter_colors[i % len(iter_colors)]
                break
        if color is None and img_idx in annotated_from_anno_folder:
            color = [100, 149, 237]  # CornflowerBlue — только из папки
        if color is not None:
            bg = strip[0, x].astype(float)
            blend = 0.75 if show_density else 1.0
            strip[:, x] = (blend * np.array(color) + (1 - blend) * bg).astype(np.uint8)

    thumb_h = thumbnail_height if (show_thumbnails and index_to_path and cv2) else 0
    is_density_3levels = show_thumbnails and thumbnail_mode == "density" and need_density
    thumb_total_h = 3 * thumb_h if is_density_3levels else thumb_h
    fig_h = 5 if (show_thumbnails and is_density_3levels) else (4 if show_thumbnails else 3)
    fig = plt.figure(figsize=(max(10, w / 80), fig_h))
    ax = fig.add_subplot(111)

    ax.imshow(strip, extent=[0, w, strip_height, 0], aspect="auto", zorder=1)

    if show_thumbnails and index_to_path and cv2 is not None:
        if thumbnail_mode == "density" and need_density:
            # TODO: при близких пиках плотности превью накладываются по x;
            # варианты: минимальная дистанция между пиками, слияние соседних пиков,
            # или ограничение thumbnail_count по фактическому числу разделённых пиков
            dw = thumbnail_density_window if thumbnail_density_window is not None else density_window
            triplets = _sample_by_density(
                density,
                total_images,
                thumbnail_count,
                offset=thumbnail_density_offset,
                window=dw,
                min_distance=thumbnail_density_min_distance,
            )
            thumb_total_h = 3 * thumb_h
            for idx_before, idx_peak, idx_after in triplets:
                x_center = idx_peak * (w - 1) / max(1, total_images - 1)
                level_h = thumb_h
                for level, idx in enumerate([idx_before, idx_peak, idx_after]):
                    if idx not in index_to_path:
                        continue
                    img_bgr = cv2.imread(str(index_to_path[idx]))
                    if img_bgr is None:
                        continue
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    h_src, w_src = img_rgb.shape[:2]
                    thumb_w = int(level_h * w_src / max(1, h_src))
                    if thumb_w <= 0:
                        continue
                    img_small = cv2.resize(img_rgb, (thumb_w, level_h), interpolation=cv2.INTER_LINEAR)
                    img_small = np.flipud(img_small)
                    half_span = thumb_w / 2
                    y_bottom = -level_h * (3 - level)
                    y_top = -level_h * (2 - level)
                    ext = [x_center - half_span, x_center + half_span, y_bottom, y_top]
                    ax.imshow(img_small, extent=ext, aspect="equal", zorder=2, clip_on=False)
        else:
            sample_indices = [
                int(i * (total_images - 1) / max(1, thumbnail_count - 1))
                for i in range(thumbnail_count)
            ]
            sample_indices = list(dict.fromkeys(sample_indices))
            for idx in sample_indices:
                if idx not in index_to_path:
                    continue
                img_bgr = cv2.imread(str(index_to_path[idx]))
                if img_bgr is None:
                    continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                h_src, w_src = img_rgb.shape[:2]
                thumb_w = int(thumb_h * w_src / max(1, h_src))
                if thumb_w <= 0:
                    continue
                img_small = cv2.resize(img_rgb, (thumb_w, thumb_h), interpolation=cv2.INTER_LINEAR)
                img_small = np.flipud(img_small)
                x_center = idx * (w - 1) / max(1, total_images - 1)
                half_span = thumb_w / 2
                ext = [x_center - half_span, x_center + half_span, -thumb_h, 0]
                ax.imshow(img_small, extent=ext, aspect="equal", zorder=2, clip_on=False)

    # Метки оси X: 0, 150, 300, ... и последний кадр
    step = 150
    tick_indices = list(range(0, total_images, step))
    if tick_indices[-1] != total_images - 1:
        tick_indices.append(total_images - 1)
    tick_positions = [idx * (w - 1) / max(1, total_images - 1) for idx in tick_indices]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(i) for i in tick_indices])
    ax.set_xlim(0, w)
    ax.set_ylim(strip_height, -thumb_total_h if thumb_h else 0)
    ax.set_xlabel("Номер кадра")
    ax.set_yticks([])

    if show_legend:
        patches = []
        for i, (name, inds) in enumerate(iteration_indices.items()):
            if inds:
                patches.append(mpatches.Patch(color=np.array(iter_colors[i % len(iter_colors)]) / 255, label=f"{name} ({len(inds)})"))
        only_in_folder = annotated_from_anno_folder - all_iter
        if only_in_folder:
            patches.append(mpatches.Patch(color=np.array([100, 149, 237]) / 255, label=f"Только в папке ({len(only_in_folder)})"))
        ax.legend(handles=patches, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return strip


def main():
    parser = argparse.ArgumentParser(description="Визуализация полосы аннотаций")
    parser.add_argument("--images", type=str, default=DEFAULT_PATH_IMAGES, help="Папка с изображениями")
    parser.add_argument("--annotations", type=str, default=DEFAULT_PATH_ANNOTATIONS, help="Папка с аннотациями (SegmentationClass)")
    parser.add_argument("--output", type=str, default=DEFAULT_PATH_OUTPUT, help="Путь к выходному PNG")
    parser.add_argument("--height", type=int, default=64, help="Высота полосы в пикселях")
    parser.add_argument("--width", type=int, default=None, help="Целевая ширина в пикселях; вся полоса масштабируется (по умолчанию 1 px на кадр)")
    parser.add_argument("--density", action="store_true", default=True, help="Показать colormap плотности (по умолчанию: вкл)")
    parser.add_argument("--no-density", dest="density", action="store_false", help="Отключить colormap плотности")
    parser.add_argument("--density-window", type=int, default=75, help="Окно для подсчёта плотности (в кадрах)")
    parser.add_argument("--density-cmap", type=str, default="YlOrRd", help="Colormap для плотности (YlOrRd, viridis, plasma)")
    parser.add_argument("--density-alpha", type=float, default=0.4, help="Прозрачность colormap (0–1), меньше — светлее")
    parser.add_argument("--thumbnails", action="store_true", help="Показать мини-превью изображений над полосой")
    parser.add_argument("--no-thumbnails", dest="thumbnails", action="store_false", default=False, help="Без превью (по умолчанию)")
    parser.add_argument("--thumbnails-count", type=int, default=8, help="Число превью при --thumbnails")
    parser.add_argument("--thumbnails-height", type=int, default=80, help="Высота превью в пикселях")
    parser.add_argument("--thumbnails-mode", type=str, default="uniform", choices=["uniform", "density"],
        help="uniform: равномерно; density: 3 уровня (max-30, max, max+30)")
    parser.add_argument("--thumbnails-density-offset", type=int, default=30,
        help="Смещение ± кадров для уровней 1 и 3 в режиме density")
    parser.add_argument("--thumbnails-density-window", type=int, default=None,
        help="Окно для поиска пиков (по умолчанию = --density-window)")
    parser.add_argument("--thumbnails-density-min-distance", type=int, default=150,
        help="Мин. дистанция между пиками, чтобы не накладывались стеки")
    args = parser.parse_args()

    print("Читаю изображения...")
    img_indices = get_image_indices(args.images)
    num_files = len(img_indices)
    max_idx = max(img_indices) if img_indices else 0
    total_images = max_idx + 1  # полоса: 0..max_idx
    print(f"  Найдено изображений: {num_files} (индексы 0..{max_idx})")

    print("Читаю аннотации...")
    anno_indices = get_annotated_indices(args.annotations)
    print(f"  Аннотировано в папке: {len(anno_indices)}")

    iters = get_all_iteration_indices(max_idx)
    
    print("Итерации:")
    for name, inds in iters.items():
        if inds:
            # print(sorted([i for i in inds]))
            print(f"  {name}: {len(inds)} кадров")

    index_to_path = get_index_to_path(args.images) if args.thumbnails else None
    if args.thumbnails:
        if cv2 is None:
            print("  Предупреждение: cv2 не найден, превью отключены. Установите opencv-python.")
            args.thumbnails = False
        else:
            print(f"  Индексов для превью: {len(index_to_path)}")

    build_annotation_strip(
        total_images=total_images,
        annotated_from_anno_folder=anno_indices,
        iteration_indices={k: v for k, v in iters.items() if v},
        output_path=args.output,
        strip_height=args.height,
        width_px=args.width,
        density_window=args.density_window,
        density_cmap=args.density_cmap,
        density_alpha=args.density_alpha,
        show_density=args.density,
        show_thumbnails=args.thumbnails,
        index_to_path=index_to_path,
        thumbnail_count=args.thumbnails_count,
        thumbnail_height=args.thumbnails_height,
        thumbnail_mode=args.thumbnails_mode,
        thumbnail_density_offset=args.thumbnails_density_offset,
        thumbnail_density_window=args.thumbnails_density_window,
        thumbnail_density_min_distance=args.thumbnails_density_min_distance,
    )
    print(f"Сохранено: {args.output}")


if __name__ == "__main__":
    main()
