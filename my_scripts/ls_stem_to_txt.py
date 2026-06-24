from pathlib import Path


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

if __name__ == "__main__":
    save_stems_to_txt(Path("/workspace/seq4img"), Path("/workspace/seq4img_stems.txt"))