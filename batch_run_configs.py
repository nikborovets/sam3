from sam3_batch_run import SAM3BatchProcessor

def fill_config(
    video_path: str,
    output_dir: str,
    output_video_name: str,
    prompts: list[str],
    show_box: bool,
    show_label: bool,
    show_mask: bool,
    show_original_image: bool,
    save_frames: bool
) -> dict:
    processor = SAM3BatchProcessor(video_path, output_dir)

    all_outputs = {}
    for prompt in prompts:
        all_outputs[prompt] = processor.process_prompt(prompt)

    merged = processor.merge_results(all_outputs)
    
    processor.visualize_merged(
        merged, 
        output_video_name=output_video_name,
        show_box=show_box,
        show_label=show_label,
        show_mask=show_mask,
        show_original_image=show_original_image,
        save_frames=save_frames
    )

    return merged

def run_config_1():
    prompts = [
        "chair", 
        "table", 
        "keyboard", 
        "touchpad",
        "mouse",
        "usb hub",
        "monitor",
        "imac",
        "wires",
        "cushion",
        "sofa",
        "bin",
        "screwdriver",
        "window",
        "window blind",
        "door",
        "door handle",
        "floor",
        "wall",
        "ceiling",
        "pipe",
        "socket",
        "plug",
        "switch",
        "column",
        "concrete",
    ]
    merged = fill_config(
        video_path="/workspace/ivan_images_slice",
        output_dir="/workspace/sam3_batch_results",
        output_video_name="merged_output_no_image.mp4",
        prompts=prompts,
        show_box=False,
        show_label=False,
        show_mask=True,
        show_original_image=False,
        save_frames=True)

    print("Done!")
    return merged

def run_config_2():
    # prompts = [
    #     "table", 
    #     "keyboard", 
    #     "touchpad",
    #     "mouse",
    #     "usb adapter",
    #     "wires",
    #     "cushion",
    #     "sofa",
    #     "window",
    #     "window blind",
    #     "door",
    #     "door handle",
    #     "floor",
    #     "wall",
    #     "ceiling",
    #     "socket",
    #     "plug",
    #     "switch",
    #     "armrest",
    #     "battery",
    #     "wardrobe",
    #     "cabinet wall",
    #     "baseboard",
    #     "glass",
    #     "book",
    #     "frame",
    #     "cardboard boxes",
    #     "shelves",
    #     "helmet",
    #     # "router",
    #     "box",
    #     "mug",
    #     "paper",
    #     "package",
    #     "cabinet door",
    #     "black thing",
    #     "trash",
    #     "blackboard",
    # ]
    prompts = [
        # Архитектура и фон (Room Structure)
        "wall",
        "ceiling",
        "floor",
        "baseboard",
        "window",
        "door",
        
        # Элементы на стенах (Fixtures)
        "socket",
        "switch",
        "light switch",
        "window blind",
        "door handle",
        "blackboard",
        "black plate",
        "paperboard",


        # Крупная мебель и ее части (Large Furniture)
        "wardrobe",
        "cabinet wall",
        "shelves",
        "cabinet door",
        "glass",
        "sofa",
        "armrest",
        "cushion",
        "table",

        # Предметы в шкафу/на полках (Objects on shelves/background)
        "cardboard boxes",
        "box",
        "package",
        "frame",
        "book",
        "helmet",
        "mug",
        "black thing",
        "router box",
        "inside of the cabinet",

        # Предметы на столе/переднем плане (Objects on desk/foreground)
        "computer keyboard",
        "touchpad",
        "mouse",
        "usb adapter",
        "wires",
        "plug",
        "battery",
        "paper",
    ]


    merged = fill_config(
        video_path="/workspace/ivan_images_slice2",
        output_dir="/workspace/sam3_batch_results_2",
        output_video_name="merged_output_with_image.mp4",
        prompts=prompts,
        show_box=False,
        show_label=False,
        show_mask=True,
        show_original_image=True,
        save_frames=True)

    print("Done!")
    return merged

def run_big_config():
    prompts = [
        # --- Архитектура и фон (Room Structure) ---
        "wall",
        "ceiling",
        "floor",
        "baseboard",
        "concrete",
        "column",
        "window",
        "door",
        "pipe",

        # --- Элементы на стенах (Fixtures) ---
        "socket",
        "switch",
        "light switch",
        "window blind",
        "door handle",
        "blackboard",
        "black plate",
        "paperboard",

        # --- Крупная мебель (Large Furniture) ---
        "wardrobe",
        "cabinet wall",
        "shelves",
        "cabinet door",
        "glass",
        "inside of the cabinet",
        "sofa",
        "armrest",
        "cushion",
        "chair",
        "table",
        "bin",

        # --- Предметы в шкафу/на полках (Objects on shelves/background) ---
        "cardboard boxes",
        "box",
        "router box",
        "package",
        "frame",
        "book",
        "helmet",
        "mug",
        "black thing",
        
        # --- Техника на столе ---
        "monitor",
        "imac",

        # --- Мелкие предметы на столе/переднем плане (Objects on desk/foreground) ---
        "keyboard",
        "touchpad",
        "mouse",
        "usb adapter",
        "wire",
        "plug",
        "battery",
        "paper",
        "screwdriver",
    ]
    merged = fill_config(
        # video_path="/workspace/edited_color_and_edge_sharpness_every_3_frame_new",
        video_path="/workspace/ivan_input_images",
        output_dir="/workspace/sam3_batch_results_all_input_images_0",
        output_video_name="merged_output_3023_all_input_no_image.mp4",
        prompts=prompts,
        show_box=False,
        show_label=False,
        show_mask=True,
        show_original_image=False,
        save_frames=True)
    print("Done!")
    return merged

def run_music_room_config():
    prompts = [
        # # --- Архитектура и фон (Room Structure) ---
        "wall",
        "ceiling",
        "floor",
        "carpet",
        "baseboard",
        "concrete",
        "column",
        "door",
        "pipe",
        "illumination",
        "radiator", # Радиатор отопления
        "ventilation shaft", # Вентиляционные шахты
        "ventilation pipe", # Вентиляционные трубы
        "mosaic on the wall", # Мозаика на стене
        "hexagon", # Шестиугольник
        "mosaic of hexagons",
        "black mosaic",
        "shadow",

        # --- Элементы на стенах (Fixtures) ---
        "socket",
        "switch",
        "light switch",
        "door handle",
        "blackboard",
        "paperboard",

        # --- Крупная мебель (Large Furniture) ---
        "wardrobe", # Шкаф
        "cabinet wall",
        "shelves",
        "closet",
        "cabinet door",
        "glass",
        "sofa",
        "armrest",
        "cushion",
        "chair",
        "table",
        "bin", # Мусорное ведро
        "trash can",
        "water cooler", # Кулер водяной
        "pouf", # Пуфики

        # --- Музыкальное оборудование и инструменты (Musical Equipment) ---
        "drums", # Барабаны
        "cymbal", # Тарелки барабанные
        "drumstick", # Палочки барабанные
        "guitar", # Гитара
        "ukulele", # Укулеле
        "piano", # Пианино
        "digital piano stand", # Стойка для пианино
        "accordion", # Баян или аккордеон
        "guitar amplifier", # Комбоусилитель
        "guitar pedal", # Гитарная педаль
        "mixing console", # Музыкальный пульт
        "microphone stand", # Стойки для микрофонов
        "tambourine", # Тамбурин
        "bongo drum", # Бонго (ударный инструмент)
        "guitar case", # Гитарный чехол
        "instrument case", # Чехлы для инструментов

        # --- Предметы в шкафу/на полках (Objects on shelves/background) ---
        "cardboard box",
        "box",
        "package",
        "frame",
        "book",
        "mug",
        "bottle", # Бутылка
        "bag", # Сумка
        "plastic cup", # Пластиковые стаканчики

        # --- Техника на столе ---
        "monitor",
        "imac",

        # --- Мелкие предметы на столе/переднем плане (Objects on desk/foreground) ---
        "keyboard",
        "touchpad",
        "usb adapter",
        "wire",
        "plug",
        "battery",
        "paper",
        "object on the table",
        "mixing console",
    ]
    # prompts = [
    #     # # --- Архитектура и фон (Room Structure) ---
    #     "wall",
    #     "ceiling",
    #     "floor",
    #     "carpet",
    #     "baseboard",
    #     "concrete",
    #     "column",
    #     "door",
    #     "pipe",
    #     "illumination",
    #     "radiator", # Радиатор отопления
    #     "mosaic on the wall", # Мозаика на стене
    #     # "hexagon", # Шестиугольник
    #     "mosaic of hexagons",
    #     "black mosaic",
    #     # "shadow",
    # ]

    # merged = fill_config(
    #     video_path="/workspace/music_room_input_3frames",
    #     output_dir="/workspace/sam3_batch_results_music_room",
    #     output_video_name="bb_label_image.mp4",
    #     prompts=prompts,
    #     show_box=True,
    #     show_label=True,
    #     show_mask=True,
    #     show_original_image=True,
    #     save_frames=True)
    merged = fill_config(
        video_path="/workspace/music_room_input_3frames",
        output_dir="/workspace/sam3_batch_results_music_room",
        output_video_name="merged_output_music_room_const_no_image_all6.mp4",
        prompts=prompts,
        show_box=False,
        show_label=False,
        show_mask=True,
        show_original_image=False,
        save_frames=True)
    print("Done!")
    return merged

def run_config_seq1_2711():
    prompts = [
        # --- Архитектура и фон (Room Structure) ---
        "wall",
        "ceiling",
        "floor",
        "baseboard",
        "concrete",
        "column",
        "window",
        "door",
        "pipe",
        "ventilation shaft", # Вентиляционные шахты
        "ventilation pipe", # Вентиляционные трубы
        "illumination",
        "air-conditioner",

        # --- Элементы на стенах (Fixtures) ---
        "socket",
        "switch",
        "light switch",
        "window blind",
        "door handle",
        "blackboard",
        "black plate",
        "paperboard",
        "router box",

        # --- Крупная мебель (Large Furniture) ---
        "desk wall",
        "wardrobe",
        "cabinet",
        "office cabinet",
        "cabinet wall",
        "shelves",
        "cabinet door",
        "glass",
        "inside of the cabinet",
        "sofa",
        "armrest",
        "cushion",
        "chair",
        "table",
        "bin",
        "PC",
        "computer",
        "skateboard",
        "backpack",
        "camping chair",

        # --- Предметы в шкафу/на полках (Objects on shelves/background) ---
        "quadcopter",
        "drone",
        "cardboard boxes",
        "box",
        "router box",
        "package",
        "frame",
        "book",
        "helmet",
        "mug",
        "bottle",
        "black thing",
        
        # --- Техника на столе ---
        "monitor",
        "imac",
        "laptop",

        # --- Мелкие предметы на столе/переднем плане (Objects on desk/foreground) ---
        "keyboard",
        "touchpad",
        "mouse",
        "usb adapter",
        "wire",
        "plug",
        "battery",
        "paper",
        "object on the table",
        "screwdriver",
    ]
    merged = fill_config(
        # video_path="/workspace/edited_color_and_edge_sharpness_every_3_frame_new",
        video_path="/workspace/seq1_2711_input_every_3_frame",
        output_dir="/workspace/sam3_batch_results_seq1_2711",
        output_video_name="merged_output_seq1_2711_no_image.mp4",
        prompts=prompts,
        show_box=False,
        show_label=False,
        show_mask=True,
        show_original_image=False,
        save_frames=True)
    print("Done!")
    return merged

if __name__ == "__main__":
    # run_config_1()
    # run_config_2()
    # run_big_config()
    # run_music_room_config()
    run_config_seq1_2711()