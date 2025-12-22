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
        "bin",  # Обычно стоит на полу, ближе к мебели

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
        
        # --- Техника на столе (отсутствующая на фото, но была в списке) ---
        "monitor",
        "imac",

        # --- Мелкие предметы на столе/переднем плане (Objects on desk/foreground) ---
        "keyboard",          # Объединено с computer keyboard
        "touchpad",
        "mouse",
        "usb adapter",       # Объединено с usb hub
        "wires",
        "plug",
        "battery",
        "paper",
        "screwdriver",
    ]
    merged = fill_config(
        video_path="/workspace/ivan_input_images",
        output_dir="/workspace/sam3_batch_results_all_input_images_0",
        output_video_name="merged_output_all_input_images_with_image.mp4",
        prompts=prompts,
        show_box=False,
        show_label=False,
        show_mask=True,
        show_original_image=True,
        save_frames=True)
    print("Done!")
    return merged


if __name__ == "__main__":
    # run_config_1()
    # run_config_2()
    run_big_config()