import os
import cv2
import numpy as np
from src.config import config

def get_image_path(
        target_dir:str,
        extensions=[".jpg", ".png", ".JPG", ".JPEG", ".PNG"]
        )-> list:
    """
    Get image path list from target directory.
    """
    path_list = []
    all_files = os.listdir(target_dir)
    for file_name in all_files:
        for extension in extensions:
            if file_name.endswith(extension):
                path_list.append(os.path.join(target_dir, file_name))
    return sorted(path_list)


def load_image(
        path: str,
        reseized_height: int,
        reseized_width: int,
        task: str,
        is_mask=False,
        )-> np.ndarray:
    """
    Load image from path and resize it.
    Return image as 3d np.ndarray.
    is_mask: If True, return image as mask.
    is_plant: If True, return image as plant mask(Background = 0, plant = 1).
    """
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if is_mask:
        if task == "plant1d" or task == "plant2d":
            img = img[:,:,0] != config["label"]["background"]
            img = img[:,:,np.newaxis]
        elif task == "all":
            img = img[:,:,0]
            img = img[:,:,np.newaxis]
        elif task == "crop":
            img = img[:,:,0] == config["label"]["crop"]
            img = img[:,:,np.newaxis]
        else:
            raise ValueError(f"task: {task} is not supported")
        img = img.astype(np.uint8)
        img = cv2.resize(img, (reseized_width, reseized_height), interpolation=cv2.INTER_NEAREST)
    else:
        img = cv2.resize(img, (reseized_width, reseized_height))
    return img
