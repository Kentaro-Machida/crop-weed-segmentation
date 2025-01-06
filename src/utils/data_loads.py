import os
import cv2
import numpy as np

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
        reseized_width: int
        )-> np.ndarray:
    """
    Load RGB image from path and resize it.
    Return image as 3d np.ndarray.
    """
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (reseized_width, reseized_height))
    return img


def load_mask(
        path: str,
        reseized_height: int,
        reseized_width: int,
        task: str,
        )-> np.ndarray:
    """
    Load mask from path and resize it.
    Return mask as 3d np.ndarray.
    task: If "plant", return mask as Anything other than 0 is 1, and 0 is 0.
    If "crop", return mask as Anything other than 1 is 0, and 1 is 1.
    If "all", return mask as 3 classes mask.
    """
    mask = cv2.imread(path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    if task == "plant1d" or task == "plant2d" or task == "plant":
        mask = mask[:,:,0] != 0
        mask = mask[:,:,np.newaxis]
    elif task == "all":
        mask = mask[:,:,0]
        mask = mask[:,:,np.newaxis]
    elif task == "crop":
        mask = mask[:,:,0] == 1
        mask = mask[:,:,np.newaxis]
    else:
        raise ValueError(f"task: {task} is not supported")
    mask = mask.astype(np.uint8)
    mask = cv2.resize(mask, (reseized_width, reseized_height), interpolation=cv2.INTER_NEAREST)

    return mask
