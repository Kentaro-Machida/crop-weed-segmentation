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
        reseized_width: int,
        is_mask=False,
        is_plant=False,
        )-> np.ndarray:
    """
    Load image from path and resize it.
    Return image as 3d np.ndarray.
    is_mask: If True, return image as mask.
    is_plant: If True, return image as plant mask.
    """
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if is_mask:
        if is_plant:
            img = img[:,:,0] != 0
            img = img.astype(np.uint8)
            img = img[:,:,np.newaxis]
        else:
            img = img[:,:,0]
            img = img[:,:,np.newaxis]
        img = cv2.resize(img, (reseized_width, reseized_height), interpolation=cv2.INTER_NEAREST)
    else:
        img = cv2.resize(img, (reseized_width, reseized_height))
    return img
