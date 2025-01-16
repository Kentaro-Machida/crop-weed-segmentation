import os
import cv2
import numpy as np


class LabelConverter:
    def __init__(self):
        self._task_label_dict = {
        "3_classes": {'background': 0, 'crop': 1, 'weed': 2},
        "plant": {'background': 0, 'plant': 1},
        "crop": {'background': 0, 'crop': 1},
        "5_classes": {'background': 0, 'crop': 1, 'weed1_broad': 2, 'weed2_cyperaceae': 3, 'weed3_aquatic': 4}
    }
        
    def task_to_label_dict(self, task: str)->dict:
        """
        このリポジトリ全体で使用されるtaskとlabel_dictの対応を返す関数
        Args:
            task (str): crop | plant | all
        Returns:
            dict: 画像の整数とラベルの対応
        """
        task_label_dict = {
            "3_classes": {'background': 0, 'crop': 1, 'weed': 2},
            "plant": {'background': 0, 'plant': 1},
            "crop": {'background': 0, 'crop': 1},
            "4_classes": {'background': 0, 'crop': 1, 'weed1_broad': [2,4], 'weed2_cyperaceae': 3},
            "5_classes": {'background': 0, 'crop': 1, 'weed1_broad': 2, 'weed2_cyperaceae': 3, 'weed3_aquatic': 4}
        }
        if task == "3_classes":
            return task_label_dict["3_classes"]
        elif task == "plant":
            return task_label_dict["plant"]
        elif task == "crop":
            return task_label_dict["crop"]
        elif task == "4_classes":
            return task_label_dict["4_classes"]
        elif task == "5_classes":
            return task_label_dict["5_classes"]
        else:
            raise ValueError(f"task: {task} is not supported.")
        
    def get_task_label_dict(self)->dict:
        return self._task_label_dict


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
    If "3_classes", return mask as 3 classes (0, 1, the others) mask.
    If "5_classes", return mask as 5 classes
    """
    mask = cv2.imread(path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    label_converter = LabelConverter()
    task_label_dict = label_converter.get_task_label_dict()
    label_dict = task_label_dict[task]

    if task == "plant":
        # backgroundクラス以外を1に、backgroundクラスを0にする
        background_label = label_dict['background']

        mask = mask[:,:,0] != background_label
        mask = mask[:,:,np.newaxis]

    elif task == "3_classes":
        # backgroundクラスを0に、cropクラスを1に、それ以外をweedクラスにする
        background_label = label_dict['background']
        mask = mask[:,:,0]
        mask[mask == background_label] = 0
        crop_label = label_dict['crop']

        mask[mask == crop_label] = 1
        mask[mask >= 2] = 2
        mask = mask[:,:,np.newaxis]

    elif task == "crop":
        # cropクラスを1に、それ以外を0にする
        crop_label = label_dict['crop']

        mask = mask[:,:,0] == crop_label
        mask = mask[:,:,np.newaxis]

    elif task == "4_classes":
        # weed3_aquaticクラスをweed1_broadクラスに統合
        background_label = label_dict['background']
        crop_label = label_dict['crop']
        weed1_label = label_dict['weed1_broad'][0]
        weed2_label = label_dict['weed2_cyperaceae']
        weed3_label = label_dict['weed1_broad'][1]

        mask = mask[:,:,0]
        mask[mask == background_label] = 0
        mask[mask == crop_label] = 1
        mask[mask == weed1_label] = 2
        mask[mask == weed3_label] = 2
        mask[mask == weed2_label] = 3
        mask = mask[:,:,np.newaxis]

    elif task == "5_classes":
        # 全てのクラスをそのまま使用
        background_label = label_dict['background']
        crop_label = label_dict['crop']
        weed1_label = label_dict['weed1_broad']
        weed2_label = label_dict['weed2_cyperaceae']
        weed3_label = label_dict['weed3_aquatic']

        mask = mask[:,:,0]
        mask[mask == background_label] = 0
        mask[mask == crop_label] = 1
        mask[mask == weed1_label] = 2
        mask[mask == weed2_label] = 3
        mask[mask == weed3_label] = 4
        mask = mask[:,:,np.newaxis]
        
    else:
        raise ValueError(f"task: {task} is not supported")
    mask = mask.astype(np.uint8)
    mask = cv2.resize(mask, (reseized_width, reseized_height), interpolation=cv2.INTER_NEAREST)

    return mask
