import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from transformers import SegformerImageProcessor


from pkgs.data_classes.config_class import ConfigData
from pkgs.preproceses.patch_preprocess import PatchProcessor1d, PatchProcessor2d
from pkgs.preproceses.data_augmentation import DataTransformBuilder
from utils.data_loads import get_image_path, load_image


class DatasetFactory:
    """
    configファイルに書かれたtaskによって、データセットを作成
    data_rootにはtrain, val, testのフォルダが含まれている必要がある
    """
    def __init__(self, config: ConfigData):
        self.config = config

    def create_dataset(self) -> dict:
        """
        config file にしたがってデータセットを作成
        return {
            "train": Dataset,
            "val": Dataset,
            "test": Dataset
        }
        """
        if self.config.task == "plant1d":
            return self._dataset_to_dataloader(
                {
                    "train": Plant1dDataset(self.config, "train"),
                    "val": Plant1dDataset(self.config, "val"),
                    "test": Plant1dDataset(self.config, "test"),
                }
            )
        elif self.config.task == "plant2d":
            return self._dataset_to_dataloader(
                {
                    "train": Plant2dDataset(self.config, "train"),
                    "val": Plant2dDataset(self.config, "val"),
                    "test": Plant2dDataset(self.config, "test"),
                }
            )
        elif self.config.task == "crop":
            return self._dataset_to_dataloader(
                {
                     "train": WholeImageDataset(self.config, "train", "crop"),
                    "val": WholeImageDataset(self.config, "val", "crop"),
                    "test": WholeImageDataset(self.config, "test", "crop")
                }
            )
        elif self.config.task == "all":
            return self._dataset_to_dataloader(
                {
                    "train": WholeImageDataset(self.config, "train", "all"),
                    "val": WholeImageDataset(self.config, "val", "all"),
                    "test": WholeImageDataset(self.config, "test", "all")
                }
            )
        else:
            raise ValueError(
                "Invalid task. Look the config file and check a task.")

    def _dataset_to_dataloader(self, dataset_dict: dict) -> dict:
        """
        datasetをDataLoaderに変換
        """
        return {
            "train": DataLoader(dataset_dict["train"], batch_size=self.config.train_parameter.batch_size),
            "val": DataLoader(dataset_dict["val"], batch_size=self.config.train_parameter.batch_size),
            "test": DataLoader(dataset_dict["test"], batch_size=self.config.train_parameter.batch_size)
        }

class BaseDataset(Dataset):
    def __init__(self, config: ConfigData, target_sub_dir:str):
        """
        config file にしたがってデータセットを定義
        data_rootにはtrain, val, testのフォルダが含まれている必要がある
        config: config data of experiment
        target_sub_dir: train | val | test
        """
        assert target_sub_dir in ["train", "val", "test"], "You have to select train, val, or test"

        self.config = config
        self._target_dir = os.path.join(self.config.data_root, target_sub_dir)
        
        self._target_img_dir = os.path.join(self._target_dir, "images")
        self._target_mask_dir = os.path.join(self._target_dir, "masks")

        # 画像とマスクのパスリストを取得
        self.img_path_list = get_image_path(self._target_img_dir)
        self.mask_path_list = get_image_path(self._target_mask_dir)

        # Data Augmentation
        self.data_augmentator = DataTransformBuilder(config)

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


class Plant1dDataset(BaseDataset):
    """
    画像を1次元のパッチに分割し、それらを1つの入力とするデータセット
    """
    def __init__(self, config: ConfigData, target_sub_dir:str):
        super().__init__(config, target_sub_dir)
        self._img_patches = []
        self._mask_patches = []
        # Made patch preprocessors for image and mask
        self._img_patch_processor = PatchProcessor1d(
            patch_h=self.config.plant_model.patch_size,
            patch_w=self.config.plant_model.patch_size,
            image_height=self.config.resized_image_height,
            image_widht=self.config.resized_image_width,
            channels=3
        )
        self._mask_patch_processor = PatchProcessor1d(
            patch_h=self.config.plant_model.patch_size,
            patch_w=self.config.plant_model.patch_size,
            image_height=self.config.resized_image_height,
            image_widht=self.config.resized_image_width,
            channels=1
        )
        # Load images and masks and make patches
        for target_img_path, target_mask_path in zip(self.img_path_list, self.mask_path_list):
            img = load_image(target_img_path, self.config.resized_image_height, self.config.resized_image_width, task="plant1d")
            mask = load_image(target_mask_path, self.config.resized_image_height, self.config.resized_image_width, is_mask=True, task="plant1d")
            img_patches = self._img_patch_processor.image_to_patches(img)
            mask_patches = self._mask_patch_processor.image_to_patches(mask)
            self._img_patches.extend(img_patches)
            self._mask_patches.extend(mask_patches)

    def __len__(self):
        return len(self._img_patches)

    def __getitem__(self, idx):
        img_patch = self._img_patches[idx]
        mask_patch = self._mask_patches[idx]
        img_patch, mask_patch = self.data_augmentator(
            image=img_patch,mask = mask_patch
        )
        return img_patch, mask_patch


class Plant2dDataset(BaseDataset):
    """
    画像を2次元のパッチに分割し、それらを1つの入力とするデータセット
    """
    def __init__(self, config: ConfigData, target_sub_dir:str):
        super().__init__(config, target_sub_dir)
        self._img_patches = []
        self._mask_patches = []
        # Made patch preprocessors for image and mask
        self._img_patch_processor = PatchProcessor2d(
            patch_h=self.config.plant_model.patch_size,
            patch_w=self.config.plant_model.patch_size,
            image_height=self.config.resized_image_height,
            image_widht=self.config.resized_image_width,
            channels=3
        )
        self._mask_patch_processor = PatchProcessor2d(
            patch_h=self.config.plant_model.patch_size,
            patch_w=self.config.plant_model.patch_size,
            image_height=self.config.resized_image_height,
            image_widht=self.config.resized_image_width,
            channels=1
        )
        # Load images and masks and make patches
        for target_img_path, target_mask_path in zip(self.img_path_list, self.mask_path_list):
            img = load_image(target_img_path, self.config.resized_image_height, self.config.resized_image_width, task="plant2d")
            mask = load_image(target_mask_path, self.config.resized_image_height, self.config.resized_image_width, is_mask=True, task="plant2d")
            img_patches = self._img_patch_processor.image_to_patches(img)
            mask_patches = self._mask_patch_processor.image_to_patches(mask)
            self._img_patches.extend(img_patches)
            self._mask_patches.extend(mask_patches)

    def __len__(self):
        return len(self._img_patches)

    def __getitem__(self, idx):
        img_patch = self._img_patches[idx]
        mask_patch = self._mask_patches[idx]
        img_patch, mask_patch = self.data_augmentator(
            image=img_patch,mask = mask_patch
        )
        return img_patch, mask_patch


class WholeImageDataset(BaseDataset):
    """
    画像全体を入力とするデータセット
    config: 実験用のconfig data
    target_sub_dir: train | val | test のいずれかの文字列
    task: all | crop のいずれかの文字列, allの場合は3クラス, cropの場合は2クラス
    """
    def __init__(self, config: ConfigData, target_sub_dir:str, task:str):
        super().__init__(config, target_sub_dir)
        if self.config.model_type == "segformer":
            self.preprocessor = SegformerImageProcessor.from_pretrained(config.segformer_pretrained_model)
            self.task = task

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        img = load_image(
            self.img_path_list[idx],
            self.config.resized_image_height,
            self.config.resized_image_width,
            task=self.task)
        mask = load_image(
            self.mask_path_list[idx],
            self.config.resized_image_height,
            self.config.resized_image_width,
            is_mask=True,
            task=self.task)
        augmentated = self.data_augmentator(
            image=img, mask=mask
        )
        img = augmentated["image"]
        mask = augmentated["mask"]
        if self.config.model_type == "segformer":
            SegFormer_input = self.preprocessor(img, return_tensors="pt")
            for k,v in SegFormer_input.items():
                SegFormer_input[k].squeeze_()
            mask = mask.astype(np.uint8)

            mask_tensor = torch.from_numpy(mask)
            mask_tensor = mask_tensor.long()
            SegFormer_input['labels'] = mask_tensor
            return SegFormer_input
        return img, mask


if __name__ == '__main__':
    from config import config
    config_data = ConfigData.load_data(config)
    plant_train = Plant1dDataset(config_data, "train")
    plant_val = Plant1dDataset(config_data, "val")
    plant_test = Plant1dDataset(config_data, "test")
    if config_data.task == "plant1d":
        print("----- Test for Plant1dDataset -----")
        print(f"get_item: {plant_train.__getitem__(0)}")
        print(f"length: {plant_train.__len__()}")
        print(f"input image shape: {plant_train.__getitem__(0)[0].shape}")
        print(f"mask image shape: {plant_train.__getitem__(0)[1].shape}")
        print(f"unique mask: {np.unique(plant_train.__getitem__(0)[1])}")

    if config_data.task == "plant2d":
        plant_train = Plant2dDataset(config_data, "train")
        print("----- Test for Plant2dDataset -----")
        print(f"length: {plant_train.__len__()}")
        print(f"input image shape: {plant_train.__getitem__(0)[0].shape}")
        print(f"mask image shape: {plant_train.__getitem__(0)[1].shape}")
        print(f"unique mask: {np.unique(plant_train.__getitem__(0)[1])}")

    if config_data.task == "crop":
        crop_train = WholeImageDataset(config_data, "train", "crop")
        print("----- Test for CropDataset -----")
        print(f"length: {crop_train.__len__()}")
        print(f"input image shape: {crop_train.__getitem__(0)}")
        print(f"mask image shape: {crop_train.__getitem__(0)}")
        print(f"unique mask: {np.unique(crop_train.__getitem__(0))}")

    if config_data.task == "all":
        all_train = WholeImageDataset(config_data, "train", "all")
        print("----- Test for AllDataset -----")
        print(f"length: {all_train.__len__()}")
        print(f"input image shape: {all_train.__getitem__(0)}")
        print(f"mask image shape: {all_train.__getitem__(0)}")
        print(f"unique mask: {np.unique(all_train.__getitem__(0))}") 

    dataset_factory = DatasetFactory(config_data)
    dataset = dataset_factory.create_dataset()
    print("----- Test for DatasetFactory -----")
    print(f"dataset: {dataset}")
