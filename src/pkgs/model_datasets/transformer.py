import os 
import torch
import numpy as np

from src.utils.data_loads import get_image_path, load_image
from src.pkgs.preproceses.data_augmentation import DataTransformBuilder
from ..data_classes.config_class import ModelDatasetConfig
from .base_model_dataset import BaseModelDataset
from src.pkgs.data_classes.config_class import ModelDatasetConfig
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

class TransformerModelDataset(BaseModelDataset):
    """
    Pytorch Lightning で動作するTransformerモデルとデータセットを用意するクラス
    """
    def __init__(self, config: ModelDatasetConfig, data_root_path: str):
        super().__init__(config)

        self._pretrained_model = config.transformer_setting.pretrained_model
        self._preprocessor = SegformerImageProcessor.from_pretrained(
            self._pretrained_model
        )
        self._data_augmentator = DataTransformBuilder(config.data_augmentation_config)

        self.image_height = config.image_height
        self.image_width = config.image_width

        train_img_dir = os.path.join(data_root_path, "train", "images")
        train_mask_dir = os.path.join(data_root_path, "test", "masks")
        self._train_img_paths = get_image_path(train_img_dir)
        self._train_mask_paths = get_image_path(train_mask_dir)

        val_img_dir = os.path.join(data_root_path, "val", "images")
        val_mask_dir = os.path.join(data_root_path, "val", "masks")
        self._val_img_paths = get_image_path(val_img_dir)
        self._val_mask_paths = get_image_path(val_mask_dir)

        self.test_img_dir = os.path.join(data_root_path, "test", "images")
        self.test_mask_dir = os.path.join(data_root_path, "test", "masks")
        self._test_img_paths = get_image_path(self.test_img_dir)
        self._test_mask_paths = get_image_path(self.test_mask_dir)


    def get_model_datasets(self)->dict:
        model = SegformerForSemanticSegmentation.from_pretrained(
            self._pretrained_model
        )

        train_dataset = SegformerDataset(
            self._train_img_paths,
            self._train_mask_paths,
            self._preprocessor,
            self.image_height,
            self.image_width,
            self._data_augmentator
        )

        val_dataset = SegformerDataset(
            self._val_img_paths,
            self._val_mask_paths,
            self._preprocessor,
            self.image_height,
            self.image_width,
            self._data_augmentator
        )

        test_dataset = SegformerDataset(
            self._test_img_paths,
            self._test_mask_paths,
            self._preprocessor,
            self.image_height,
            self.image_width,
            self._data_augmentator
        )
        
        return {
            "model": model,
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "test_dataset": test_dataset
        }
    

class SegformerDataset:
    def __init__(
            self,
            img_path_list:list,
            mask_path_list:list,
            preprocessor,
            img_height:int,
            img_width:int,
            data_augmentator=None
            ):
        self.img_path_list = img_path_list
        self.mask_path_list = mask_path_list
        self.preprocessor = preprocessor
        self.img_height = img_height
        self.img_width = img_width
        self._data_augmentator = data_augmentator

    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self, idx):
        img = load_image(
            self.img_path_list[idx],
            self.img_height,
            self.img_width,
            task="all")
        mask = load_image(
            self.mask_path_list[idx],
            self.img_height,
            self.img_width,
            is_mask=True,
            task="all")
        augmentated = self._data_augmentator(
            image=img, mask=mask
        )
        img = augmentated["image"]
        mask = augmentated["mask"]
        SegFormer_input = self.preprocessor(img, return_tensors="pt")
        for k,v in SegFormer_input.items():
            SegFormer_input[k].squeeze_()
        mask = mask.astype(np.uint8)

        mask_tensor = torch.from_numpy(mask)
        mask_tensor = mask_tensor.long()
        SegFormer_input['labels'] = mask_tensor

        return SegFormer_input
