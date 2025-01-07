import os 
import torch
import numpy as np
from lightning.pytorch import LightningModule
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import BinaryJaccardIndex

from src.utils.data_loads import get_image_path, load_image, load_mask
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

        self.model = TransformerLightning(config)
        self._pretrained_model = config.transformer_setting.pretrained_model
        self._preprocessor = SegformerImageProcessor.from_pretrained(
            self._pretrained_model
        )
        # 以下はModelDataset系統クラスにおいて共通の処理
        self._data_augmentator = DataTransformBuilder(config.data_augmentation_config)

        self.image_height = config.image_height
        self.image_width = config.image_width

        self.batch_size = config.batch_size

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
            "model": self.model,
            "train_loader": DataLoader(train_dataset, self.batch_size),
            "val_loader": DataLoader(val_dataset, self.batch_size),
            "test_loader": DataLoader(test_dataset, self.batch_size)
        }
    

class SegformerDataset(Dataset):
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
            self.img_width
            )
        mask = load_mask(
            self.mask_path_list[idx],
            self.img_height,
            self.img_width,
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


class TransformerLightning(LightningModule):
    def __init__(self, config: ModelDatasetConfig):
        super(TransformerLightning, self).__init__()
        self.config = config
        self.num_classes = config.num_classes

        if config.criterion == "cross_entropy":
            self.criterion = torch.nn.CrossEntropyLoss()

        if config.metric == "binary_jaccard":
            self.metric = BinaryJaccardIndex(threshold=0.5)

        self.lr = config.lr

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name_or_path=config.transformer_setting.pretrained_model,
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs = batch["pixel_values"]
        targets = batch["labels"]
        outputs = self.model(pixel_values=inputs, labels=targets)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch["pixel_values"]
        targets = batch["labels"]
        outputs = self.model(pixel_values=inputs, labels=targets)
        loss = outputs.loss
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    import yaml
    yaml_config_path = "/home/machida/Documents/WeedSegmentation_2024/crop-weed-segmentation/config.yml"
    with open(yaml_config_path) as f:
        config = yaml.safe_load(f)
    
    data_root_path = config["experiment"]["data_root_path"]
    modeldataset_config = ModelDatasetConfig(**config["experiment"]["modeldataset_config"])
    transformer_model_dataset = TransformerModelDataset(modeldataset_config, data_root_path)
    model_datasets = transformer_model_dataset.get_model_datasets()
    model = model_datasets["model"]
    train_loader = model_datasets["train_loader"]
    val_loader = model_datasets["val_loader"]
    test_loader = model_datasets["test_loader"]
