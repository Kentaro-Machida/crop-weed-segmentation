import os 
import torch
import numpy as np
import segmentation_models_pytorch as smp
from lightning.pytorch import LightningModule
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import BinaryJaccardIndex
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

from src.utils.data_loads import get_image_path, load_image, load_mask
from src.pkgs.preproceses.data_augmentation import DataTransformBuilder, get_normalizer
from src.utils import check_no_overlap
from src.utils.roboflow_image_mask_split import sort_and_check_consistncy
from ..data_classes.config_class import ModelDatasetConfig
from .base_model_dataset import BaseModelDataset, BaseDataset, BaseLightningModule
from src.pkgs.data_classes.config_class import ModelDatasetConfig


class TransformerModelDataset(BaseModelDataset):
    """
    Pytorch Lightning で動作するTransformerモデルとデータセットを用意するクラス
    """
    def __init__(self, config: ModelDatasetConfig, data_root_path: str, load_mask_func: callable, label_dict:dict, data_augmentator):
        super().__init__(config)

        self.model = TransformerLightning(config, label_dict=label_dict)
        self._pretrained_model = config.transformer_setting.pretrained_model
        self._preprocessor = SegformerImageProcessor.from_pretrained(
            self._pretrained_model
        )
        # 以下はModelDataset系統クラスにおいて共通の処理
        self.load_mask_func = load_mask_func
        self.label_dict = label_dict

        self.image_height = config.image_height
        self.image_width = config.image_width

        self.batch_size = config.batch_size
        self.num_classes = len(label_dict)

        # 画像とマスクのパスを取得しソート
        self._train_img_paths, self._train_mask_paths = self._get_image_mask_paths(data_root_path, "train")
        self._val_img_paths, self._val_mask_paths = self._get_image_mask_paths(data_root_path, "val")
        self._test_img_paths, self._test_mask_paths = self._get_image_mask_paths(data_root_path, "test")

        # データをソートし、パスの整合性を確認
        self._train_img_paths, self._train_mask_paths = sort_and_check_consistncy(self._train_img_paths, self._train_mask_paths)
        self._val_img_paths, self._val_mask_paths = sort_and_check_consistncy(self._val_img_paths, self._val_mask_paths)
        self._test_img_paths, self._test_mask_paths = sort_and_check_consistncy(self._test_img_paths, self._test_mask_paths)

        # データリークの確認
        if not check_no_overlap(self._train_img_paths, self._val_img_paths, self._test_img_paths):
            raise ValueError(f"Data leak detected. Please check the dataset in {data_root_path}.")

        # データセットを作成
        normalizer = get_normalizer()
        self._train_dataset = self._get_dataset(self._train_img_paths, self._train_mask_paths, data_augmentator)
        self._val_dataset = self._get_dataset(self._val_img_paths, self._val_mask_paths)
        self._test_dataset = self._get_dataset(self._test_img_paths, self._test_mask_paths)

    def get_model_datasets(self)->dict:
        return {
            "model": self.model,
            "train_loader": DataLoader(self._train_dataset, self.batch_size, num_workers=2),
            "val_loader": DataLoader(self._val_dataset, self.batch_size, num_workers=2),
            "test_loader": DataLoader(self._test_dataset, self.batch_size, num_workers=2)
        }
    
    def _get_dataset(self, img_paths: list, mask_paths: list, data_augmentator=None):
        return SegformerDataset(
            img_path_list=img_paths,
            mask_path_list=mask_paths,
            preprocessor=self._preprocessor,
            img_height=self.image_height,
            img_width=self.image_width,
            data_augmentator=data_augmentator,
            num_classes=self.num_classes,
            load_mask_func=self.load_mask_func
        )
    
    def _get_image_mask_paths(self, data_root_path:str, phase:str) -> tuple:
        img_dir = os.path.join(data_root_path, phase, "images")
        mask_dir = os.path.join(data_root_path, phase, "masks")
        img_paths = get_image_path(img_dir)
        mask_paths = get_image_path(mask_dir)
        return img_paths, mask_paths

    def get_dataset(self, phase: str):
        if phase == "train":
            return self._train_dataset
        elif phase == "val":
            return self._val_dataset
        elif phase == "test":
            return self._test_dataset
        else:
            raise ValueError(f"phase: {phase} is not supported.")
    
    def get_model(self):
        return self.model
    
    def get_image_mask_paths(self, phase: str) -> tuple:
        """
        Return (image_paths, mask_paths)
        """
        if phase == "train":
            return self._train_img_paths, self._train_mask_paths
        elif phase == "val":
            return self._val_img_paths, self._val_mask_paths
        elif phase == "test":
            return self._test_img_paths, self._test_mask_paths
        else:
            raise ValueError(f"phase: {phase} is not supported.")

    
class SegformerDataset(BaseDataset):
    def __init__(
            self,
            img_path_list:list,
            mask_path_list:list,
            preprocessor,
            img_height:int,
            img_width:int,
            num_classes:int,
            load_mask_func:callable,
            data_augmentator=None,
            ):
        
        self.img_path_list = img_path_list
        self.mask_path_list = mask_path_list
        self.preprocessor = preprocessor
        self.img_height = img_height
        self.img_width = img_width
        self._data_augmentator = data_augmentator
        self.num_classes = num_classes
        self.load_mask_func = load_mask_func

    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self, idx):
        img = load_image(
            self.img_path_list[idx],
            self.img_height,
            self.img_width
            )
        mask = self.load_mask_func(
            self.mask_path_list[idx],
            self.img_height,
            self.img_width
            )

        if self._data_augmentator:
            augmented = self._data_augmentator(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        SegFormer_input = self.preprocessor(img, return_tensors="pt")
        for k,v in SegFormer_input.items():
            SegFormer_input[k].squeeze_()
        mask = mask.astype(np.uint8)

        mask_tensor = torch.from_numpy(mask)
        mask_tensor = mask_tensor.long()
        SegFormer_input['labels'] = mask_tensor

        return SegFormer_input


class TransformerLightning(BaseLightningModule):
    def __init__(self, config: ModelDatasetConfig, label_dict:dict):
        super(TransformerLightning, self).__init__()
        self.config = config
        self.num_classes = len(label_dict)
        self.label_dict = label_dict

        if config.criterion == "cross_entropy":
            if self.num_classes > 2:
                self.criterion = torch.nn.CrossEntropyLoss()
            else:
                self.criterion = torch.nn.BCEWithLogitsLoss()
            self.loss_func = self._get_cross_entropy_loss
        elif config.criterion == "dice":
            self.criterion = smp.losses.DiceLoss(mode='binary' if self.num_classes == 2 else 'multiclass', from_logits=True)
            self.loss_func = self._get_dice_loss

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
        print(outputs)
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
    
    def test_step(self, batch, batch_idx):
        inputs = batch["pixel_values"]
        targets = batch["labels"]
        outputs = self.model(pixel_values=inputs, labels=targets)
        loss = outputs.loss
        self.log("test_loss", loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        inputs = batch["pixel_values"]
        targets = batch["labels"]
        outputs = self.model(pixel_values=inputs, labels=targets)
        return outputs.logits
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        return optimizer
    
    def _get_cross_entropy_loss(self, y_hat, y):
        """
        y_hat(logit): torch.Size([N, num_classes, H, W], dtype=torch.float)
        y: torch.Size([N, num_classes, H, W], dtype=torch.float)
        """
        y_hat = y_hat.to(torch.float)
        y = y.to(torch.float)
        return self.criterion(y_hat, y)
    
    def _get_dice_loss(self, y_hat, y):
        """
        y_hat(logit): torch.Size([N, num_classes, H, W], dtype=torch.float)
        y: torch.Size([N, num_classes, H, W], dtype=int)
        """
        y_hat = y_hat.to(torch.float)
        y = y.to(torch.long)
    
        return self.criterion(y_hat, y)


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
