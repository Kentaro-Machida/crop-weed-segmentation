import os 
import pickle
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import BinaryJaccardIndex
import segmentation_models_pytorch as smp

from src.utils.data_loads import get_image_path, load_image, load_mask
from src.pkgs.preproceses.data_augmentation import DataTransformBuilder
from ..data_classes.config_class import ModelDatasetConfig
from .base_model_dataset import BaseModelDataset
from src.pkgs.data_classes.config_class import ModelDatasetConfig


def image_to_2d_patches(image, patch_h, patch_w):
    """
    Convert image to some patches.
    input: image(2d or 3d np.ndarray)の画像, patch size
    output: list of 1d ndarray
    """
    if len(image.shape) == 2:
        image = image[:, :, None]
    h, w, _ = image.shape

    # パッチに分割
    patches = [image[i:i+patch_h, j:j+patch_w]
               for i in range(0, h, patch_h) 
               for j in range(0, w, patch_w)]

    return patches


def patches_2d_to_image(patches, h, w, patch_h=5, patch_w=5, channels=1):
    """
    Reconstruct patches as an image.
    input: list of 2d ndarray, image size
    output: 2d nparray
    """
    image_2d = np.zeros((h, w, channels))
    h_patch_num = h / patch_h
    w_patch_num = w / patch_w
    for i_h in range(0, h, patch_h):
        for i_w in range(0, w, patch_w):
            patch_num = int(i_h / patch_h * w_patch_num + i_w / patch_w)
            patch = patches[patch_num]
            image_2d[i_h:i_h+patch_h, i_w:i_w+patch_w] = patch 
    
    return image_2d


class Patch2dModelDataset(BaseModelDataset):
    def __init__(self, config: ModelDatasetConfig, data_root_path: str):
        super().__init__(config)
        self.model = UNetppLightning(config)
        self.data_root_path = data_root_path
        self.patch_height = config.patch2d_setting.patch_size
        self.patch_width = config.patch2d_setting.patch_size
        
        # 以下はModelDataset系統クラスにおいて共通の処理
        self._data_augmentator = DataTransformBuilder(config.data_augmentation_config)

        self.image_height = config.image_height
        self.image_width = config.image_width

        self.batch_size = config.batch_size
        self.num_classes = config.num_classes

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

    def get_model_datasets(self):
        
        train_dataset = self._get_dataset(self._train_img_paths, self._train_mask_paths)
        val_dataset = self._get_dataset(self._val_img_paths, self._val_mask_paths)
        test_dataset = self._get_dataset(self._test_img_paths, self._test_mask_paths)

        return {
            "model": self.model,
            "train_loader": DataLoader(train_dataset, self.batch_size),
            "val_loader": DataLoader(val_dataset, self.batch_size),
            "test_loader": DataLoader(test_dataset, self.batch_size)
        }

    def _get_dataset(self, img_paths, mask_paths):
        return Patch2dDataset(
            img_path_list=img_paths,
            mask_path_list=mask_paths,
            img_height=self.image_height,
            img_width=self.image_width,
            patch_height=self.patch_height,
            patch_width=self.patch_width,
            data_root_path=self.data_root_path,
            data_augmentator=self._data_augmentator,
            num_classes=self.num_classes
        )


class UNetppLightning(pl.LightningModule):
    def __init__(self, config: ModelDatasetConfig):
        super(UNetppLightning, self).__init__()
        self.iou = BinaryJaccardIndex()
        
        self.model = smp.UnetPlusPlus(
            encoder_name=config.patch2d_setting.backborn,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
            classes=config.num_classes,                      # model output channels (number of classes in your dataset)
        )

        if config.criterion == "cross_entropy":
            if config.num_classes > 2:
                self.criterion = torch.nn.CrossEntropyLoss()
            else:
                self.criterion = torch.nn.BCEWithLogitsLoss()
        elif config.criterion == "dice":
            self.criterion = smp.losses.DiceLoss(mode='binary' if config.num_classes == 2 else 'multiclass')

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss_{}'.format(self.criterion), loss, on_epoch=True)

        y_hat_class = (torch.sigmoid(y_hat) > 0.5).float()  # Sigmoid activation function is added here
        iou_score = self.iou(y_hat_class, y)
        self.log('train_IoU', iou_score, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss_{}'.format(self.criterion), loss, on_epoch=True)

        y_hat_class = (torch.sigmoid(y_hat) > 0.5).float()  # Sigmoid activation function is added here
        iou_score = self.iou(y_hat_class, y)
        self.log('val_IoU', iou_score, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-5)  # weight decay added
        return optimizer


class Patch2dDataset(Dataset):
    def __init__(
        self,
        img_path_list: list,
        mask_path_list: list,
        img_height: int,
        img_width: int,
        patch_height: int,
        patch_width: int,
        data_root_path: str,
        data_augmentator=None,
        num_classes=2
    ):
        self.data_augmentator = data_augmentator
        self.num_classes = num_classes

        # キャッシュディレクトリ設定
        self.cache_dir = os.path.join(data_root_path, "_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.img_pkl_list, self.mask_pkl_list = self._create_patches(
            img_path_list, mask_path_list, img_height, img_width, patch_height, patch_width
        )

    def _create_patches(self, img_paths, mask_paths, img_h, img_w, patch_h, patch_w):
        """
        入力画像とマスクをパッチに分割し、キャッシュに保存するヘルパーメソッド
        """
        img_pkl_list = []
        mask_pkl_list = []

        for i, (img_path, mask_path) in enumerate(zip(img_paths, mask_paths), 1):
            img = load_image(img_path, img_h, img_w)
            mask = load_mask(mask_path, img_h, img_w, task="plant2d")

            img_patches = image_to_2d_patches(img, patch_h, patch_w)
            mask_patches = image_to_2d_patches(mask, patch_h, patch_w)

            for patch_idx, (img_patch, mask_patch) in enumerate(zip(img_patches, mask_patches), i):
                img_pkl_path = os.path.join(self.cache_dir, f"{patch_idx}_img.pkl")
                mask_pkl_path = os.path.join(self.cache_dir, f"{patch_idx}_mask.pkl")

                with open(img_pkl_path, "wb") as img_file, open(mask_pkl_path, "wb") as mask_file:
                    pickle.dump(img_patch, img_file)
                    pickle.dump(mask_patch, mask_file)

                img_pkl_list.append(img_pkl_path)
                mask_pkl_list.append(mask_pkl_path)

        return img_pkl_list, mask_pkl_list

    def __len__(self):
        return len(self.img_pkl_list)

    def __getitem__(self, idx):
        # キャッシュから読み込み
        with open(self.img_pkl_list[idx], "rb") as f:
            img = pickle.load(f)
        with open(self.mask_pkl_list[idx], "rb") as f:
            mask = pickle.load(f)

        # データ拡張
        if self.data_augmentator:
            augmented = self.data_augmentator(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255  # HWC -> CHW

        # マスクの変換とエンコーディング
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)  # [H, W, 1] -> [H, W]
        mask = torch.nn.functional.one_hot(torch.from_numpy(mask).long(), num_classes=self.num_classes).permute(2, 0, 1).float()  # [C, H, W]

        return img, mask
    