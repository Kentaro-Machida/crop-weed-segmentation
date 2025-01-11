import os 
import torch
from lightning.pytorch import LightningModule
from torch.utils.data import DataLoader
from torchmetrics.functional import jaccard_index
import segmentation_models_pytorch as smp

from src.utils.data_loads import get_image_path, load_image
from src.utils import check_no_overlap
from src.utils.roboflow_image_mask_split import sort_and_check_consistncy
from src.pkgs.preproceses.data_augmentation import DataTransformBuilder, get_normalizer
from .base_model_dataset import BaseModelDataset, BaseDataset
from src.pkgs.data_classes.config_class import ModelDatasetConfig


class CNNModelDataset(BaseModelDataset):
    """
    CNNモデルとそれに対応するデータセットを用意するクラス
    """
    def __init__(self, config: ModelDatasetConfig, data_root_path: str, load_mask_func: callable, label_dict:dict):
        super().__init__(config)
        self.model = UNetppLightning(config, label_dict=label_dict)

        # 以下はModelDataset系統クラスにおいて共通の処理
        data_augmentator = DataTransformBuilder(config.data_augmentation_config)
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
        self._val_dataset = self._get_dataset(self._val_img_paths, self._val_mask_paths, normalizer)
        self._test_dataset = self._get_dataset(self._test_img_paths, self._test_mask_paths, normalizer)

    def get_model_datasets(self):

        return {
            "model": self.model,
            "train_loader": DataLoader(self._train_dataset, self.batch_size, num_workers=2),
            "val_loader": DataLoader(self._val_dataset, self.batch_size, num_workers=2),
            "test_loader": DataLoader(self._test_dataset, self.batch_size, num_workers=2)
        }

    def _get_image_mask_paths(self, data_root_path:str, phase:str) -> tuple:
        img_dir = os.path.join(data_root_path, phase, "images")
        mask_dir = os.path.join(data_root_path, phase, "masks")
        img_paths = get_image_path(img_dir)
        mask_paths = get_image_path(mask_dir)
        return img_paths, mask_paths
    
    def _get_dataset(self, img_paths: list, mask_paths: list, data_augmentator=None):
        return CNNDataset(
            img_path_list=img_paths,
            mask_path_list=mask_paths,
            img_height=self.image_height,
            img_width=self.image_width,
            data_augmentator=data_augmentator,
            num_classes=self.num_classes,
            load_mask_func=self.load_mask_func
        )
    
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


class UNetppLightning(LightningModule):
    def __init__(self, config: ModelDatasetConfig, label_dict:dict):
        super(UNetppLightning, self).__init__()
        self.num_classes = len(label_dict)
        self.label_dict = label_dict
        
        self.model = smp.UnetPlusPlus(
            encoder_name=config.cnn_setting.backborn,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
            classes=self.num_classes,                      # model output channels (number of classes in your dataset)
        )

        if config.criterion == "cross_entropy":
            if self.num_classes > 2:
                self.criterion = torch.nn.CrossEntropyLoss()
            else:
                self.criterion = torch.nn.BCEWithLogitsLoss()
            self.loss_func = self._get_cross_entropy_loss
        elif config.criterion == "dice":
            self.criterion = smp.losses.DiceLoss(mode='binary' if self.num_classes == 2 else 'multiclass', from_logits=True)
            self.loss_func = self._get_dice_loss

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)

        self._record_iou(y_hat, y, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        self.log('val_loss', loss, on_epoch=True)

        self._record_iou(y_hat, y, "val")
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        self.log('test_loss', loss, on_epoch=True)

        self._record_iou(y_hat, y, "test")
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-5)  # weight decay added
        return optimizer
    
    def _record_iou(self, y_hat, y, step:str):
        # IoUを計算して記録
        y_hat_probs = torch.softmax(y_hat, dim=1)  # 確率分布を計算
        y_hat_class = torch.argmax(y_hat_probs, dim=1)
        iou_scores = jaccard_index(y_hat_class, y, task='multiclass', num_classes=len(self.label_dict), average=None)
        for label, i in self.label_dict.items():
            self.log(f'{step}_IoU_{label}', iou_scores[i], on_epoch=True)

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
        

class CNNDataset(BaseDataset):
    def __init__(
        self,
        img_path_list: list,
        mask_path_list: list,
        img_height: int,
        img_width: int,
        load_mask_func: callable,
        num_classes: int,
        data_augmentator=None,
    ):
        self._data_augmentator = data_augmentator
        self.num_classes = num_classes
        self.load_mask_func = load_mask_func
        
        self.img_height = img_height
        self.img_width = img_width
        self.img_path_list = img_path_list
        self.mask_path_list = mask_path_list

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

        # データ拡張
        if self._data_augmentator:
            augmented = self._data_augmentator(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255  # HWC -> CHW
        mask = torch.from_numpy(mask).long()

        return img, mask
    
    def check_image(self, idx) -> tuple:
        img = load_image(
            self.img_path_list[idx],
            self.img_height,
            self.img_width
            )
        
        return img
    
