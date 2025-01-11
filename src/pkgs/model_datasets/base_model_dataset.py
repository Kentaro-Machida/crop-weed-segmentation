import torch
from torchmetrics.functional import jaccard_index, accuracy
from torch.utils.data import Dataset
from lightning.pytorch import LightningModule

from abc import ABC, abstractmethod
from src.pkgs.data_classes.config_class import ModelDatasetConfig
from src.pkgs.preproceses.data_augmentation import DataTransformBuilder


class BaseDataset(ABC, Dataset):
    """
    Base class for dataset.
    このクラスの子クラスは、データセットを返す責任を追う
    """

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass


class BaseLightningModule(ABC, LightningModule):
    """
    Base class for LightningModule.
    このクラスの子クラスは、LightningModuleを返す責任を追う
    """
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def test_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def predict_step(self):
        pass

    @abstractmethod
    def configure_optimizers(self):
        pass

    def record_iou(self, y_hat, y, step:str):
        # IoUを計算して記録
        y_hat_probs = torch.softmax(y_hat, dim=1)  # 確率分布を計算
        y_hat_class = torch.argmax(y_hat_probs, dim=1)
        iou_scores = jaccard_index(y_hat_class, y, task='multiclass', num_classes=len(self.label_dict), average=None)
        for label, i in self.label_dict.items():
            self.log(f'{step}_IoU_{label}', iou_scores[i], on_epoch=True)

    def record_pixel_accuracy(self, y_hat, y, step:str):
        # Pixel Accuracyを計算して記録
        y_hat_probs = torch.softmax(y_hat, dim=1)
        y_hat_class = torch.argmax(y_hat_probs, dim=1)
        for label, i in self.label_dict.items():
            accuracy_scores = accuracy(y_hat_class, y, num_classes=self.num_classes, average=None, task="multiclass")
            self.log(f'{step}_Pixel_Accuracy_{label}', accuracy_scores[i], on_epoch=True)

    
class BaseModelDataset(ABC):
    """
    Base class for model and dataset.
    このクラスの子クラスは、モデルと対応するデータセットを返す責任を追う
    """
    def __init__(self, config: ModelDatasetConfig):
        self.image_height = config.image_height
        self.image_width = config.image_width

    @abstractmethod
    def get_model_datasets(self)->dict:
        """
        Return model and dataset pair.
        {
            "model": model,
            "train_dataset": train_dataset_name,
            "val_dataset": val_dataset_name,
            "test_dataset": test_dataset_name
        }
        """
        pass

    @abstractmethod
    def get_dataset(self):
        pass

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def get_image_mask_paths(self):
        pass


class TestModelDataset(BaseModelDataset):
    """
    Test dataset for model.
    """
    def __init__(self, config: ModelDatasetConfig):
        super().__init__(config)

    
    def get_model_datasets(self)->dict:
        return {
            "model": "test_model",
            "train_loader": "train_loader",
            "val_loader": "val_loader",
            "test_loader": "test_loader"
        }

if __name__ == "__main__":
    import yaml
    yaml_config_path = "/home/machida/Documents/WeedSegmentation_2024/crop-weed-segmentation/config.yml"
    with open(yaml_config_path) as f:
        config = yaml.safe_load(f)
    
    modeldataset_config = ModelDatasetConfig(**config["experiment"]["modeldataset_config"])
    test_model_dataset = TestModelDataset(modeldataset_config)
    print(test_model_dataset.get_model_datasets())

