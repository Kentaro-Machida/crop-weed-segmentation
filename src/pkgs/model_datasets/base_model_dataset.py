from abc import ABC, abstractmethod
from src.pkgs.data_classes.config_class import ModelDatasetConfig
from src.pkgs.preproceses.data_augmentation import DataTransformBuilder


class BaseModelDataset(ABC):
    """
    Base class for model and dataset.
    このクラスの子クラスは、モデルと対応するデータセットを返す責任を追う
    """
    def __init__(self, config: ModelDatasetConfig):
        self.image_height = config.image_height
        self.image_width = config.image_width
        self.transform = DataTransformBuilder(config.data_augmentation_config)

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

