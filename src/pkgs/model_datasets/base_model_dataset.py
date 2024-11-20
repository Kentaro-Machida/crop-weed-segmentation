from abc import ABC, abstractmethod
from src.pkgs.data_classes.config_class import ModelDatasetConfig


class BaseModelDataset(ABC):
    """
    Base class for dataset.
    """
    def __init__(self, config: ModelDatasetConfig):
        self.image_height = config.image_height
        self.image_width = config.image_width
        self.class_num = config.num_classes

    @abstractmethod
    def get_model_dataset(self)->dict:
        """
        Return model and dataset pair.
        """
        pass


class TestModelDataset(BaseModelDataset):
    """
    Test dataset for model.
    """
    def get_model_dataset(self)->dict:
        return {
            "model": "test_model",
            "dataset": "test_dataset"
        }

if __name__ == "__main__":
    import yaml
    yaml_config_path = "/home/machida/Documents/WeedSegmentation_2024/crop-weed-segmentation/config.yml"
    with open(yaml_config_path) as f:
        config = yaml.safe_load(f)
    
    model_dataset_config = ModelDatasetConfig(**config["experiment"]["model_dataset_config"])
    test_model_dataset = TestModelDataset(model_dataset_config)
    print(test_model_dataset.get_model_dataset())

