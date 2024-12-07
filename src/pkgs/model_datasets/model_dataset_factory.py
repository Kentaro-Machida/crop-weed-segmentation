from src.pkgs.data_classes.config_class import ModelDatasetConfig
from src.pkgs.model_datasets.transformer import TransformerModelDataset

class ModelDatasetFactory:
    """
    ModelDatasetクラスを生成するクラス
    受け取ったModelDatasetConfigによって、適切なModelDatasetクラスを生成する責任を持つ
    """
    def __init__(
            self,
            config:ModelDatasetConfig,
            data_root_path:str,
            model_dataset_type:str
            ):
        
        self.config = config
        self.data_root_path = data_root_path
        self.model_dataset_type = model_dataset_type

    
    def create(self):
        if self.model_dataset_type == "transformer":
            return TransformerModelDataset(self.config, self.data_root_path)
        

if __name__ == "__main__":
    import yaml
    yaml_config_path = "/home/machida/Documents/WeedSegmentation_2024/crop-weed-segmentation/config.yml"
    with open(yaml_config_path) as f:
        config = yaml.safe_load(f)
    
    model_dataset_config = ModelDatasetConfig(**config["experiment"]["model_dataset_config"])
    model_dataset_type = config["experiment"]["model_dataset_type"]
    data_root_path = config["experiment"]["data_root_path"]
    model_dataset_factory = ModelDatasetFactory(model_dataset_config, data_root_path, model_dataset_type)
    model_dataset = model_dataset_factory.create()

    print(model_dataset.get_model_datasets())
