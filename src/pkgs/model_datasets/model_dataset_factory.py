from functools import partial

from src.pkgs.data_classes.config_class import ModelDatasetConfig
from src.pkgs.model_datasets.transformer import TransformerModelDataset
from src.pkgs.model_datasets.patches import Patch2dModelDataset
from src.pkgs.model_datasets.cnns import CNNModelDataset
from src.utils.data_loads import load_mask, LabelConverter

class ModelDatasetFactory:
    """
    ModelDatasetクラスを生成するクラス
    受け取ったModelDatasetConfigによって、適切なModelDatasetクラスを生成する責任を持つ
    """
    def __init__(
            self,
            config:ModelDatasetConfig,
            data_root_path:str,
            model_dataset_type:str,
            task:str
            ):
        
        self.config = config
        self.data_root_path = data_root_path
        self.model_dataset_type = model_dataset_type
        self.load_mask_func = partial(load_mask, task=task)
        label_converter = LabelConverter()
        self.label_dict = label_converter.task_to_label_dict(task)

    
    def create(self)->dict:
        """ModelDatasetクラスを生成するメソッド

        Raises:
            ValueError: model_dataset_typeがサポートされていない場合

        Returns:
            dict: {"model":model, "train_loader":train_loader, "val_loader":val_loader, "test_loader":test_loader}
        """        
        if self.model_dataset_type == "transformer":
            modeldataset = TransformerModelDataset(self.config, self.data_root_path)
        elif self.model_dataset_type == "patch2d":
            modeldataset = Patch2dModelDataset(self.config, self.data_root_path)
        elif self.model_dataset_type == "cnn":
            modeldataset = CNNModelDataset(self.config, self.data_root_path, self.load_mask_func, self.label_dict)
        else:
            raise ValueError(f"model_dataset_type: {self.model_dataset_type} is not supported.")
        return modeldataset
        

if __name__ == "__main__":
    import yaml
    yaml_config_path = "/home/machida/Documents/WeedSegmentation_2024/crop-weed-segmentation/config.yml"
    with open(yaml_config_path) as f:
        config = yaml.safe_load(f)
    
    modeldataset_config = ModelDatasetConfig(**config["experiment"]["modeldataset_config"])
    model_dataset_type = config["experiment"]["model_dataset_type"]
    data_root_path = config["experiment"]["data_root_path"]
    model_dataset_factory = ModelDatasetFactory(modeldataset_config, data_root_path, model_dataset_type)
    model_dataset = model_dataset_factory.create()
    print(model_dataset)
