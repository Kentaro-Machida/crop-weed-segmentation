import os
from pydantic import (
    BaseModel,
    ValidationError,
    field_validator
)

"""
実験の設定を管理する責任を持つクラス
Configファイルで指定された値が正しいかどうかの検証の責任も持つ
"""

class PatchSetting(BaseModel):
    patch_size: int


class TransformerSetting(BaseModel):
    pretrained_model: str


class CNNSetting(BaseModel):
    pass


class TwoSetting(BaseModel):
    two_size: int


class DataAugmentationConfig(BaseModel):
    vertical_flip: bool
    horizontal_flip: bool
    random_brightness_contrast: bool


class ModelDatasetConfig(BaseModel):
    image_height: int
    image_width: int
    num_classes: int
    lr: float
    criterion: str
    metric: str
    batch_size: int
    data_augmentation_config: DataAugmentationConfig

    # Depending on the model_dataset_type, the following classes are used
    patch_setting: PatchSetting = None
    transformer_setting: TransformerSetting = None
    cnn_setting: CNNSetting = None
    two_setting: TwoSetting = None

    @field_validator("criterion")
    def check_criterion(cls, v):
        allowed = ["cross_entropy", "dice"]
        if v not in allowed:
            raise ValueError(f"Criterion is '{v}', it should be one of {allowed}.")
        return v
    
    @field_validator("metric")
    def check_metric(cls, v):
        allowed = ["binary_jaccard"]
        if v not in allowed:
            raise ValueError(f"Metric is '{v}', it should be one of {allowed}.")
        return v

    def from_dict(config:dict):
        data_augmentation_config = DataAugmentationConfig(**config["data_augmentation_config"])
        transformer_setting = TransformerSetting(**config["transformer_setting"])
        patch_setting = PatchSetting(**config["patch_setting"])
        cnn_setting = CNNSetting(**config["cnn_setting"])
        two_setting = TwoSetting(**config["two_setting"])

        return ModelDatasetConfig(
            image_height = config["image_height"],
            image_width = config["image_width"],
            num_classes = config["num_classes"],
            lr = config["lr"],
            criterion = config["criterion"],
            metric = config["metric"],
            batch_size = config["batch_size"],
            data_augmentation_config=data_augmentation_config,
            transformer_setting=transformer_setting,
            patch_setting=patch_setting,
            cnn_setting=cnn_setting,
            two_setting=two_setting
        )



class TrainConfig(BaseModel):
    max_epochs: int
    optimizer: str
    scheduler: str

    @field_validator("optimizer")
    def check_optimizer(cls, v):
        allowed= ["adamw"]
        if v not in allowed:
            raise ValueError(f"Optimizer is '{v}', it should be one of {allowed}.")
        return v
    
    @field_validator("scheduler")
    def check_scheduler(cls, v):
        allowed = ["step"]
        if v not in allowed:
            raise ValueError(f"Scheduler is '{v}', it should be one of {allowed}.")
        return v


class MLflowConfig(BaseModel):
    tracking_uri: str
    experiment_name: str


class ExperimentConfig(BaseModel):
    data_root_path: str  # train, val, test folders are needed in this path
    modeldataset_type: str  

    train_config: TrainConfig
    mlflow_config: MLflowConfig
    modeldataset_config: ModelDatasetConfig

    @field_validator("data_root_path")
    def check_data_root_path(cls, v):
        child_dirs = os.listdir(v)
        if "train" not in child_dirs or "val" not in child_dirs or "test" not in child_dirs:
            raise ValueError(f"data_root_path: {v} should include 'train', 'val', 'test' folders.")
        return v

    @field_validator("modeldataset_type")
    def check_model_dataset_type(cls, v):
        allowed = ["patch", "transformer", "cnn", "two_input"]
        if v not in allowed:
            raise ValueError(f"model_dataset_type is '{v}', it should be one of {allowed}.")
        return v
    
    def from_dict(config_dict:dict):
        data_root_path = config_dict["data_root_path"]
        modeldataset_type = config_dict["modeldataset_type"]

        train_config = TrainConfig(**config_dict["train_config"])
        mlflow_config = MLflowConfig(**config_dict["mlflow_config"])
        modeldataset_config = ModelDatasetConfig(**config_dict["modeldataset_config"])

        return ExperimentConfig(
            data_root_path=data_root_path,
            modeldataset_type=modeldataset_type,
            train_config=train_config,
            mlflow_config=mlflow_config,
            modeldataset_config=modeldataset_config
        )


if __name__ == "__main__":
    import yaml
    yaml_config_path = "/home/machida/Documents/WeedSegmentation_2024/crop-weed-segmentation/config.yml"
    with open(yaml_config_path) as f:
        config = yaml.safe_load(f)
    
    try:
        # train_config test
        train_config = TrainConfig(**config["experiment"]["train_config"])
        print("train_config test is complete.")

        # mlflow_config test
        mlflow_config = MLflowConfig(**config["experiment"]["mlflow_config"])
        print("mlflow_config test is complete.")

        # modeldataset_config test
        modeldataset_config = ModelDatasetConfig.from_dict(config["experiment"]["modeldataset_config"])
        print("modeldataset_config test is complete.")

        # experiment_config test
        experiment_config = ExperimentConfig.from_dict(config["experiment"])
        print("experiment_config test is complete.")
    except ValidationError as e:
        print(e)
    
