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

class DataAugmentationConfig(BaseModel):
    vertical_flip: bool
    horizontal_flip: bool
    random_brightness_contrast: bool


class ModelDatasetConfig(BaseModel):
    image_height: int
    image_width: int
    num_classes: int
    data_augmentation_config: DataAugmentationConfig


class PatchModelDatasetConfig(ModelDatasetConfig):
    patch_size: int


class TransformerModelDatasetConfig(ModelDatasetConfig):
    pretrained_model: str


class CNNModelDatasetConfig(ModelDatasetConfig):
    pass


class TwoInputModelDatasetConfig(ModelDatasetConfig):
    patch_size: int





class TrainConfig(BaseModel):
    batch_size: int
    lr: float
    max_epochs: int
    optimizer: str
    scheduler: str
    criterion: str

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
    
    @field_validator("criterion")
    def check_criterion(cls, v):
        allowed = ["cross_entropy", "dice"]
        if v not in allowed:
            raise ValueError(f"Criterion is '{v}', it should be one of {allowed}.")
        return v


class MLflowConfig(BaseModel):
    tracking_uri: str
    experiment_name: str


class ExperimentConfig(BaseModel):
    data_root_path: str  # train, val, test folders are needed in this path
    model_dataset_type: str  

    train_config: TrainConfig
    mlflow_config: MLflowConfig

    @field_validator("data_root_path")
    def check_data_root_path(cls, v):
        child_dirs = os.listdir(v)
        if "train" not in child_dirs or "val" not in child_dirs or "test" not in child_dirs:
            raise ValueError(f"data_root_path: {v} should include 'train', 'val', 'test' folders.")
        return v

    @field_validator("model_dataset_type")
    def check_model_dataset_type(cls, v):
        allowed = ["patch", "transformer", "cnn", "two_input"]
        if v not in allowed:
            raise ValueError(f"model_dataset_type is '{v}', it should be one of {allowed}.")
        return v


if __name__ == "__main__":
    import yaml
    yaml_config_path = "/home/machida/Documents/WeedSegmentation_2024/crop-weed-segmentation/config.yml"
    with open(yaml_config_path) as f:
        config = yaml.safe_load(f)
    
    try:
        # data_validation_config test
        data_validation_config = DataValidationConfig(**config["experiment"]["data_validation_config"])
        print(data_validation_config)

        # train_config test
        train_config = TrainConfig(**config["experiment"]["train_config"])
        print(train_config)

        # mlflow_config test
        mlflow_config = MLflowConfig(**config["experiment"]["mlflow_config"])
        print(mlflow_config)

        # experiment_config test
        experiment_config = ExperimentConfig(
            data_root_path=config["experiment"]["data_root_path"],
            model_dataset_type=config["experiment"]["model_dataset_type"],
            num_classes=config["experiment"]["num_classes"],
            data_validation_config=data_validation_config,
            train_config=train_config,
            mlflow_config=mlflow_config
        )
        print(experiment_config)
        print("Complete.")
    except ValidationError as e:
        print(e)
    
