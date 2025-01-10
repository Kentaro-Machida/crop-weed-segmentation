"""
学習済みモデルを読み込んで予測を行うためのスクリプト
"""

import mlflow
from mlflow import MlflowClient
from lightning.pytorch import Trainer
import yaml
import os
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from src.pkgs.data_classes.config_class import ExperimentConfig, TrainConfig, MLflowConfig
from src.pkgs.model_datasets.model_dataset_factory import ModelDatasetFactory
from src.utils.class_labels import task_to_label_dict
from src.pkgs.model_datasets.cnns import UNetppLightning

def predict(experiment_id: str, run_id: str):
    with open("./config.yml") as f:
        config = yaml.safe_load(f)
    predict_config_dict = config["predict"]

    # 実験ディレクトリのパス
    experiment_dir = os.path.join(predict_config_dict["tracking_dir"], experiment_id)
    # runまでのパス
    run_dir = os.path.join(experiment_dir, run_id)
    # 学習時のconfigファイルへのパス
    config_path = os.path.join(run_dir, "artifacts/config.yml")
    # 学習時の最良モデルへのパス
    model_path = os.path.join(run_dir, "artifacts/best_model.ckpt")

    with open(config_path) as f:
        train_config = yaml.safe_load(f)
    
    experiment_config = ExperimentConfig.from_dict(train_config["experiment"])
    model_dataset_config = experiment_config.modeldataset_config
    label_dict = task_to_label_dict(experiment_config.task)

    model = UNetppLightning.load_from_checkpoint(model_path, label_dict=label_dict, config=model_dataset_config)
    # test_set = 

if __name__ == "__main__":
    predict("685040055724994125", "148e22baf384423faad772ba5ab6ba3c")
    