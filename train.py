import mlflow
from mlflow import MlflowClient
from lightning.pytorch import Trainer
import yaml
import tempfile
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from src.pkgs.data_classes.config_class import ExperimentConfig, TrainConfig, MLflowConfig
from src.pkgs.model_datasets.model_dataset_factory import ModelDatasetFactory

def print_auto_logged_info(r):
    """
    Logの結果を標準出力する関数
    """
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print(f"run_id: {r.info.run_id}")
    print(f"artifacts: {artifacts}")
    print(f"params: {r.data.params}")
    print(f"metrics: {r.data.metrics}")
    print(f"tags: {tags}")

def train(yaml_path: str, tempdir: str):
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    experiment_config = ExperimentConfig.from_dict(config["experiment"])
    train_config = TrainConfig(**config["experiment"]["train_config"])
    mlflow_config = MLflowConfig(**config["experiment"]["mlflow_config"])
    
    model_dataset_factory = ModelDatasetFactory(
        experiment_config.modeldataset_config,
        experiment_config.data_root_path,
        experiment_config.modeldataset_type,
        task=config["experiment"]["task"]
        )
    model_dataset = model_dataset_factory.create()
    model_dataset_dict = model_dataset.get_model_datasets()
    
    model = model_dataset_dict["model"]
    train_loader = model_dataset_dict["train_loader"]
    val_loader = model_dataset_dict["val_loader"]
    test_loader = model_dataset_dict["test_loader"]

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,  # Number of epochs to wait before stopping
        mode="min"
    )

    # 一時ディレクトリを作成
    model_checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=tempdir,
        filename="best_model",
        save_top_k=1,
        mode="min"
    )

    trainer = Trainer(
        max_epochs=train_config.max_epochs,
        accelerator='cpu',
        devices=1,
        callbacks=[early_stopping_callback, model_checkpoint_callback],
        log_every_n_steps=10
    )

    # Auto log all MLflow entities
    mlflow.set_experiment(mlflow_config.experiment_name)
    mlflow.set_tracking_uri(mlflow_config.tracking_uri)
    mlflow.pytorch.autolog()

    with mlflow.start_run() as run:
        # MLflowのタグを設定
        mlflow.set_tags({
            "model_dataset_type": experiment_config.modeldataset_type,
            "task": experiment_config.task
        })
        trainer.fit(model, train_loader, val_loader)
        mlflow.log_artifact(local_path="./config.yml")

        best_model_path = model_checkpoint_callback.best_model_path
        # アーティファクトにモデルを保存
        mlflow.log_artifact(local_path=best_model_path)
        trainer.test(dataloaders=test_loader, ckpt_path=best_model_path)
    
    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tempdir:
        train("./config.yml", tempdir)    
