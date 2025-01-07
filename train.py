import mlflow
from mlflow import MlflowClient
from lightning.pytorch import Trainer
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from src.pkgs.data_classes.config_class import ExperimentConfig, TrainConfig, MLflowConfig
from src.pkgs.model_datasets.model_dataset_factory import ModelDatasetFactory

def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print(f"run_id: {r.info.run_id}")
    print(f"artifacts: {artifacts}")
    print(f"params: {r.data.params}")
    print(f"metrics: {r.data.metrics}")
    print(f"tags: {tags}")

def main():
    with open("./config.yml") as f:
        config = yaml.safe_load(f)

    experiment_config = ExperimentConfig.from_dict(config["experiment"])
    train_config = TrainConfig(**config["experiment"]["train_config"])
    model_dataset_factory = ModelDatasetFactory(
        experiment_config.modeldataset_config,
        experiment_config.data_root_path,
        experiment_config.modeldataset_type
        )
    model_dataset_dict = model_dataset_factory.create()
    
    model = model_dataset_dict["model"]
    train_loader = model_dataset_dict["train_loader"]
    val_loader = model_dataset_dict["val_loader"]
    test_loader = model_dataset_dict["test_loader"]

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,  # Number of epochs to wait before stopping
        mode="min"
    )

    model_checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="./best_model",
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
    mlflow.pytorch.autolog()

    with mlflow.start_run() as run:
        trainer.fit(model, train_loader, val_loader)
        mlflow.log_artifact(local_path="./config.yml")

        best_model_path = model_checkpoint_callback.best_model_path
        trainer.test(dataloaders=test_loader, ckpt_path=best_model_path)
    
    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))

if __name__ == "__main__":
    main()
