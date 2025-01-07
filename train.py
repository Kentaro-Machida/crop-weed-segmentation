import mlflow
from lightning.pytorch import Trainer
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from src.pkgs.data_classes.config_class import ExperimentConfig, TrainConfig, MLflowConfig
from src.pkgs.model_datasets.model_dataset_factory import ModelDatasetFactory

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

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  # Metric to monitor
        dirpath="checkpoints/",  # Directory to save checkpoints
        filename="best-checkpoint-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,  # Save only the best model
        mode="min"  # Minimize the monitored metric
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,  # Number of epochs to wait before stopping
        mode="min"
    )

    trainer = Trainer(
        max_epochs=train_config.max_epochs,
        accelerator='cpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stopping_callback],
        log_every_n_steps=10
    )

    # Auto log all MLflow entities
    mlflow.pytorch.autolog()

    with mlflow.start_run():
        trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
