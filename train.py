import pytorch_lightning as pl
import yaml
from src.pkgs.data_classes.config_class import ExperimentConfig, TrainConfig
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

    trainer = pl.Trainer(
        max_epochs=train_config.max_epochs,
        accelerator='cpu'
    )
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
