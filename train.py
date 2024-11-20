import pytorch_lightning as pl
import yaml

def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    config = ConfigData.load_data(config)
    dataset_factory = DatasetFactory(config)
    dataset_dict = dataset_factory.create_dataset()
    train_loader = dataset_dict["train"]
    val_loader = dataset_dict["val"]
    test_loadtrain_loader = dataset_dict["test"]

    model_factory = ModelFactory(config)
    model = model_factory.create_model()

    trainer = pl.Trainer(
        max_epochs=config.train_parameter.max_epocks,
        accelerator=config.train_parameter.device,
    )

    trainer.fit(model, train_loader, val_loader)
