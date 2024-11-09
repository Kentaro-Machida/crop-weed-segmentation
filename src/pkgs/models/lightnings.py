import torch
import torch.nn as nn
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torchmetrics.classification import BinaryJaccardIndex
from transformers import SegformerForSemanticSegmentation

from pkgs.data_classes.config_class import ConfigData
from pkgs.models.models import PatchFCN


class ModelFactory:
    def __init__(self, config: ConfigData):
        self.config = config

    def create_model(self)->pl.LightningModule:
        """
        指定されたタスクに適するモデルをconfig fileに従って生成し返す
        """
        if self.config.task == "plant1d":
            return Plant1dLightning(self.config)
        elif self.config.task == "plant2d":
            return Plant2dLightning(self.config)
        elif self.config.task == "crop":
            return CropLightning(self.config)
        elif self.config.task == "all":
            return AllLightning(self.config)
        else:
            raise ValueError(
                "Invalid task. Look the config file and check a task.")


class Plant1dLightning(pl.LightningModule):
    def __init__(self, config):
        super(Plant1dLightning, self).__init__()
        self.config = config
        self.model = PatchFCN(
            patch_heigth=config.plant_model.patch_size,
            patch_width=config.plant_model.patch_size,
            number_of_layers=6
        )
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        y_hat_class = (torch.sigmoid(y_hat) > 0.5).float()  # Sigmoid activation function is added here
        iou_score = self.iou(y_hat_class, y)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        y_hat_class = (torch.sigmoid(y_hat) > 0.5).float()  # Sigmoid activation function is added here
        iou_score = self.iou(y_hat_class, y)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-5)  # weight decay added
        return optimizer


class Plant2dLightning(pl.LightningModule):
    def __init__(self, config):
        super(Plant2dLightning, self).__init__()
        self.config = config
        self.model = smp.Unet(
            encoder_name="mobilenet_v2",
            encoder_weights="imagenet",
            classes=2
        )

        self.criterion = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        y_hat_class = (torch.sigmoid(y_hat) > 0.5).float()  # Sigmoid activation function is added here
        iou_score = self.iou(y_hat_class, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        y_hat_class = (torch.sigmoid(y_hat) > 0.5).float()  # Sigmoid activation function is added here
        iou_score = self.iou(y_hat_class, y)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-5)  # weight decay added
        return optimizer


class CropLightning(pl.LightningModule):
    def __init__(self, config: ConfigData):
        super(AllLightning, self).__init__()
        self.config = config
        self._backborn = self.config.all_model.backborn
        assert self._backborn in ["resnet50", "resnet101", "mobilenet_v2"], "you should choose resnet50, resnet101 or mobilenet_v2"
        self._model_type = self.config.all_model.modeltype

        self.criterion = torch.nn.CrossEntropyLoss()
        self.metric = BinaryJaccardIndex(threshold=0.5)

        if self._model_type == "unet":
            self.model = smp.Unet(
                encoder_name=self._backborn,
                encoder_weights="imagenet",
                classes=2,
            )
        elif self._model_type == "segformer":
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                pretrained_model_name_or_path="nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
                num_labels=2,
                ignore_mismatched_sizes=True
            )
        else:
            raise ValueError("Invalid model type")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if self._model_type == "segformer":
            inputs = batch["pixel_values"]
            targets = batch["labels"]
            outputs = self.model(pixel_values=inputs, labels=targets)
            loss = outputs.loss
        elif self._model_type == "unet":
            x, y = batch
            y_hat = self.model(x)
            loss = self.criterion(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        if self._model_type == "segformer":
            inputs = batch["pixel_values"]
            targets = batch["labels"]
            outputs = self.model(pixel_values=inputs, labels=targets)
            loss = outputs.loss
            self.log("val_loss", loss)
        elif self._model_type == "unet":    
            x, y = batch
            y_hat = self.model(x)
            loss = self.criterion(y_hat, y)
            self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class AllLightning(pl.LightningModule):
    def __init__(self, config: ConfigData):
        super(AllLightning, self).__init__()
        self.config = config
        self._backborn = self.config.all_model.backborn
        assert self._backborn in ["resnet50", "resnet101", "mobilenet_v2"], "you should choose resnet50, resnet101 or mobilenet_v2"
        self._model_type = self.config.all_model.modeltype

        self.criterion = torch.nn.CrossEntropyLoss()
        self.metric = BinaryJaccardIndex(threshold=0.5)

        if self._model_type == "unet":
            self.model = smp.Unet(
                encoder_name=self._backborn,
                encoder_weights="imagenet",
                classes=3,
            )
        elif self._model_type == "segformer":
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                pretrained_model_name_or_path="nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
                num_labels=3,
                ignore_mismatched_sizes=True
            )
        else:
            raise ValueError("Invalid model type")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if self._model_type == "segformer":
            inputs = batch["pixel_values"]
            targets = batch["labels"]
            outputs = self.model(pixel_values=inputs, labels=targets)
            loss = outputs.loss
        elif self._model_type == "unet":
            x, y = batch
            y_hat = self.model(x)
            loss = self.criterion(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        if self._model_type == "segformer":
            inputs = batch["pixel_values"]
            targets = batch["labels"]
            outputs = self.model(pixel_values=inputs, labels=targets)
            loss = outputs.loss
            self.log("val_loss", loss)
        elif self._model_type == "unet":    
            x, y = batch
            y_hat = self.model(x)
            loss = self.criterion(y_hat, y)
            self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

if __name__ == "__main__":
    from config import config
    config_data = ConfigData.load_data(config)
    model_factory = ModelFactory(config_data)
    model = model_factory.create_model()
    print(model)
