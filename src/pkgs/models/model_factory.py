from pkgs.data_classes.config_class import ConfigData
from all_models import AllModel
from crop_models import CropModel
from plant_models import Plant1dModel, Plant2dModel
import pytorch_lightning as pl


class ModelFactory:
    def __init__(self, config: ConfigData):
        self.config = config

    def create_model(self)->pl.LightningModule:
        """
        指定されたタスクに適するモデルをconfig fileに従って生成し返す
        """
        if self.config.task == "plant1d":
            return Plant1dModel(self.config)
        elif self.config.task == "plant2d":
            return Plant2dModel(self.config)
        elif self.config.task == "crop":
            return CropModel(self.config)
        elif self.config.task == "all":
            return AllModel(self.config)
        else:
            raise ValueError(
                "Invalid task. Look the config file and check a task.")
        
