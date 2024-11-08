import pytorch_lightning as pl


class CropModel(pl.LightningModule):
    def __init__(self, config):
        super(CropModel, self).__init__()
        self.config = config

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass
