import pytorch_lightning as pl


class Plant1dModel(pl.LightningModule):
    def __init__(self, config):
        super(Plant1dModel, self).__init__()
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


class Plant2dModel(pl.LightningModule):
    def __init__(self, config):
        super(Plant2dModel, self).__init__()
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
