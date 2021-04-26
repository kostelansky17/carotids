from pytorch_lightning import LightningModule
from torch.optim import RMSprop


class SegModule(LightningModule):
    def __init__(self, model, loss, lr_rate: float = 0.001) -> None:
        super().__init__()
        self.model = model
        self.loss = loss
        self.lr_rate = lr_rate

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch)

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("test_loss", loss)

    def _shared_step(self, batch):
        x, y = batch
        y_pred = self.forward(x)

        return self.loss(y, y_pred)

    def configure_optimizers(self):
        return RMSprop(self.model.parameters(), lr=self.lr_rate)
