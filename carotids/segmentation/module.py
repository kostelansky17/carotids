from os import makedirs
from os.path import join

from pytorch_lightning import LightningModule
from torch import device
from torch.optim import RMSprop

from carotids.segmentation.dataset import SegmentationEvaluationDataset
from carotids.segmentation.visualization import plot_segmentation_prediction


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

    def plot_dataset(
        self,
        data_set: SegmentationEvaluationDataset,
        save_path: str,
        folder_name: str,
        device: device,
        image_shape: tuple = (224, 224),
    ) -> None:
        folder_path = join(save_path, folder_name)
        makedirs(folder_path)

        for img, raw_img, img_name, label, raw_label in data_set:
            img.to(device)

            prediction = model(img.unsqueeze(0))
            prediction = prediction.squeeze().detach().numpy().argmax(0)
            label = label.numpy()

            plot_segmentation_prediction(
                prediction, label, raw_img, image_shape, img_name, folder_path
            )

    def plot_datasets(
        self,
        datasets: tuple,
        save_path: str,
        device: device,
        image_shape: tuple = (224, 224),
    ):
        for data_set, name in zip(
            array(datasets), ["train_set", "validation_set", "test_set"]
        ):
            self.plot_dataset(data_set, save_path, name, device, image_shape)
