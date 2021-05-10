from os import makedirs
from os.path import exists, join

from numpy import arange, asarray, ndarray
from pytorch_lightning import LightningModule
from sklearn.metrics import confusion_matrix
from torch import device, Tensor
from torch.nn import Module
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau

from carotids.segmentation.dataset import SegmentationEvaluationDataset
from carotids.segmentation.metrics import SegAccuracy
from carotids.segmentation.visualization import plot_segmentation_prediction


class SegModule(LightningModule):
    """The segmentation module is used during the training, evaluation and
    visualization of the results.
    """

    def __init__(
        self,
        model: Module,
        loss: Module,
        learning_rate: float = 0.001,
        patience: int = 50, 
        accuracy: SegAccuracy = None
    ) -> None:
        """Initializes the Segmentation Module.

        Parameters
        ----------
        model : Module
            The model to use.
        loss : Module
            The loss function used during the training and the evaluation.
        learning_rate : float
            The learning rate used during the traing.
        patience : int
            Number of epochs with no improvement after which learning rate will 
            be reduced. 
        accuracy : SegAccuracy
            The Accuracy module.
        """
        super().__init__()
        self.model = model
        self.loss = loss
        self.learning_rate = learning_rate
        self.patience = patience
        self.accuracy = accuracy

    def forward(self, x: Tensor) -> Tensor:
        """Passes an input trough the network.

        Parameters
        ----------
        x: Tensor
            Input to the model.
        
        Returns
        -------
        Tensor
            The prediction of the model.
        """
        return self.model.forward(x)

    def training_step(self, batch: tuple, batch_idx: int) -> Tensor:
        """Passes an batch trough the network and evaluates it using loss 
        function. The training loss is logged.

        Parameters
        ----------
        batch : tuple
            The tuple with the input to the model and the label.
        batch_idx : int
            Index of the batch.
        
        Returns
        -------
        Tensor
            The loss of the batch.
        """

        loss = self._shared_step(batch)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> Tensor:
        """Passes an batch trough the network and evaluates it using loss 
        function. The validation loss is logged.

        Parameters
        ----------
        batch : tuple
            The tuple with the input to the model and the label.
        batch_idx : int
            Index of the batch.
        
        Returns
        -------
        Tensor
            The loss of the batch.
        """
        loss = self._shared_step(batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        """Passes an batch trough the network and evaluates it using loss 
        function. The test loss is logged (if the Accuracy module was passed 
        during the initialization, the accuracy is logged as well).

        Parameters
        ----------
        batch : tuple
            The tuple with the input to the model and the label.
        batch_idx : int
            Index of the batch.

        """
        x, y = batch
        y_pred = self.forward(x)
        self.log("test_loss", self.loss(y, y_pred))
        
        if self.accuracy is not None:
            self.log("test_acc", self.accuracy(y, y_pred))

    def _shared_step(self, batch: tuple) -> Tensor:
        """Passes an batch trough the network and evaluates it using loss 
        function.

        Parameters
        ----------
        batch : tuple
            The tuple with the input to the model and the label.
        
        Returns
        -------
        Tensor
            The loss of the batch.
        """
        x, y = batch
        y_pred = self.forward(x)

        return self.loss(y, y_pred)

    def configure_optimizers(self)-> dict:
        """Configures the optimizers used in training. Returns a dictionary with
        optimizer, leartning rate scheduler and the name of the metric to 
        monitor for reducing the learning rate.

        Returns
        -------
        dict
            The optimizers used during the training.
        """
        optimizer = RMSprop(self.model.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": ReduceLROnPlateau(optimizer, patience=self.patience),
            "monitor": "train_loss_epoch",
        }
 
    def confusion_matrix(
        self,
        data_set: SegmentationEvaluationDataset,
        device: device,
    ) -> ndarray:
        """Computes the confusion matrix on a data_set.

        Parameters
        ----------
        data_set : SegmentationEvaluationDataset
            Dataset to evaluate.
        device : device
            Device used for computation.
        
        Returns
        -------
        ndarray
            The confusion matrix.
        """
        self.model.to(device)
        cnf_matrices = []

        for img, _, _, label, _ in data_set:
            img = img.to(device)
            prediction = self.model(img.unsqueeze(0))
            
            cnf = confusion_matrix(
                    prediction.argmax(dim=1).view(-1).cpu().numpy(),
                    label.argmax(dim=0).view(-1).cpu().numpy(),
                    labels=arange(label.size()[0])
                )

            cnf_matrices.append(cnf)

        cnf_matrices = asarray(cnf_matrices)
        return cnf_matrices.mean(axis=0)

    def plot_dataset(
        self,
        data_set: SegmentationEvaluationDataset,
        save_path: str,
        folder_name: str,
        device: device,
        image_shape: tuple = (256, 256),
    ) -> None:
        """.

        Parameters
        ----------
        data_set : SegmentationEvaluationDataset
            .
        save_path : str
            .
        folder_name : str
            .
        device : device
            Device used for computation.
        image_shape : tuple
            The shape of the network's input.
        """
        folder_path = join(save_path, folder_name)
        if not exists(folder_path):
            makedirs(folder_path)

        for img, raw_img, img_name, label, raw_label in data_set:
            img = img.to(device)

            prediction = self.model(img.unsqueeze(0))
            prediction = prediction.squeeze().cpu().detach().numpy().argmax(0)
            label = label.numpy()

            plot_segmentation_prediction(
                prediction, label, raw_img, raw_label, image_shape, img_name, folder_path
            )

    def plot_datasets(
        self,
        datasets: list,
        save_path: str,
        device: device,
        image_shape: tuple = (256, 256),
    ):
        """.

        Parameters
        ----------
        datasets : list
            List of datasets to plot the figures.
        save_path : str
            The directory in which are saved the plots.
        device : device
            Device used for computation.
        image_shape : tuple
            The shape of the network's input.
        """
        self.model.to(device)
        save_path = join (save_path, "model_predictions")
        if not exists(save_path):
            makedirs(save_path)

        for data_set, name in zip(
            datasets, ["train_set", "validation_set", "test_set"]
        ):
            print(f"Plotting {name}")
            self.plot_dataset(data_set, save_path, name, device, image_shape)
