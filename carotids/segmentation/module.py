from os import makedirs
from os.path import exists, join

from numpy import arange, asarray, mean, ndarray
import plotly.graph_objects as go
from pytorch_lightning import LightningModule
from sklearn.metrics import confusion_matrix
from torch import device, load, logical_and, logical_or, save, set_grad_enabled, Tensor
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau

from carotids.segmentation.dataset import SegmentationEvaluationDataset
from carotids.segmentation.metrics import dataset_classes_iou, SegAccuracy
from carotids.segmentation.visualization import plot_segmentation_prediction


import matplotlib.pyplot as plt


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
        accuracy: SegAccuracy = None,
        loss_weights: list = [],
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
        self.loss_weights = loss_weights

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

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
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
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def test_step(self, batch: tuple, batch_idx: int) -> tuple:
        """Passes an batch trough the network and evaluates it using loss
        function. The test loss is logged (if the Accuracy module was passed
        during the initialization, the accuracy is logged as well).

        Parameters
        ----------
        batch : tuple
            The tuple with the input to the model and the label.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        tuple
            The loss of the batch. If the accuracy in the module was defined, it
            is computed and returned as well.
        """
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y)
        self.log("test_loss", loss)

        if self.accuracy is not None:
            accuracy = self.accuracy(y_pred, y)
            self.log("test_acc", accuracy)

            return loss, accuracy

        return loss, None

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

        return self.loss(y_pred, y, self.loss_weights)

    def configure_optimizers(self) -> dict:
        """Configures the optimizers used in training. Returns a dictionary with
        optimizer, learning rate scheduler and the name of the metric to
        monitor for reducing the learning rate.

        Returns
        -------
        dict
            The optimizers used during the training.
        """
        optimizer = RMSprop(self.model.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": ReduceLROnPlateau(
                optimizer, patience=self.patience, min_lr=10e-7
            ),
            "monitor": "train_loss_epoch",
        }

    def load_model(self, model_path: str) -> None:
        """Loads the model's parameters from the specified path.

        Parameters
        ----------
        model_path : str
            Path to the model's parameters.

        Returns
        -------
        dict
            The optimizers used during the training.
        """
        self.model.load_state_dict(load(model_path))

    def evaluate_dataloader(self, data_loader: DataLoader, device: device) -> float:
        """Loads the model's parameters from the specified path.

        Parameters
        ----------
        data_loader : DataLoader
            DataLoader to evaluate.
        device : device
            Device used for computation.

        Returns
        -------
        float
            The mean loss of the DataLoader.
        """
        self.model.eval()
        self.model.to(device)
        losses = []
        accuracies = []

        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            loss, accuracy = self.test_step((inputs, labels), 0)

            if self.accuracy is not None:
                accuracies.append(accuracy.item())

            losses.append(loss.item())

        if self.accuracy is not None:
            return mean(losses), mean(accuracies)

        return mean(losses)

    def train_model(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        device: device,
        epochs: int,
        save_path: str,
        patience: int = 250,
    ) -> None:
        """Trains the model on the given data.

        Parameters
        ----------
        train_loader : DataLoader
            The training data.
        validation_loader : DataLoader
            The validation data.
        device : device
            Device used for computation.
        epochs : int
            The maximal number of training epochs.
        save_path : str
            Path to save the best model.
        patience : int
            Number of epoch to wait for improvement od the validation loss.
        """
        losses = {"train_loss": [], "val_loss": []}

        optimizers = self.configure_optimizers()
        optimizer = optimizers["optimizer"]
        lr_scheduler = optimizers["lr_scheduler"]
        best_val_loss = 10 ** 8

        self.model.to(device)
        for epoch in range(epochs):
            print(f"Epoch {epoch}/{epochs - 1}")
            print("-" * 12)

            train_epoch_loss = 0.0
            self.model.train()

            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with set_grad_enabled(True):
                    l = self.training_step((inputs, labels), 0)
                    l.backward()
                    optimizer.step()
                    train_epoch_loss += l.item() * inputs.size(0)

            lr_scheduler.step(train_epoch_loss)

            val_epoch_loss = 0.0
            self.model.eval()

            for inputs, labels in validation_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                with set_grad_enabled(False):
                    l = self.validation_step((inputs, labels), 0)
                    val_epoch_loss += l.item() * inputs.size(0)

            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                save(self.model.state_dict(), save_path + ".pt")
                val_improved_epoch = epoch

            losses["train_loss"].append(train_epoch_loss / len(train_loader.dataset))
            losses["val_loss"].append(val_epoch_loss / len(validation_loader.dataset))

            print(
                f"Train loss: {train_epoch_loss/len(train_loader.dataset)}, Val. loss: {val_epoch_loss/len(validation_loader.dataset)}"
            )

            if epoch - val_improved_epoch > patience:
                print("Training terminated...")
                break

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=[i for i in range(len(losses["train_loss"]))],
                y=losses["train_loss"],
                mode="lines",
                name="train_loss",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[i for i in range(len(losses["val_loss"]))],
                y=losses["val_loss"],
                mode="lines",
                name="val_loss",
            )
        )

        fig.update_layout(xaxis_title="Epoch", yaxis_title="Loss")
        fig.write_image(save_path + ".png")

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
        self.model.eval()
        self.model.to(device)
        cnf_matrices = []

        for img, _, _, label, _ in data_set:
            img = img.to(device)
            prediction = self.model(img.unsqueeze(0))

            cnf = confusion_matrix(
                prediction.argmax(dim=1).view(-1).cpu().numpy(),
                label.argmax(dim=0).view(-1).cpu().numpy(),
                labels=arange(label.size()[0]),
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
        """Plots the predictions for the data_set.

        Parameters
        ----------
        data_set : SegmentationEvaluationDataset
            Dataset to plot.
        save_path : str
            Path to save folder with the plots.
        folder_name : str
            A name of the folder to which save the plots.
        device : device
            Device used for computation.
        image_shape : tuple
            The shape of the network's input.
        """
        self.model.eval()
        self.model.to(device)
        folder_path = join(save_path, folder_name)

        if not exists(folder_path):
            makedirs(folder_path)

        for img, raw_img, img_name, label, raw_label in data_set:
            img = img.to(device)

            prediction = self.model(img.unsqueeze(0))
            prediction = prediction.squeeze().cpu().detach().numpy().argmax(0)
            label = label.numpy()

            plot_segmentation_prediction(
                prediction,
                label,
                raw_img,
                raw_label,
                image_shape,
                img_name,
                folder_path,
            )

    def plot_datasets(
        self,
        datasets: list,
        save_path: str,
        device: device,
        image_shape: tuple = (256, 256),
    ) -> None:
        """Plots the provided datasets. Usually training, validation and test
        one. Function expects the datasets to be sorted in this way.

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
        save_path = join(save_path, "model_predictions")
        if not exists(save_path):
            makedirs(save_path)

        for data_set, name in zip(
            datasets, ["train_set", "validation_set", "test_set"]
        ):
            print(f"Plotting {name}")
            self.plot_dataset(data_set, save_path, name, device, image_shape)

    def dataset_iou(
        self, dataset: SegmentationEvaluationDataset, device: device, n_classes: int
    ) -> list:
        """Computes IoU of all of the classes in the dataset.

        Parameters
        ----------
        dataset : SegmentationEvaluationDataset
            Dataset to evaluate.
        device : device
            Device used for computation.
        n_classes : int
            Number of classes.

        Returns
        -------
        list
            The mean IoUs of the classes.
        """
        self.model.to(device)
        self.model.eval()

        classes_iou = {i: [] for i in range(n_classes)}

        for img, _, _, label, _ in dataset:
            img = img.to(device)
            label = label.to(device)

            prediction = self.model(img.unsqueeze(0))

            prediction = prediction.squeeze().argmax(0)
            label = label.argmax(0)

            for i in range(n_classes):
                classes_iou[i].append(
                    logical_and(prediction == i, label == i).sum().detach().item()
                    / logical_or(prediction == i, label == i).sum().detach().item()
                )

        return [mean(classes_iou[i]) for i in range(n_classes)]

    def datasets_iou(self, datasets: list, device: device, n_classes: int) -> dict:
        """Computes IoU of all of the classes in the datasets.

        Parameters
        ----------
        dataset : SegmentationEvaluationDataset
            Dataset to evaluate.
        device : device
            Device used for computation.
        n_classes : int
            Number of classes.

        Returns
        -------
        dict
            Classes IoU of all the sets.
        """
        ious = {}
        for dataloader, name in zip(
            datasets, ["train_set", "validation_set", "test_set"]
        ):

            ious[name] = self.dataset_iou(dataloader, device, n_classes)

        return ious
