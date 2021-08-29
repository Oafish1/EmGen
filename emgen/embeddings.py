from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
import torch as t
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .utilities import BasicBlock


device = t.device("cuda:0" if t.cuda.is_available() else "cpu")


class emgen_model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        def main_block(inputs, outputs):
            return nn.Sequential(
                BasicBlock(inputs, outputs),
                BasicBlock(outputs, outputs),
            )

        self.preprocess = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.blocks = nn.ModuleList([
            main_block(64, 64),
            main_block(64, 128),
            main_block(128, 256),
            main_block(256, 512),
        ])
        self.postprocess = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 10),
        )

        self.device_param = nn.Parameter(t.empty(0))
        self.to(device)

    def forward(self, x):
        x = x.to(self.device_param.device)
        out = x.permute(0, 3, 1, 2)

        out = self.preprocess(out)
        for block in self.blocks:
            out = block(out)
        out = self.postprocess(out)

        return out

    def loss(self, labels, logits):
        ids = t.unique(labels)
        ids = [id.item() for id in ids]
        embeddings = {id: t.mean(logits[labels == id], 0)
                      for id in ids}

        # Separability, compactness, and magnitude penalties
        separability = [1 / (1 + t.sum(t.square(embeddings[id1] - embeddings[id2])))
                        for id2 in ids for id1 in ids if id1 != id2]
        compactness = [t.sum(t.square(logits[labels == id] - embeddings[id]))
                       for id in ids]
        magnitude = [t.sum(t.square(logits[labels == id])) for id in ids]
        separability = t.sum(t.stack(separability))
        compactness = t.sum(t.stack(compactness))
        magnitude = t.sum(t.stack(magnitude))

        # If magnitude is too high, the embeddings will converge to zeros
        loss = (3)*separability + (1)*compactness + (1/100)*magnitude
        return loss, (separability, compactness, magnitude)

    def configure_optimizers(self):
        optimizer = t.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device_param.device), y.to(self.device_param.device)
        images = x
        labels = y.squeeze()
        logits = self(images)
        loss, (separability, compactness, magnitude) = self.loss(labels, logits)
        self.log('train_loss', loss)
        self.log('train_separability', separability)
        self.log('train_compactness', compactness)
        self.log('train_magnitude', magnitude)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(device), y.to(device)
        images = x
        labels = y.squeeze()
        logits = self(images)
        loss, (separability, compactness, magnitude) = self.loss(labels, logits)
        self.log('val_loss', loss)
        self.log('val_separability', separability)
        self.log('val_compactness', compactness)
        self.log('val_magnitude', magnitude)
        return loss


class emgen_dataset(Dataset):
    def __init__(self,
                 path,
                 fnames,
                 labels,
                 image_size=128):
        self.path = path
        self.fnames = fnames
        self.labels = labels
        self.image_size = image_size

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Naeve approach
        if not isinstance(idx, int):
            return [self.__getitem__(i) for i in idx]
        fname = self.fnames[idx]
        image_path = self.path / fname
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = (
            cv2.resize(image, (self.image_size, self.image_size))
            .astype(np.float64)
        )
        image = t.tensor(image, dtype=t.float) / 255

        label = self.labels[idx]
        label = t.tensor(label, dtype=t.int)
        return (image, label)


class emgen_dataloader(pl.LightningDataModule):
    def __init__(self,
                 label_path,
                 data_path='.',
                 batch_size=64,
                 num_workers=8,
                 seed=42):
        super().__init__()

        self.label_path = Path(label_path)
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def setup(self, stage=None):
        # csv should have columns 'file_name' and 'label'
        labels_csv = pd.read_csv(self.label_path)
        ids = labels_csv['file_name'].values
        labels = labels_csv['label'].values
        self.split_data = train_test_split(ids,
                                           labels,
                                           random_state=self.seed)
        x_train, x_val, y_train, y_val = self.split_data
        self.train_dataset = emgen_dataset(self.data_path,
                                           x_train,
                                           y_train)
        self.val_dataset = emgen_dataset(self.data_path,
                                         x_val,
                                         y_val)

    def train_dataloader(self):
        return self._dataloader(self.train_dataset)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset)

    def _dataloader(self, dataset):
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)


class GetMetrics(pl.callbacks.Callback):
    """Callback to record training-time metrics"""
    def __init__(self):
        super().__init__()
        self.history = {}

    def on_train_epoch_end(self, *args):
        self._on_end(*args[:-1], self.history)

    def _on_end(self, trainer, pl_module, history):
        for k in trainer.callback_metrics:
            if k not in history:
                history[k] = []
            history[k].append(trainer.callback_metrics[k])
