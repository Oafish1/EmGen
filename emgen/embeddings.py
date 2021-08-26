from math import prod
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
import torch as t
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .utilities import conv_output_shape, pool_output_shape


device = t.device("cuda:0" if t.cuda.is_available() else "cpu")


class emgen_model(pl.LightningModule):
    def __init__(self,
                 input_size=128,
                 conv_depth=64,
                 conv_kernel=5,
                 pool_scale=2,
                 num_conv_layers=4,
                 embed_size=50):
        super().__init__()

        self.input_size = input_size
        self.conv_depth = conv_depth
        self.conv_kernel = conv_kernel
        self.pool_scale = pool_scale
        self.num_conv_layers = num_conv_layers
        self.embed_size = embed_size

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(self.pool_scale)
        self.flatten = nn.Flatten()

        self.conv_layers = [nn.Conv2d(3 if i == 0 else self.conv_depth,
                                      self.conv_depth,
                                      self.conv_kernel)
                            for i in range(num_conv_layers)]
        self.batchnorm_layers = [nn.BatchNorm2d(self.conv_depth)
                                 for i in range(num_conv_layers)]
        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.batchnorm_layers = nn.ModuleList(self.batchnorm_layers)

        self.dropout = nn.Dropout(.8)

        dim = (3, input_size, input_size)
        for i in range(num_conv_layers):
            dim = self.conv_depth, *dim[1:]
            dim = conv_output_shape(dim, self.conv_kernel)
            dim = pool_output_shape(dim, self.pool_scale)
        calc_dim = prod(dim)
        self.linear = nn.Linear(calc_dim, self.embed_size)

        self.device_param = nn.Parameter(t.empty(0))
        self.to(device)

    def forward(self, x):
        x = x.to(self.device_param.device)
        out = x.permute(0, 3, 1, 2)
        for conv, batchnorm in zip(self.conv_layers, self.batchnorm_layers):
            out = conv(out)
            out = self.relu(out)
            out = self.maxpool(out)
            out = batchnorm(out)
        out = self.flatten(out)
        out = self.dropout(out)
        out = self.linear(out)
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
        self.log('separability', separability)
        self.log('compactness', compactness)
        self.log('magnitude', magnitude)

        # If magnitude is too high, the embeddings will converge to zeros
        loss = (3)*separability + (1)*compactness + (1/100)*magnitude
        return loss

    def configure_optimizers(self):
        optimizer = t.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device_param.device), y.to(self.device_param.device)
        images = x
        labels = y.squeeze()
        logits = self(images)
        loss = self.loss(labels, logits)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(device), y.to(device)
        images = x
        labels = y.squeeze()
        logits = self(images)
        loss = self.loss(labels, logits)
        self.log('val_loss', loss)
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
        fname = str(self.fnames[idx])
        image_path = self.path / fname
        image = cv2.imread(str(image_path))
        image = (
            cv2.resize(image, (self.image_size, self.image_size))
            .astype(np.float64)
        )
        image = t.tensor(image, dtype=t.float)

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
    def __init__(self):
        super().__init__()
        self.history_train = {}
        self.history_val = {}

    def on_train_epoch_end(self, *args):
        self._on_end(*args[:-1], self.history_train)

    def on_validation_epoch_end(self, *args):
        self._on_end(*args, self.history_val)

    def _on_end(self, trainer, pl_module, history):
        for k in trainer.callback_metrics:
            if k not in history:
                history[k] = []
            history[k].append(trainer.callback_metrics[k])
