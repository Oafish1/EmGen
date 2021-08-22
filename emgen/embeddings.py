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


class emgen_model(pl.LightningModule):
    def __init__(self,
                 input_size=512,
                 num_conv_layers=2):
        super().__init__()

        self.input_size = input_size
        self.num_conv_layers = num_conv_layers
        self.pool_scale = 4

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(self.pool_scale)
        self.flatten = nn.Flatten()

        self.conv_layers = [nn.Conv2d(3 if i==0 else 128*i, 128*(i+1), 5) for i in range(num_conv_layers)]
        self.batchnorm_layers = [nn.BatchNorm2d(128*(i+1)) for i in range(num_conv_layers)]

        calc_dim = self.input_size / (self.pool_scale**num_conv_layers)
        assert calc_dim % 1 == 0, f'input_size must be divisible by \
                                    {self.pool_scale}**{num_conv_layers}'
        dim = (128 * self.num_conv_layers, int(calc_dim), int(calc_dim))
        #self.linear = nn.Linear(prod(dim), 10)230400
        self.linear = nn.Linear(230400, 10)

    def forward(self, x):
        out = x

        for conv, batchnorm in zip(self.conv_layers, self.batchnorm_layers):
            out = conv(out)
            out = self.relu(out)
            out = self.maxpool(out)
            out = batchnorm(out)

        out = self.flatten(out)
        out = self.linear(out)
        return out

    def loss(self, labels, logits):
        ids = np.unique(labels)
        embeddings = {id: t.mean(logits[labels == id], 0)
                      for id in ids}

        # Separability, inverse compactness, magnitude penalty
        separability = [t.sum(t.square(embeddings[id1] - embeddings[id2])) for id2 in ids
                        for id1 in ids]
        compactness = [1/(1 + t.sum(t.square(logits[labels == id] - embeddings[id]))) for id in ids]
        magnitude = [t.sum(t.square(logits[labels == id])) for id in ids]
        separability = t.sum(t.stack(separability))
        compactness = t.sum(t.stack(compactness))
        magnitude = t.sum(t.stack(magnitude))

        loss = (1/10)*separability + (1)*compactness + (1/100)*magnitude
        return loss

    def configure_optimizers(self):
        optimizer = t.optim.AdamW(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x, y = batch
        images = x.view(-1, 3, self.input_size, self.input_size)
        labels = y.view(-1)
        logits = self(images)
        loss = self.loss(labels, logits)
        self.log('train_loss', loss)
        return loss


class emgen_dataset(Dataset):
    def __init__(self, path, image_ids, labels, extension='.jpg'):
        self.path = path
        self.image_ids = image_ids
        self.labels = labels
        self.extension = extension

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_id = str(self.image_ids[idx])
        image_path = self.path / image_id
        image = cv2.imread(str(image_path) + self.extension)
        image = cv2.resize(image,(512, 512)).astype(np.float64)
        image = t.tensor(image, dtype=t.float)

        label = self.labels[idx]
        label = t.tensor(label, dtype=t.int)
        return (image, label)


class emgen_dataloader(pl.LightningDataModule):
    def __init__(self, path='.', extension='.jpg', batch_size=64):
        super().__init__()

        self.path = Path(path)
        self.extension = extension
        self.batch_size = batch_size

    def setup(self, stage=None):
        # csv should have columns 'id' and 'label'
        # images should just have ints and an extension as their names
        labels_csv = pd.read_csv(self.path / 'labels.csv')
        ids = labels_csv['id'].values
        labels = labels_csv['label'].values
        x_train, x_val, y_train, y_val = train_test_split(ids, labels)
        self.train_dataset = emgen_dataset(self.path,
                                           x_train,
                                           y_train,
                                           self.extension)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
