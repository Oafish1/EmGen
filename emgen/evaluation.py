from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch as t


def pca_visualize(embeddings,
                  labels,
                  legend_loc: Union[str, int] = 'best'):
    """Visualize embeddings with PCA"""
    # Note: Very sensitive to sample sizes for each embedding type
    pca = PCA(n_components=2)
    pca.fit(embeddings.detach().numpy())
    embeddings_pca = pca.transform(embeddings.detach().numpy())
    for i, (label, count) in enumerate(zip(*np.unique(labels, return_counts=True))):
        idx = labels == label
        plt.scatter(embeddings_pca[idx, 0],
                    embeddings_pca[idx, 1],
                    color='C'+str(i),
                    label=f'Label: {label.item()}, Count: {count}')
    plt.title('PCA Plot')
    plt.legend(loc=legend_loc)


def plot_training(history,
                  keys: Optional[list] = None,
                  title: Optional[str] = None,
                  iter_key: Optional[Any] = None,
                  log_plot: bool = True,
                  prefix: str = None):
    """Plot training progress"""
    if keys is None:
        keys = history.keys()
    if prefix is not None:
        keys = [k for k in keys if k.startswith(prefix)]
    if iter_key is not None:
        x = history[iter_key]
    for k in keys:
        y = [e.item() for e in history[k]]
        if iter_key is None:
            x = range(len(y))
        plt.plot(x, y, label=k)

    plt.xlabel('Iteration')
    plt.ylabel('Error Function')
    if log_plot:
        plt.yscale('log')

    if title is None:
        title = 'Error over Time'
    plt.title(title)

    plt.legend(loc='upper right')


def equal_samples_labels_images(labels, instances=20):
    """Returns idx of `entries` sample per label, if possible"""
    samples = (pd.DataFrame({'labels': labels})
                 .sort_values('labels')
                 .groupby('labels')
                 .head(instances))
    labels, labels_counts = np.unique(samples['labels'], return_counts=True)
    samples = [
        i for i, s in samples.iterrows()
        if labels_counts[np.argwhere(s['labels'] == labels)] >= instances
    ]
    return samples


def plot_embeddings(model, dataset, sets=5, instances=20):
    """Perform sample embedding on a dataset and plot the result with PCA"""
    samples = equal_samples_labels_images(dataset.labels, instances=instances)
    batch = [None, None]
    for idx in samples[:instances*sets]:
        datapoint = dataset[idx]
        if batch[0] is None:
            batch = [*datapoint]
            batch[0] = batch[0].unsqueeze(0)
            batch[1] = batch[1].reshape((1))
        else:
            batch[0] = t.cat([batch[0], datapoint[0].unsqueeze(0)])
            batch[1] = t.cat([batch[1], datapoint[1].reshape((1))])

    embeddings = model(batch[0])
    labels = batch[1]
    pca_visualize(embeddings, labels)


def plot_sample_images(dataset, dim=(3, 5)):
    """Plot a sample of the dataset images"""
    for i in range(dim[0]):
        for j in range(dim[1]):
            idx = i*dim[1] + j
            plt.subplot(*dim, idx + 1)
            image, label = dataset[idx]
            plt.title(label.item())
            plt.imshow(image)
