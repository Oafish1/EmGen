from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def pca_visualize(embeddings,
                  labels,
                  legend_loc: Union[str, int] = 'best'):
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
                  log_plot: bool = True
                  ):
    if keys is None:
        keys = history.keys()
    if iter_key is not None:
        x = history[iter_key]
    for k in keys:
        y = history[k]
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
