import itertools

import numpy as np
import matplotlib.pyplot as plt

from ..global_variables import CMAP

def plot_matrix(matrix, names = None, legend = None):
    """
    Returns a matplotlib figure containing the plotted matrix.
    Args:
    matrix (array, shape = [n, n]): a matrix
    names (array, shape = [n]): String names of the rows/lines
    """
    
    if names is None:
        names = np.arange(matrix.shape[0]).astype(str)
        
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(matrix, interpolation='nearest', cmap=plt.get_cmap(CMAP + "_r"))
    plt.colorbar()
    
    if len(names) <= 16:
        tick_marks = np.arange(len(names))
        plt.xticks(tick_marks, names, rotation=45)
        plt.yticks(tick_marks, names)

        # Normalize the confusion matrix.
        normed_matrix = np.around(matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis],
                              decimals=2)

        # Use white text if squares are dark; otherwise black.
        threshold = matrix.max() / 2.
        for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
            color = "white" if matrix[i, j] > threshold else "black"
            plt.text(j, i, normed_matrix[i, j],
                     horizontalalignment="center",
                     color=color)

        plt.tight_layout()
    
    if legend is None:
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title("Matrix")
    else:
        plt.ylabel(legend["y"])
        plt.xlabel(legend["x"])
        plt.title(legend["title"])
    return figure