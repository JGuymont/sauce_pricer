from typing import List
import matplotlib.pyplot as plt
import numpy as np


def plot(
        yvalues: List[List[float]],
        colors: List[str] = None,
        legend: List[str] = None,
        xlabel: str = None,
        ylabel: str = None,
        save_path: str = None
) -> None:
    """Plot a list of series of values on the same plot.
    :param yvalues: List of series to plot
    :param colors: List of colors for the curves
    :param legend: List of name for the curve
    :param xlabel: Title of horizontal axis
    :param ylabel: Title of vertical axis
    :param save_path: Path where to save the figure
    """
    for i, y in enumerate(yvalues):
        plt.plot(y, color=colors[i])
    plt.grid()
    plt.legend(legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def scatter_plot(x, y, xlabel, ylabel, save_path, marker='.', show=False):
    max_value = round(max(max(x), max(y)), -1)
    plt.ylim(0, max_value)
    plt.xlim(0, max_value)
    plt.plot([0, max_value], [0, max_value])
    plt.scatter(x, y, marker=marker)
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def plot_train_valid_rmse_loss(train_loss: List[float], valid_loss: List[float], save_path: str) -> None:
    plot([np.sqrt(train_loss), np.sqrt(valid_loss)], ['black', 'red'], ['train loss', 'valid loss'], 'iteration', 'RMSE', save_path)


def scatter_plot_predictions(predictions, targets, save_path):
    scatter_plot(predictions, targets, 'predictions', 'targets', save_path)
