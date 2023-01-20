import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import math
import random
import numpy as np
import pandas as pd

class PlotClusters:
    """
    Plots the clusters using matplotlib.
    """
    def __init__(self):
        self.colors = self.get_available_colors(mcolors.CSS4_COLORS)

    @staticmethod
    def get_available_colors(colors, sort_colors=True):
        """
        Creates
        :param colors:
        :param ncols:
        :param sort_colors:
        :return:
        """
        # Sort colors by hue, saturation, value and name.
        if sort_colors is True:
            names = sorted(
                colors, key=lambda c: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(c))))
        else:
            names = list(colors)
        return names

    def plot_clusters(self, clusters, data, title=None, save=False, save_path=None):
        """
        Plots the clusters using matplotlib.
        """


        color_options = self.colors
        unique_labels = range(len(clusters))
        color_palette = random.sample(color_options, len(unique_labels))

        plt.figure()
        plt.title(title)

        for (label, color) in zip(unique_labels, color_palette):
            xy = data.iloc[clusters[label]]
            if type(xy) is not np.ndarray:
                xy = np.array(xy)

            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=color, markersize=10)

        plt.show()




