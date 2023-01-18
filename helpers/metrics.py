import numpy as np


class Distance:
    """
    Consists of the distance metrics that are most commonly used.
    For details, you can check my notes: https://marmalade-cilantro-afb.notion.site/Distance-Metrics-b2a1e27dd0ef4306be13ef71b5d888bd
    """

    def __init__(self):
        self.distance_measures = ['euclidean', 'manhattan']

    @staticmethod
    def euclidean_distance(x_1, x_2):
        """
        Calculates the euclidean distance between two arrays
        """
        X_1 = np.array(x_1)
        X_2 = np.array(x_2)
        return np.sqrt(np.sum((X_1 - X_2) ** 2))

    @staticmethod
    def manhattan_distance(x_1, x_2):
        """
        Calculates the manhattan distance between two arrays
        """
        X_1 = np.array(x_1)
        X_2 = np.array(x_2)
        return np.sum((np.abs(X_1 - X_2)))


if __name__ == '__main__':
    X_1 = [5, 3, 2, 10, 15]
    X_2 = [6, 5, 6, 3, 4]
    distance = Distance.manhattan_distance(X_1, X_2)
    print(distance)
