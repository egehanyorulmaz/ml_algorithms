import numpy as np


class Distance:
    """
    Consists of the distance metrics that are most commonly used.
    For details, you can check my notes: https://marmalade-cilantro-afb.notion.site/Distance-Metrics-b2a1e27dd0ef4306be13ef71b5d888bd
    """

    def __init__(self):
        self.distance_measures = ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'cosine', 'jaccard']

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

    @staticmethod
    def cosine_distance(x_1, x_2):
        """
        Calculates the cosine distance between two arrays.
        a measure of similarity between two non-zero vectors of an
        inner product space. It is defined as 1 minus the cosine
        of the angle between the vectors. Cosine distance ranges
        from 0 (indicating that the vectors are identical) to 2
        (indicating that the vectors are orthogonal, or maximally dissimilar).
        It is commonly used in natural language processing and
        information retrieval to measure the similarity between two texts or documents.
        """
        X_1 = np.array(x_1)
        X_2 = np.array(x_2)
        return np.dot(X_1, X_2) / (np.linalg.norm(X_1) * np.linalg.norm(X_2))

    @staticmethod
    def chebyshev_distance(x_1, x_2):
        """
        Calculates the chebyshev distance between two arrays
        """
        X_1 = np.array(x_1)
        X_2 = np.array(x_2)
        return np.max(np.abs(X_1 - X_2))

    @staticmethod
    def minkowski_distance(x_1, x_2, p):
        """
        Calculates the minkowski distance between two arrays
        """
        X_1 = np.array(x_1)
        X_2 = np.array(x_2)
        return np.sum(np.abs(X_1 - X_2) ** p) ** (1 / p)

    @staticmethod
    def hamming_distance(x_1, x_2):
        """
        Calculates the hamming distance between two arrays
        """
        X_1 = np.array(x_1)
        X_2 = np.array(x_2)
        return np.sum(X_1 != X_2) / len(X_1)

    @staticmethod
    def jaccard_distance(x_1, x_2):
        """
        Calculates the jaccard distance between two arrays
        """
        X_1 = np.array(x_1)
        X_2 = np.array(x_2)
        return np.sum(np.minimum(X_1, X_2)) / np.sum(np.maximum(X_1, X_2))

    @staticmethod
    def canberra_distance(x_1, x_2):
        """
        Calculates the canberra distance between two arrays.
        The Canberra distance is particularly useful for comparing
        ordinal or categorical data, as it is insensitive to the
        scale of the data and is less sensitive to outliers than
        other distance metrics such as the Euclidean distance.This
        distance metric is less sensitive to extreme values and more robust
        to outliers than other distance metric like Euclidean distance.
        It is widely used in bioinformatics, ecology, and other fields
        where the data may have a lot of zeroes or other extreme values.
        """
        X_1 = np.array(x_1)
        X_2 = np.array(x_2)
        return np.sum(np.abs(X_1 - X_2) / (np.abs(X_1) + np.abs(X_2)))

    @staticmethod
    def braycurtis_distance(x_1, x_2):
        """
        Calculates the braycurtis distance between two arrays.
        The Bray-Curtis distance is particularly useful for comparing data
        that has been normalized or standardized.
        """
        X_1 = np.array(x_1)
        X_2 = np.array(x_2)
        return np.sum(np.abs(X_1 - X_2)) / np.sum(np.abs(X_1 + X_2))

    @staticmethod
    def correlation_distance(x_1, x_2):
        """
        Calculates the correlation distance between two arrays
        """
        X_1 = np.array(x_1)
        X_2 = np.array(x_2)
        return 1 - np.corrcoef(X_1, X_2)[0, 1]

    @staticmethod
    def mahalanobis_distance(x_1, x_2, cov):
        """
        Calculates the mahalanobis distance between two arrays
        """
        X_1 = np.array(x_1)
        X_2 = np.array(x_2)
        return np.sqrt(np.dot(np.dot((X_1 - X_2), np.linalg.inv(cov)), (X_1 - X_2).T))

    @staticmethod
    def weighted_minkowski_distance(x_1, x_2, p, w):
        """
        Calculates the weighted minkowski distance between two arrays
        """
        X_1 = np.array(x_1)
        X_2 = np.array(x_2)
        return np.sum(np.abs(X_1 - X_2) ** p * w) ** (1 / p)

    @staticmethod
    def seuclidean_distance(x_1, x_2, V):
        """
        Calculates the seuclidean distance between two arrays
        """
        X_1 = np.array(x_1)
        X_2 = np.array(x_2)
        return np.sqrt(np.sum((X_1 - X_2) ** 2 / V))

    @staticmethod
    def sqeuclidean_distance(x_1, x_2):
        """
        Calculates the squared euclidean distance between two arrays
        """
        X_1 = np.array(x_1)
        X_2 = np.array(x_2)
        return np.sum((X_1 - X_2) ** 2)

    @staticmethod
    def wminkowski_distance(x_1, x_2, p, w):
        """
        Calculates the weighted minkowski distance between two arrays
        """
        X_1 = np.array(x_1)
        X_2 = np.array(x_2)
        return np.sum(np.abs(X_1 - X_2) ** p * w) ** (1 / p)

    @staticmethod
    def cityblock_distance(x_1, x_2):
        """
        Calculates the cityblock distance between two arrays
        """
        X_1 = np.array(x_1)
        X_2 = np.array(x_2)
        return np.sum(np.abs(X_1 - X_2))


if __name__ == '__main__':
    X_1 = [5, 3, 2, 10, 15]
    X_2 = [6, 5, 6, 3, 4]
    print("Euclidean Distance: ", Distance.euclidean_distance(X_1, X_2))
    print("Manhattan Distance: ", Distance.manhattan_distance(X_1, X_2))
    print("Cosine Distance: ", Distance.cosine_distance(X_1, X_2))
    print("Chebyshev Distance: ", Distance.chebyshev_distance(X_1, X_2))

    # generalized version of the distance. if p=2, it is euclidean distance, if p=1 it is manhattan distance
    print("Minkowski Distance: ", Distance.minkowski_distance(X_1, X_2, 3))

    print("Hamming Distance: ", Distance.hamming_distance(X_1, X_2))
    print("Jaccard Distance: ", Distance.jaccard_distance(X_1, X_2))
    print("Canberra Distance: ", Distance.canberra_distance(X_1, X_2))
