import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DBSCAN:
    def __init__(self, epsilon, min_samples):
        self.epsilon = epsilon
        self.min_samples = min_samples
        self.clustered_points = []

    def neighborhood(self, point, data):
        """
        Find all the points within epsilon of the point
        :param point:
        :param data:
        :return:
        """
        neighbors = []
        for idx, p in enumerate(data):
            if np.linalg.norm(point - p) < self.epsilon:
                neighbors.append(idx)
        return neighbors

    def core_point(self, point, data):
        """
        A point is a core point if it has at least min_samples within epsilon
        """
        neighbors = self.neighborhood(point, data)
        return len(neighbors) >= self.min_samples

    def expand_cluster(self, point, data, cluster):
        """
        Expand the cluster by adding all the points within epsilon
        """
        cluster.append(point)
        neighbors = self.neighborhood(point, data)
        for neighbor in neighbors:
            if neighbor in self.clustered_points:
                continue
            neighbor_point = data[neighbor]
            if (not self.check_point(neighbor_point, cluster)) & (neighbor not in self.clustered_points):
                cluster.append(neighbor_point)
                # blacklisting the clustered points
                self.clustered_points = np.append(self.clustered_points, int(neighbor))
                if self.core_point(neighbor, data):
                    self.expand_cluster(neighbor, data, cluster)
        assert len(self.clustered_points) == len(set(self.clustered_points)), "Duplicate points in cluster"
        print(len(self.clustered_points))
        return cluster

    @staticmethod
    def check_point(point, cluster):
        for clustered_point in cluster:
            if np.array_equal(point, clustered_point):
                return True
        return False

    def delete_clustered_points(self, data):
        """
        Delete the clustered points from the data
        """
        for clustered_point in self.clustered_points:
            try:
                data = np.delete(data, clustered_point, axis=0)
            except IndexError:
                pass
        return data

    def dbscan(self, data):
        """
        Run the DBSCAN algorithm
        """
        clusters = []
        for point in data:
            is_core_point = self.core_point(point, data)
            if is_core_point:
                cluster = []
                cluster = self.expand_cluster(point, data, cluster)
                clusters.append(cluster)

            # remove the clustered points from the data
            data = self.delete_clustered_points(data)
        return clusters

    def plot_clusters(self, clusters, data):
        import matplotlib.pyplot as plt
        for cluster in clusters:
            for point in cluster:
                plt.scatter(data[point][0], data[point][1])
        plt.show()


def main():
    # create an instance of the class
    epsilon = .5
    min_samples = 20
    dbscan = DBSCAN(epsilon, min_samples)
    # modify the data to be used in the functions
    data = pd.read_csv('data/xclara.csv')
    data = np.array(data)

    data = StandardScaler().fit_transform(data)

    clusters = dbscan.dbscan(data)
    dbscan.plot_clusters(clusters, data)


if __name__ == '__main__':
    main()
