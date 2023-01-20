import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler

from helpers.utils import save_clusters, load_clusters
from helpers.plots import PlotClusters


class DBSCAN:
    def __init__(self, epsilon, min_samples):
        self.epsilon = epsilon
        self.min_samples = min_samples
        self.clustered_points = {'index': [], 'point': []}
        self.clusters = []

    def neighborhood(self, point, X):
        """
        Find all the points within epsilon of the point
        """
        neighbors = []
        indices = []
        for idx, p in enumerate(X):
            if np.linalg.norm(point - p) < self.epsilon:
                neighbors.append(p)
                indices.append(idx)
        return indices, neighbors

    def core_point(self, point, X):
        """
        A point is a core point if it has at least min_samples within epsilon
        """
        indices, neighbors = self.neighborhood(point, X)
        return len(neighbors) >= self.min_samples

    def insert_clustered_point(self, point, point_index):
        """
        Insert the clustered point into the list
        """
        self.clustered_points['point'].append(point)
        self.clustered_points['index'].append(point_index)

    def expand_cluster(self, point, point_index, X, cluster):
        """
        Expand the cluster by adding all the points within epsilon
        """
        if point_index not in cluster:
            # not visited
            self.insert_clustered_point(point, point_index)
        else:
            # already visited
            return cluster

        indices, neighbors = self.neighborhood(point, X)
        for neighbor_idx, neighbor_point in zip(indices, neighbors):
            if self.check_point(neighbor_point, self.clustered_points['point']):
                # if the point is already clustered, skip it
                continue
            else:
                # the point is not already clustered
                self.insert_clustered_point(neighbor_point, neighbor_idx)
                cluster.append(neighbor_idx)
                if self.core_point(neighbor_point, X):
                    # have enough power to change others
                    cluster = self.expand_cluster(neighbor_point, neighbor_idx, X, cluster)

        assert len(self.clustered_points['index']) == len(
            set(self.clustered_points['index'])), "Duplicate points in cluster"
        print(len(self.clustered_points['index']))
        return cluster

    @staticmethod
    def check_point(point, cluster):
        for clustered_point in cluster:
            if np.array_equal(point, clustered_point):
                return True
        return False

    def dbscan(self, X):
        """
        Run the DBSCAN algorithm
        """
        clusters = []
        for idx, point in enumerate(X):
            if idx in self.clustered_points['index']:
                continue

            is_core_point = self.core_point(point, X)
            if is_core_point:
                cluster = self.expand_cluster(point=point, point_index=idx, X=X, cluster=[])
                clusters.append(cluster)

        # combine clusters that have less number of elements than min_samples and modify the clusters accordingly
        self.clusters = self.combine_clusters(clusters)
        return self.clusters

    def combine_clusters(self, clusters):
        """
        Combine clusters that have less number of elements than min_samples
        """
        combined_clusters = []
        anomaly_clusters = []
        for cluster in clusters:
            if len(cluster) < self.min_samples:
                anomaly_clusters += cluster
            else:
                combined_clusters.append(cluster)

        combined_clusters.append(anomaly_clusters)
        return combined_clusters


def run_clustering(X, args=None):
    """
    Run the DBSCAN algorithm and save the clusters to a pickle file.
    """
    if args is None:
        args = {'epsilon': 0.2, 'min_samples': 5}
    dbscan = DBSCAN(epsilon=args['epsilon'], min_samples=args['min_samples'])
    X = np.array(X)
    X = StandardScaler().fit_transform(X)
    clusters = dbscan.dbscan(X)
    save_clusters(clusters)


def plot_clustering(X):
    """
    Plot the clusters from the pickle file
    """
    clusters = load_clusters()
    plotcluster = PlotClusters()
    plotcluster.plot_clusters(clusters, X)


if __name__ == '__main__':
    data = pd.read_csv('data/xclara.csv')
    run_clustering(data, args={'epsilon': 1.5, 'min_samples': 20})
    plot_clustering(data)
