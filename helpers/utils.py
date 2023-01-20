import pickle


def save_clusters(clusters, cluster_file='clusters.pkl'):
    """
    Save the clusters to a pickle file
    """
    with open(cluster_file, 'wb') as f:
        pickle.dump(clusters, f)


def load_clusters(cluster_file='clusters.pkl'):
    """
    Load the clusters from the pickle file
    """
    with open(cluster_file, 'rb') as f:
        clusters = pickle.load(f)
    return clusters
