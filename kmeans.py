import numpy as np
import pandas as pd
from helpers.metrics import Distance


def count_features(data):
    if type(data) == pd.DataFrame:
        return np.array(data).shape[1]
    elif type(data) == np.array:
        return data.shape[1]


def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


def initialize_centroids(data, N):
    candidate_centroid_idx = np.random.randint(data.shape[0], size=N)
    return data[candidate_centroid_idx, :]


# coding kmeans from scratch
N = 3

data = pd.read_csv('data/xclara.csv')

X = np.array(data)
C_old = initialize_centroids(X, N)
C = C_old.copy()
point_assignments = np.zeros(len(X))
print(C)

while True:
    point_assignments = np.zeros(len(X))
    for idx1, point in enumerate(X):
        distance_to_centroid = np.zeros(N)
        for idx2, centroid in enumerate(C):
            distance_to_centroid[idx2] = Distance.euclidean_distance(point, centroid)
        assignment = np.argmin(distance_to_centroid)
        point_assignments[idx1] = assignment

    # update the centroids
    for idx, centroid in enumerate(C):
        indices = np.where(point_assignments == idx)
        sliced_X = np.array(X[indices, :])
        C[idx] = np.mean(sliced_X, axis=1)

    print("New centroids are: ")
    print(C)

    if dist(C, C_old, None) == 0:
        break
    else:
        C_old = C.copy()


# repeat the process until the points don't change its assignment
# pick random points with N
# assign the points to the closest centroid
# update the position of the centroid
