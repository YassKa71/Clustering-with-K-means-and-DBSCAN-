# Ruspini dataset

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math
import numpy as np
import copy

df = pd.read_csv("C:/Users/kaddy/Downloads/ruspini/ruspini.csv", sep=",")

"""The data is suited for KMeans algorithm : isotropics clusters. 4 clusters"""

# _________________________________K-Means algorithm without using sklearn____________________________________________

def raw_KM(K):
    converge = False
    # generating random initial centroïds
    centroids = df.sample(K)
    centroids = centroids[["x", "y"]].to_numpy()

    while converge is False:
        # calculating distances between points and each centroïd (generating array of size (N,K,1))
        points = df[["x","y"]].values[:, np.newaxis]
        distances = np.linalg.norm(points - centroids, axis=2)

        # assigning each point to a cluster (labels array)
        labels = np.argmin(distances, axis=1)

        # compute the mean of points in each cluster to generate new centroïds until it converges
        new_centroids = np.array([df[labels == k][["x","y"]].mean(axis=0) for k in range(K)])
        if np.array_equal(new_centroids, centroids):
            converge = True
        else:
            centroids = new_centroids

    return centroids


df.plot(kind = 'scatter', x="x", y="y", label="data")
centroids = raw_KM(4)
plt.scatter(centroids[: , 0], centroids[:, 1], label='Centroids', marker='^', color= 'red')
plt.legend()
plt.show()







