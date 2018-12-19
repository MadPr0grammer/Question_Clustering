from sklearn.cluster import KMeans
import numpy as np

def get_labels(X, number_of_cluster):
    kmeans = KMeans(n_clusters=number_of_cluster, random_state=0).fit(X)
    return kmeans.labels_