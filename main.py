import sys
import numpy as np
import matplotlib.pyplot as plt


# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


# K-means clustering algorithm
def KMeans(data, k, max_iterations=100, label="A"):
    # Initialize centroids by randomly selecting k data points
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]

    for n in range(max_iterations):
        # Create an array to store the cluster assignments for each data point
        cluster_assignments = np.zeros(data.shape[0])

        # Assign each data point to the nearest centroid
        for i in range(data.shape[0]):
            distances = [euclidean_distance(data[i], centroid) for centroid in centroids]
            cluster_assignments[i] = np.argmin(distances)

        # Update centroids based on the mean of data points in each cluster
        for j in range(k):
            cluster_points = data[cluster_assignments == j]
            if len(cluster_points) > 0:
                centroids[j] = np.mean(cluster_points, axis=0)

        # Plot the data points and centroids
        if n < 5:
            plt.scatter(data[:, 0], data[:, 1], c=cluster_assignments, cmap='viridis')
            plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=100, label='Centroids')
            plt.legend()
            plt.title('K-means Clustering')
            plt.savefig("./figures/"+ label + str(n + 1) + ".png")
            plt.show()

    return cluster_assignments, centroids



if __name__ == "__main__":
    if sys.version_info[0:2] != (3, 11):
        raise Exception("Requires python 3.11")

    np.random.seed(0)

    # Create four distinct clusters
    data = np.concatenate([np.random.normal(loc=[2, 2], scale=1, size=(100, 2)),
                           np.random.normal(loc=[8, 2], scale=1, size=(100, 2)),
                           np.random.normal(loc=[2, 8], scale=1, size=(100, 2)),
                           np.random.normal(loc=[8, 8], scale=1, size=(100, 2))])

    # Add some noise between the clusters
    noise = np.random.normal(loc=[5, 5], scale=1, size=(40, 2))
    data = np.vstack((data, noise))


    # Initialize and fit the K-means model
    cluster_assignments, centroids = KMeans(data=data, k=4, label="A")

    # Plot the data points and centroids
    plt.scatter(data[:, 0], data[:, 1], c=cluster_assignments, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=100, label='Centroids')
    plt.legend()
    plt.title('K-means Clustering (using scikit-learn)')
    plt.savefig("./figures/A_clustering.png")
    plt.show()

    cluster_assignments, centroids = KMeans(data=data, k=3, label="B")

    # Plot the data points and centroids
    plt.scatter(data[:, 0], data[:, 1], c=cluster_assignments, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=100, label='Centroids')
    plt.legend()
    plt.title('K-means Clustering (using scikit-learn)')
    plt.savefig("./figures/B_clustering.png")
    plt.show()