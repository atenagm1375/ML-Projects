import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


class KMeans:
    def __init__(self, k=2):
        self.k = k
        self.centroids = []

    def fit(self, data, max_iter=100):
        self.centroids = np.zeros((self.k, data.shape[1]))
        for i in range(self.k):
            rand = np.random.randint(0, data.shape[0] - 1)
            self.centroids[i][:] = data.iloc[rand, :]
        old_centroids = deepcopy(self.centroids)
        y = np.zeros((data.shape[0],), dtype=int)
        for i in range(max_iter):
            distances = [self._euclidean_distance(
                data.values, c) for c in self.centroids]
            y = pd.DataFrame(distances).idxmin(axis=0)
            for k in range(self.k):
                self.centroids[k] = np.mean(
                    data.loc[y == k, :], axis=0).values
            if np.array_equal(old_centroids, self.centroids):
                break
            old_centroids = deepcopy(self.centroids)
        return self, y, self.score(data, y)

    def score(self, data, labels):
        score = 0
        score = np.sum(self._euclidean_distance(
            data.values, self.centroids[labels]))
        return score

    def _euclidean_distance(self, x1, x2):
        return np.linalg.norm(x1 - x2, axis=1)


def load_data(file):
    return pd.read_csv(file, names=['x1', 'x2'])


def plot_data(x, y, centers, name):
    plt.figure()
    plt.scatter(x.iloc[:, 0], x.iloc[:, 1], c=y)
    plt.scatter(centers.iloc[:, 0], centers.iloc[:, 1], marker="x", color='r')
    plt.show()
    k = centers.shape[0]
    # plt.savefig(name)


def find_elbow(lst):
    diff = []
    ans = []
    for i in range(len(lst) - 1):
        diff.append(lst[i + 1] - lst[i])
    for i in range(len(diff) - 1):
        ans.append(diff[i] / diff[i + 1])
    return np.argmax(ans) + 1


def __main__():
    data = load_data('cluster.csv')
    k_range = list(range(2, 16))
    scores = []
    labels = []
    centroids = []
    for k in k_range:
        print('k =', k)
        best = np.inf
        best_y = []
        best_c = []
        for i in range(200):
            model = KMeans(k)
            model, y, score = model.fit(data)
            if score < best:
                best = score
                best_y = y
                best_c = model.centroids
        print("best:", best)
        scores.append(best)
        labels.append(best_y)
        centroids.append(best_c)
        # plot_data(data, best_y, pd.DataFrame(best_c), f'{k}-means.png')
    plt.figure()
    plt.plot(k_range, scores)
    # plt.savefig("scores.png")
    plt.show()
    ind = find_elbow(scores)
    print("best k =", ind + 2)
    plot_data(data, labels[ind], pd.DataFrame(
        centroids[ind]), f'best_cluster.png')


__main__()
