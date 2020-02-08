import numpy as np
from scipy.io import loadmat
from scipy.linalg import eig
import matplotlib.pyplot as plt


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.pc_ = []
        self.__mean = 0
        # self.reconstruction_error = np.inf

    def fit(self, x, svd=True):
        x = x.astype('float64')[:, :]
        n_samples, n_feats = x.shape
        self.__mean = np.mean(x, axis=0, keepdims=True)
        x_tilde = x - self.__mean
        if not svd:
            C = np.dot(x_tilde.T, x_tilde) / (n_samples - 1)
            _, vectors = eig(C)
        else:
            u, s, v = np.linalg.svd(x_tilde.T)
            vectors = u
        self.pc_ = vectors[:, :self.n_components]
        return self

    def transform(self, x):
        x_prime = np.dot(x, self.pc_)
        return self.__mean + np.dot(self.pc_, x_prime.T).T

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


def load_split_data(file_name, ratio=0.7):
    mat = loadmat(file_name)['faces'].T
    l = int(ratio * mat.shape[0])
    return mat[:l], mat[l:]


def show_image(before, after, proc="train"):
    columns = 7
    rows = 10
    per_fig = columns * rows
    for i, a in zip(range(before.shape[0] // per_fig), range(after.shape[0] // per_fig)):
        fig = plt.figure(figsize=(100, 100))
        fig.suptitle("{} Images {}-{} Before PCA".format(proc, i *
                                                         per_fig + 1, (i + 1) * per_fig))
        for j in range(1, per_fig + 1):
            if i * per_fig + j - 1 >= before.shape[0]:
                break
            fig.add_subplot(rows, columns, j)
            plt.imshow(before[i * per_fig + j - 1].reshape((64, 64)).T)
        # fig.savefig("{} Images {}-{} Before PCA.png".format(proc, i *
        #                                                     per_fig + 1, (i + 1) * per_fig))
        plt.show()

        fig = plt.figure(figsize=(100, 100))
        fig.suptitle("{} Images {}-{} After PCA".format(proc, a *
                                                        per_fig + 1, (a + 1) * per_fig))
        for j in range(1, per_fig + 1):
            if a * per_fig + j - 1 >= after.shape[0]:
                break
            fig.add_subplot(rows, columns, j)
            plt.imshow(after[a * per_fig + j - 1].reshape((64, 64)).T)
        # fig.savefig("{} Images {}-{} After PCA.png".format(proc, a *
        #                                                    per_fig + 1, (a + 1) * per_fig))
        plt.show()


def __main__():
    train, test = load_split_data("faces.mat")
    n = int(input("Enter number of components:"))
    print("NUMBER OF COMPONENTS =", n)
    pca = PCA(n)
    train_imgs = pca.fit_transform(train)
    test_imgs = pca.transform(test)
    print("TRAIN IMAGES:")
    show_image(train, train_imgs)
    print("TEST IMAGES:")
    show_image(test, test_imgs)


__main__()
