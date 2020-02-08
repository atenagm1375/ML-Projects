import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from types import SimpleNamespace


class Perceptron:
    def __init__(self, algorithm="batch"):
        self._coef = []
        self._n_feats = 0
        self.algorithm = algorithm

    def fit(self, X, Y, theta, eta=None):
        self._n_feats = X.shape[1]
        y = np.where(Y == 0, -1, Y)
        x = np.insert(X, 0, 1, axis=1)
        self._coef = np.zeros((self._n_feats + 1, ))
        if self.algorithm == "batch":
            while True:
                pred = self.predict(x, False)
                f = np.sign(pred)

                miss = np.zeros(self._n_feats + 1)
                for i in range(X.shape[0]):
                    if f[i] != y[i]:
                        miss += y[i] * x[i, :]

                self._coef = self._coef + eta * miss
                if all(x < theta for x in eta * miss):
                    break
        elif self.algorithm == "pocket":
            if isinstance(theta, float):
                raise ValueError(
                    "theta should be integer in pocket algorithm!")
            for t in range(theta):
                i = t % X.shape[0]
                pred = np.sign(np.dot(self._coef.T, x[i, :]))
                if pred != y[i]:
                    w = self._coef + y[i] * x[i, :]
                    if np.sum(self.predict(x, False) != y) > np.sum(np.sign(np.matmul(w.T, x.T).reshape((x.shape[0],)))):
                        self._coef = w
        else:
            raise ValueError("algorithm can either be batch or pocket!")

        return self

    def predict(self, x, user=True):
        if x.shape[1] == self._n_feats + 1:
            pred = np.sign(np.matmul(self._coef.T, x.T).reshape((x.shape[0],)))
        else:
            pred = np.sign(np.matmul(self._coef.T, np.insert(
                x, 0, 1, axis=1).T).reshape((x.shape[0],)))
        return np.where(pred == -1, 0, pred) if user else pred

    def score(self, x, y_true):
        pred = self.predict(x)
        return 1 - np.sum((pred - y_true) ** 2) / len(y_true)


def load_data(train_file, test_file):
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    x_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]

    x_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]

    return {
        "x_train": x_train.values,
        "y_train": y_train.values,
        "x_test": x_test.values,
        "y_test": y_test.values
    }


def plot(X, Y, clf=None, h=0.01, fname=None):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    fig, ax = plt.subplots()
    if clf is not None:
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=plt.cm.Paired)
        ax.axis('off')

    ax.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
    if fname is not None:
        plt.savefig(fname)
    else:
        plt.show()


def __main__():
    data = SimpleNamespace(**load_data("train.csv", "test.csv"))
    # plot(data.x_train, data.y_train)
    # plot(data.x_test, data.y_test)
    print("BATCH PERCEPTRON:")
    clf = Perceptron()  # batch perceptron
    clf.fit(data.x_train, data.y_train, 0.01, 0.001)
    plot(data.x_train, data.y_train, clf, fname="batch-train.png")
    print("TRAIN ACCURACY =", clf.score(data.x_train, data.y_train))
    # train_pred = clf.predict(data.x_train)
    # print(train_pred)
    plot(data.x_test, data.y_test, clf, fname="batch-test.png")
    print("TEST ACCURACY =", clf.score(data.x_test, data.y_test))
    # test_pred = clf.predict(data.x_test)
    # print(test_pred)

    print("POCKET ALGORITHM:")
    clf = Perceptron(algorithm="pocket")  # batch perceptron
    clf.fit(data.x_train, data.y_train, 10000)
    plot(data.x_train, data.y_train, clf, fname="pocket-train.png")
    print("TRAIN ACCURACY =", clf.score(data.x_train, data.y_train))
    # train_pred = clf.predict(data.x_train)
    # print(train_pred)
    plot(data.x_test, data.y_test, clf, fname="pocket-test.png")
    print("TEST ACCURACY =", clf.score(data.x_test, data.y_test))
    # test_pred = clf.predict(data.x_test)
    # print(test_pred)


__main__()
