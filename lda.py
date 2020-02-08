import pandas as pd
import numpy as np

from types import SimpleNamespace


class LDAClassifier:
    def __init__(self):
        self._coef = []

    def fit(self, x, y):
        mu1 = np.mean(x[y == 1], axis=0)
        mu2 = np.mean(x[y == 0], axis=0)
        mu_diff = mu1 - mu2
        s1 = np.cov(x[y == 1].T)
        s2 = np.cov(x[y == 0].T)
        s_w = s1 + s2
        self._coef = np.matmul(np.linalg.inv(s_w), mu_diff)
        return self

    def predict(self, x):
        return np.dot(x, self._coef) >= 0

    def score(self, x, y_true):
        return 1 - np.sum((self.predict(x) - y_true) ** 2) / x.shape[0]


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


def __main__():
    data = SimpleNamespace(**load_data("train.csv", "test.csv"))
    clf = LDAClassifier()
    clf.fit(data.x_train, data.y_train)
    print("TRAIN ACCURACY =", clf.score(data.x_train, data.y_train))
    # train_pred = clf.predict(data.x_train)
    # print(train_pred)
    print("TEST ACCURACY =", clf.score(data.x_test, data.y_test))
    # test_pred = clf.predict(data.x_test)
    # print(test_pred)


__main__()
