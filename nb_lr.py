import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def load_data(file_name, col_names):
    df = pd.read_csv(file_name, names=col_names, header=0)
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return train_test_split(x, y, shuffle=False, test_size=0.2)


def preprocess_data(x_train, x_test, y_train, **params):
    cat_cols = x_train.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        x_train[col] = le.fit_transform(x_train[col])
        x_test[col] = le.transform(x_test[col])
    if 'scale' in params:
        if params['scale']:
            scaler = MinMaxScaler()
            x_train.loc[:, :] = scaler.fit_transform(x_train)
            x_test.loc[:, :] = scaler.transform(x_test)
    if 'k_select' in params:
        from sklearn.feature_selection import mutual_info_classif, SelectKBest
        vt = SelectKBest(mutual_info_classif, k=params['k_select'])
        x_train = pd.DataFrame(vt.fit_transform(x_train, y_train))
        x_test = pd.DataFrame(vt.transform(x_test))
    return x_train, x_test


class NaiveBayesClassifier:
    def __init__(self):
        self.summaries = {}
        self.class_prob = {}

    def fit(self, x, y):
        for label in y.unique():
            self.summaries[label] = x.loc[y == label].describe(
            ).loc[["mean", "std", "count"], :]
            self.class_prob[label] = self.summaries[label].loc["count",
                                                               x.columns[0]] / len(x)
        return self

    def predict(self, x):
        probabilities = {}
        labels = list(self.summaries.keys())
        for label in labels:
            probabilities[label] = self.class_prob[label] * np.ones(x.shape[0])
            mean = self.summaries[label].loc["mean", :]
            std = self.summaries[label].loc["std", :]
            probabilities[label] *= np.product((1 / (np.sqrt(2 * np.pi) * std)) *
                                               np.exp(-((x - mean)**2 / (2 * std**2))), axis=1)
        y_pred = probabilities[labels[0]] > probabilities[labels[1]]
        y_pred = y_pred.replace([True, False], [labels[0], labels[1]])
        return y_pred, probabilities

    def error(self, x, y_true):
        pred, probs = self.predict(x)
        e = 0
        for i in range(len(y_true)):
            if pred.iloc[i] != y_true.iloc[i]:
                e += probs[pred.iloc[i]].iloc[i]
        return e

    def score(self, x, y_true):
        return accuracy_score(y_true, self.predict(x)[0])


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression:
    def __init__(self):
        self._coef = []
        self.history = []

    def compute_cost(self, X, y):
        m = len(y)
        h = _sigmoid(X @ self._coef)
        epsilon = 1e-5
        cost = (1 / m) * (((-y).T @ np.log(h + epsilon)) -
                          ((1 - y).T @ np.log(1 - h + epsilon)))
        return cost[0][0]

    def fit(self, x, y, lr=0.1, iterations=1000):
        self._coef = np.random.rand(x.shape[1], 1)
        x = x.values
        n = len(y)
        y_new = y.replace([">50K", "<=50K"], [1, 0]).values.reshape((n, 1))
        for i in range(iterations):
            self._coef = self._coef - \
                (lr / n) * (x.T @ (_sigmoid(x @ self._coef) - y_new))
            # print(self.compute_cost(x, y_new))
            self.history.append(self.compute_cost(x, y_new))
        return self

    def error(self, x, y_true):
        return self.compute_cost(x, y_true.replace([">50K", "<=50K"], [1, 0]).values.reshape((len(y_true), 1)))

    def predict(self, x):
        pred = np.round(_sigmoid(x @ self._coef))
        return pred.replace([1, 0], [">50K", "<=50K"])

    def score(self, x, y_true):
        return accuracy_score(y_true, self.predict(x))


def __main__():
    col_names = ["age", "workclass", "financial weight", "education", "education-num", "marital status", "occupation",
                 "relationship", "race", "sex", "capital gain", "capital loss", "hour per week", "native country", "label"]
    x_train, x_test, y_train, y_test = load_data("Data.csv", col_names)
    x_train, x_test = preprocess_data(
        x_train, x_test, y_train, scale=True)
    print("Naive Bayes Classifier:")
    model = NaiveBayesClassifier()
    model.fit(x_train, y_train)
    y_pred, _ = model.predict(x_test)
    print("Accuracy Score =", model.score(x_test, y_test))
    print("Error =", model.error(x_test, y_test))

    print("Logistic Regression:")
    model2 = LogisticRegression()
    model2.fit(x_train, y_train, 0.15, 100000)
    print("Accuracy Score =", model2.score(x_test, y_test))
    print("Error =", model2.error(x_test, y_test))
    # plt.plot(model2.history)
    # plt.show()


__main__()
