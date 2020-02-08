import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


class Data:
    def __init__(self, file_name):
        data = np.load(file_name)

        x1_train = data['x1']
        x2_train = data['x2']
        x1_test = data['x1_test']
        x2_test = data['x2_test']

        X_train = np.vstack((x1_train, x2_train))
        X_test = np.vstack((x1_test, x2_test))
        y_train = data['y']
        y_test = data['y_test']

        self.data = {
            "X_train": X_train.T,
            "X_test": X_test.T,
            "y_train": y_train.T,
            "y_test": y_test.T
        }

    def get_x_train(self):
        return self.data["X_train"]

    def get_x_test(self):
        return self.data["X_test"]

    def get_y_train(self):
        return self.data["y_train"]

    def get_y_test(self):
        return self.data["y_test"]


class PolynomialRegression:
    def __init__(self, degree=1, enable_gradient_descent=False):
        self.degree = degree
        self.enable_gradient_descent = enable_gradient_descent
        self.coef_ = None

    def transform(self, X):
        poly = PolynomialFeatures(degree=self.degree)
        return poly.fit_transform(X)

    def fit(self, X, y, reg_lambda=0, eta=0, tol=0, num_iters=0):
        self.coef_ = None
        if self.enable_gradient_descent:
            self.coef_ = np.random.rand(X.shape[1])
            self.coef_.reshape((self.coef_.shape[0], 1))
            costs = []
            for iters in range(num_iters):
                self.coef_ -= eta * np.transpose(X)@(X@self.coef_ - y)
                s = self.compute_error(y, self.predict(X))
                if s < tol:
                    break
                if len(costs) > 0 and costs[-1] - s <= 0.001:
                    break
                costs.append(s)
        else:
            self.coef_ = np.matmul(np.matmul(np.linalg.inv(
                np.matmul(X.T, X) + reg_lambda * np.eye(X.shape[1])), X.T), y)
            self.coef_.reshape((self.coef_.shape[0], 1))
        return self

    def predict(self, X):
        return np.matmul(X, self.coef_)

    def compute_error(self, y_true, y_pred):
        return np.sum(np.square(y_true - y_pred))


def initialize_model(data, degree, enable_gradient_descent=False):
    model = PolynomialRegression(
        degree=degree, enable_gradient_descent=enable_gradient_descent)
    X_train = model.transform(data.get_x_train())
    X_test = model.transform(data.get_x_test())
    y_train, y_test = data.get_y_train(), data.get_y_test()
    return model, X_train, X_test, y_train, y_test


def part_a(data):
    print(">>>>> Part A <<<<<")
    for deg in [1, 3, 5]:
        print(f"degree = {deg}")
        model, X_train, X_test, y_train, y_test = initialize_model(data, deg)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_sse = model.compute_error(y_test, y_pred)
        train_sse = model.compute_error(y_train, model.predict(X_train))
        print("Coefficients:", model.coef_)
        print(f"\tTRAIN SSE = {train_sse},\tTEST SSE = {test_sse}")


def part_b(data, eta, num_iters, tol=0.001):
    print(">>>>> Part B <<<<<")
    for i, deg in enumerate([1, 3, 5]):
        print(f"degree = {deg}")
        model, X_train, X_test, y_train, y_test = initialize_model(
            data, deg, True)
        model.fit(X_train, y_train, eta=eta[i],
                  tol=tol, num_iters=num_iters[i])
        y_pred = model.predict(X_test)
        test_sse = model.compute_error(y_test, y_pred)
        train_sse = model.compute_error(y_train, model.predict(X_train))
        print("Coefficients:", model.coef_)
        print(f"\tTRAIN SSE = {train_sse},\tTEST SSE = {test_sse}")


def part_c(data):
    print(">>>>> Part C <<<<<")
    for deg in [1, 3, 5]:
        print(f"degree = {deg}")
        model, X_train, X_test, y_train, y_test = initialize_model(data, deg)

        best_reg_lambda = 1
        best_err = np.inf
        n_folds = 5
        folds = KFold(n_splits=n_folds)
        for alpha in [10**i for i in range(-4, 5)]:
            sse = 0
            for train_idx, val_idx in folds.split(X_train):
                train_x, val_x = X_train[train_idx], X_train[val_idx]
                train_y, val_y = y_train[train_idx], y_train[val_idx]
                model.fit(train_x, train_y, reg_lambda=alpha)
                sse += model.compute_error(val_y, model.predict(val_x))
            sse = sse / n_folds
            if sse < best_err:
                best_err = sse
                best_reg_lambda = alpha

        model.fit(X_train, y_train, reg_lambda=best_reg_lambda)
        y_pred = model.predict(X_test)
        test_sse = model.compute_error(y_test, y_pred)
        train_sse = model.compute_error(y_train, model.predict(X_train))
        print(f"Best regularization parameter = {best_reg_lambda}")
        print("Coefficients:", model.coef_)
        print(f"\tTRAIN SSE = {train_sse},\tTEST SSE = {test_sse}")


def __main__():
    data = Data(file_name='data.npz')
    part_a(data)
    part_b(data, eta=[9 * 10**(-7), 2.467 * 10**(-11), 2.4459 * 10**(-16)],
           num_iters=[100000, 500000, 500000])
    part_c(data)


__main__()
