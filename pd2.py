import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, make_scorer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric

import xgboost as xgb
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
# from hpelm import ELM

from operator import itemgetter


def get_tn(y1, y2):
    return confusion_matrix(y1, y2)[0][0]


def get_fp(y1, y2):
    return confusion_matrix(y1, y2)[0][1]


def get_fn(y1, y2):
    return confusion_matrix(y1, y2)[1][0]


def get_tp(y1, y2):
    return confusion_matrix(y1, y2)[1][1]


def format_print_confusion_matrix(tn, fp, fn, tp):
    for i in range(5):
        print(f"\tFOLD#{i}:")
        print("\tTN: {}\tFP: {}".format(tn[i], fp[i]))
        print("\tFN: {}\tTP: {}".format(fn[i], tp[i]))
    print("Average:")
    print("\tTN: {}\tFP: {}".format(np.sum(tn) / 5, np.sum(fp) / 5))
    print("\tFN: {}\tTP: {}".format(np.sum(fn) / 5, np.sum(tp) / 5))


def format_print_cv_score(train, test):
    print("Train:")
    for i, x in enumerate(train):
        print(f"\tFold#{i}: {x}")
    print("Average = {}".format(np.sum(train) / 5))
    print()
    print("Test:")
    for i, x in enumerate(test):
        print(f"\tFold#{i}: {x}")
    print("Average = {}".format(np.sum(test) / 5))
    print("-" * 20)


def load_data(file_name):
    df = pd.read_csv(file_name)

    y = pd.Series(df.iloc[1:, -1].values, name=df.iloc[0, -1]).astype(np.int8)
    patient_id = pd.Series(df.iloc[1:, 0].values,
                           name=df.iloc[0, 0]).astype(np.int16)
    gender = pd.Series(df.iloc[1:, 1].values,
                       name=df.iloc[0, 1]).astype(np.int8)

    baseline_feats = pd.DataFrame(
        df.iloc[1:, 2:23].values, columns=df.iloc[0, 2:23]).astype(np.float64)
    intensity_feats = pd.DataFrame(
        df.iloc[1:, 23:26].values, columns=df.iloc[0, 23:26]).astype(np.float64)
    format_feats = pd.DataFrame(
        df.iloc[1:, 26:30].values, columns=df.iloc[0, 26:30]).astype(np.float64)
    bandwidth_feats = pd.DataFrame(
        df.iloc[1:, 30:34].values, columns=df.iloc[0, 30:34]).astype(np.float64)
    vocal_feats = pd.DataFrame(
        df.iloc[1:, 34:56].values, columns=df.iloc[0, 34:56]).astype(np.float64)
    mfcc_feats = pd.DataFrame(
        df.iloc[1:, 56:140].values, columns=df.iloc[0, 56:140]).astype(np.float64)
    wavelet_feats = pd.DataFrame(
        df.iloc[1:, 140:322].values, columns=df.iloc[0, 140:322]).astype(np.float64)
    tqwt_feats = pd.DataFrame(
        df.iloc[1:, 322:-1].values, columns=df.iloc[0, 322:-1]).astype(np.float64)

    return {"patientId": patient_id,
            "gender": gender,
            "baselineFeats": baseline_feats,
            "intensityFeats": intensity_feats,
            "formantFeats": format_feats,
            "bandwidthFeats": bandwidth_feats,
            "vocalFeats": vocal_feats,
            "mfccFeats": mfcc_feats,
            "waveletFeats": wavelet_feats,
            "tqwtFeats": tqwt_feats,
            "label": y}


def convert_data(data, features):
    if len(features) == 1:
        return data[features[0]]
    return pd.concat(itemgetter(*features)(data), axis=1)


def initialize_scores():
    return {"train_accuracy": [], "test_accuracy": [],
            "train_TN": [], "test_TN": [], "train_FN": [], "test_FN": [],
            "train_FP": [], "test_FP": [], "train_TP": [], "test_TP": [],
            "train_precision": [], "test_precision": [],
            "train_recall": [], "test_recall": [],
            "train_f1": [], "test_f1": []}


def get_score(scores, y_train, y_test, y_pred_train, y_pred_test):
    scores["train_accuracy"].append(accuracy_score(y_train, y_pred_train))
    scores["test_accuracy"].append(accuracy_score(y_test, y_pred_test))
    scores["train_TN"].append(get_tn(y_train, y_pred_train))
    scores["test_TN"].append(get_tn(y_test, y_pred_test))
    scores["train_FN"].append(get_fn(y_train, y_pred_train))
    scores["test_FN"].append(get_fn(y_test, y_pred_test))
    scores["train_TP"].append(get_tp(y_train, y_pred_train))
    scores["test_TP"].append(get_tp(y_test, y_pred_test))
    scores["train_FP"].append(get_fp(y_train, y_pred_train))
    scores["test_FP"].append(get_fp(y_test, y_pred_test))
    scores["train_precision"].append(precision_score(y_train, y_pred_train))
    scores["test_precision"].append(precision_score(y_test, y_pred_test))
    scores["train_recall"].append(recall_score(y_train, y_pred_train))
    scores["test_recall"].append(recall_score(y_test, y_pred_test))
    scores["train_f1"].append(f1_score(y_train, y_pred_train))
    scores["test_f1"].append(f1_score(y_test, y_pred_test))

    return scores


def run_mlp(X, y, **params):
    hid_layers = params['hid_layers']
    n_hdim = len(hid_layers)

    kfold = KFold(n_splits=5)
    scores = initialize_scores()
    # scores = []
    for train_idx, test_idx in kfold.split(X):
        model = tf.keras.Sequential()
        model.add(
            layers.Dense(hid_layers[0][0], input_dim=X.shape[1], activation=hid_layers[0][1]))
        for i in range(1, n_hdim):
            model.add(layers.Dense(
                hid_layers[i][0], activation=hid_layers[i][1]))
        model.compile(**params['compile'])
        X_train, y_train = X[train_idx, :], y.iloc[train_idx]
        X_test, y_test = X[test_idx, :], y.iloc[test_idx]
        sclr = MinMaxScaler()
        X_train = sclr.fit_transform(X_train)
        X_test = sclr.transform(X_test)
        model.fit(X_train, y_train.values,
                  epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
        y_pred_train = model.predict(X_train).round()
        y_pred_test = model.predict(X_test).round()
        # print(y_pred_test)
        scores = get_score(scores, y_train, y_test, y_pred_train, y_pred_test)
        # scores.append(model.evaluate(X_test.values, y_test.values, batch_size=params['batch_size']))

    return scores


def run_elm(X, y, **params):
    kfolds = KFold(n_splits=5, shuffle=True)
    y = pd.concat([y, 1 - y], axis=1)
    scores = initialize_scores()
    for train_idx, test_idx in kfolds.split(X):
        X_train, y_train = X.iloc[train_idx, :], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx, :], y.iloc[test_idx]
        sclr = MinMaxScaler()
        X_train = sclr.fit_transform(X_train.values)
        X_test = sclr.transform(X_test.values)
        clf = ELM(X.shape[1], 2, 'c', **params['init'])
        clf.add_neurons(params['hid_dim'], params['func'])
        clf.train(X_train, y_train.values, *
                  params['train_args'], **params["train_kwargs"])
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        y_pred_train = y_pred_train[:, 0] > y_pred_train[:, 1]
        y_pred_test = y_pred_test[:, 0] > y_pred_test[:, 1]
        y_train = y_train.iloc[:, 0] > y_train.iloc[:, 1]
        y_test = y_test.iloc[:, 0] > y_test.iloc[:, 1]
        scores = get_score(scores, y_train, y_test, y_pred_train, y_pred_test)

    return scores


def run_model(classifier_type, X, y, **params):
    if classifier_type == "mlp":
        return run_mlp(X, y, **params)
    elif classifier_type == "elm":
        return run_elm(X, y, **params)
    elif classifier_type == "xgb":
        clf = xgb.XGBClassifier(**params)
    elif classifier_type == "svm":
        clf = Pipeline(
            [('normalizer', MinMaxScaler()), ('SVC', SVC(**params))])
    elif classifier_type == "knn":
        clf = Pipeline([('normalizer', MinMaxScaler()),
                        ('KNN', KNeighborsClassifier(**params))])
    else:
        raise ValueError(
            "Invalid classifier type. Valid types: mlp, svm, elm, xgb")
    scoring = {"accuracy": make_scorer(accuracy_score),
               "precision": make_scorer(precision_score),
               "recall": make_scorer(recall_score),
               "f1": make_scorer(f1_score),
               "TN": make_scorer(get_tn),
               "FP": make_scorer(get_fp),
               "FN": make_scorer(get_fn),
               "TP": make_scorer(get_tp)}
    return cross_validate(clf, X, y, scoring=scoring, cv=5, return_train_score=True)


def print_results(scores, features, params):
    print("#" * 30)
    print("#" * 30)
    print(f"{features} are used as features to train the model.")
    print(f"The parameters used for the model is", params)
    print("-" * 25)
    print(">>>>> CONFUSION MATRIX <<<<<")
    print("Train:")
    format_print_confusion_matrix(
        scores["train_TN"], scores["train_FP"], scores["train_FN"], scores["train_TP"])
    print("Test:")
    format_print_confusion_matrix(
        scores["test_TN"], scores["test_FP"], scores["test_FN"], scores["test_TP"])
    print(">>>>> ACCURACY <<<<<")
    format_print_cv_score(scores["train_accuracy"], scores["test_accuracy"])
    print(">>>>> PRECISION <<<<<")
    format_print_cv_score(scores["train_precision"], scores["test_precision"])
    print(">>>>> RECALL <<<<<")
    format_print_cv_score(scores["train_recall"], scores["test_recall"])
    print(">>>>> F1 SCORE <<<<<")
    format_print_cv_score(scores["train_f1"], scores["test_f1"])


def __main__():
    data = load_data('pd_speech_features.csv')

    # params = {'max_depth': 2, 'eta': 0.1, 'n_estimators': 50, 'booster': 'dart',
    #           'gamma': 0, 'reg_lambda': 0.05, 'objective': 'binary:logistic'}

    features = ["gender", "baselineFeats", "intensityFeats", "formantFeats",
                "bandwidthFeats", "vocalFeats", "mfccFeats", "waveletFeats", "tqwtFeats"]

    X = convert_data(data, features)
    params = {"hid_layers": [(280, "relu"), (1, "sigmoid")], "compile": {
        "loss": 'binary_crossentropy', "optimizer": tf.keras.optimizers.Adam(lr=0.001), "metrics": ["accuracy"]}, "epochs": 10, "batch_size": None}
    # scores = run_model("mlp", X.values, data['label'], **params)

    # params = {'hid_dim': 150, 'func': 'sigm', 'init': {'norm': 0.05},
    #           'train_args': ['OP', 'c'], 'train_kwargs': {'kmax': 100}}
    #
    # scores = run_model("elm", X, data['label'], **params)
    params = {"kernel": "poly", "degree": 20}
    # scores = run_model("svm", X, data['label'], **params)
    params = {"n_neighbors": 1,
              "metric": DistanceMetric.get_metric("manhattan")}
    scores = run_model("knn", X, data['label'], **params)
    print_results(scores, features, params)


__main__()
