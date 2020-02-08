import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, make_scorer

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


def run_model(X, y, params):
    clf = xgb.XGBClassifier(**params)
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

    params = {'max_depth': 2, 'eta': 0.1, 'n_estimators': 50, 'booster': 'dart',
              'gamma': 0, 'reg_lambda': 0.05, 'objective': 'binary:logistic'}

    features = ["gender", "baselineFeats", "intensityFeats", "formantFeats",
                "bandwidthFeats", "vocalFeats", "mfccFeats", "waveletFeats", "tqwtFeats"]

    X = convert_data(data, features)
    scores = run_model(X, data['label'], params)
    print_results(scores, features, params)


__main__()
