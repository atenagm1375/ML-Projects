import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from operator import itemgetter


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
    return pd.concat(itemgetter(*features)(data), axis=1).values


def build_block(input, size, activation, batch_norm, dropout):
    layer = keras.layers.Dense(
        size, activation=activation, input_dim=input.shape[1])(input)
    if batch_norm:
        layer = keras.layers.BatchNormalization()(layer)
    if dropout:
        layer = keras.layers.Dropout(0.1)(layer)
    return layer


def build_autoencoder(input_shape, structure, activations, batch_norm=False, dropout=False, optimizer="sgd"):
    input = keras.layers.Input(shape=input_shape)
    layer = build_block(
        input, structure[0], activations[0], batch_norm, dropout)
    for i in range(1, len(structure)):
        layer = build_block(
            layer, structure[i], activations[i], batch_norm, dropout)
    model = keras.models.Model(inputs=input, outputs=layer)
    model.compile(loss="mean_squared_error", optimizer=optimizer)
    model.summary()
    return model


def run_model(X, y, autoencoder_params, encoder_params, classifier=None, classifier_params=None):
    encoder_size = len(autoencoder_params['structure']) // 2
    folds = KFold(n_splits=5, shuffle=True)
    scores = {'accuracy': [], 'fscore': [], 'precision': [], 'recall': []}
    conf_mats = []
    for train_idx, test_idx in folds.split(X):
        x_train, x_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        autoencoder = build_autoencoder(**autoencoder_params)
        history = autoencoder.fit(x_train, x_train, epochs=encoder_params.get(
            'epochs'), batch_size=encoder_params.get('batch_size'), validation_data=(x_test, x_test))
        # plot_loss(history, encoder_params.get('epochs'), "autoencoder")
        encoder = autoencoder.layers[:encoder_size]

        for layer in encoder:
            layer.trainable = False

        if classifier is None:
            model = keras.models.Sequential(
                layers=[layer for layer in encoder])
            layer = keras.layers.Dense(1, activation='sigmoid')
            model.add(layer)
            model.compile(loss=keras.losses.binary_crossentropy,
                          optimizer=classifier_params['optimizer'], metrics=["accuracy"])
            history = model.fit(x_train, y_train, batch_size=classifier_params.get(
                'batch_size'), epochs=classifier_params.get("epochs"), validation_data=(x_test, y_test))
            # plot_loss(history, classifier_params.get('epochs'), "classifier1")

            if classifier_params.get("rerun") == True:
                for layer in model.layers[:encoder_size]:
                    layer.trainable = True
                model.compile(loss=keras.losses.binary_crossentropy,
                              optimizer=classifier_params.get("optimizer"), metrics=["accuracy"])
                history = model.fit(x_train, y_train, batch_size=classifier_params.get(
                    'batch_size'), epochs=classifier_params.get("epochs"), validation_data=(x_test, y_test))
                # plot_loss(history, classifier_params.get(
                #     'epochs'), "classifier2")
            y_pred = np.round(model.predict(x_test))
        else:
            model = keras.models.Sequential(
                layers=[layer for layer in encoder])
            new_train = model.predict(x_train)
            new_test = model.predict(x_test)
            model = classifier(**classifier_params)
            model.fit(new_train, y_train)
            y_pred = np.round(model.predict(new_test))
        scores['accuracy'].append(accuracy_score(y_test, y_pred))
        scores['fscore'].append(f1_score(y_test, y_pred))
        scores['precision'].append(precision_score(y_test, y_pred))
        scores['recall'].append(recall_score(y_test, y_pred))
        conf_mats.append(confusion_matrix(y_test, y_pred))

    return scores, conf_mats


def plot_loss(model, epochs, context):
    loss = model.history['loss']
    val_loss = model.history['val_loss']
    acc = model.history.get('accuracy')
    val_acc = model.history.get('val_accuracy')
    epochs = range(epochs)
    plt.figure()
    plt.plot(epochs, loss, 'b', label=f'Training loss-{context}')
    plt.title('Training loss')
    plt.plot(epochs, val_loss, 'r', label=f'Validation loss-{context}')
    plt.legend()
    plt.show()
    if acc is not None:
        plt.figure()
        plt.plot(epochs, acc, 'b', label=f'Training accuracy-{context}')
        plt.plot(epochs, val_acc, 'r', label=f'Validation accuracy-{context}')
        plt.legend()
        plt.show()


def __main__():
    data = load_data('pd_speech_features.csv')
    y = data['label']

    features = ["gender", "baselineFeats", "intensityFeats", "formantFeats",
                "bandwidthFeats", "vocalFeats", "mfccFeats", "waveletFeats", "tqwtFeats"]

    X = convert_data(data, features)
    params = {'input_shape': (X.shape[1],),
              'structure': [600, 300, 100, 30, 100, 300, 600, X.shape[1]],
              'activations': ['relu'] * 8,
              'optimizer': keras.optimizers.Adam(lr=0.001),
              'batch_norm': True}
    print("################################")
    encoder_params = {"epochs": 250, "batch_size": 256}
    classifier_params = {'optimizer': keras.optimizers.Adam(
        lr=0.0001), 'epochs': 200, 'rerun': False}  # fine tunes if rerun=True
    scores, confs = run_model(X, y.values, params, encoder_params=encoder_params,
                              classifier=None, classifier_params=classifier_params)
    print("ACCURACY:", np.mean(scores['accuracy']))
    print("F SCORE:", np.mean(scores['fscore']))
    print("PRECISION:", np.mean(scores['precision']))
    print("RECALL:", np.mean(scores['recall']))
    print("CONFUSION MATRICES:")
    for i in range(len(confs)):
        print(confs[i])


__main__()
