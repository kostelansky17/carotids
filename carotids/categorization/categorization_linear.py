import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from carotids.preprocessing import load_dir


def create_categorization_features(img_dirs):
    data, labels = [], []

    for key in img_dirs:
        data.append(load_dir(img_dirs[key], False))
        labels.append(np.full(len(os.listdir(img_dirs[key])), key))

    return np.vstack(data), np.concatenate(labels)


def categorize_data(data_path, crop=False):
    clf = train_classifier()
    file_names = np.asarray(os.listdir(data_path))
    X = load_dir(data_path, crop)
    y = clf.predict(X)
    return file_names[y == 1], file_names[y == 0]


def train_classifier(X, y, hps):
    clf = LogisticRegression(**hps)
    clf.fit(X, y)
    return clf


def try_hyperparameters(X, y, hps, clf=LogisticRegression):
    accuracies = []
    best_acc = 0.0
    best_hp = None
    best_clf = None
    for hp in hps:
        score = cross_val_score(clf(**hp), X, y, scoring="accuracy", cv=10)
        accuracies.append(score)
        if score.mean() > best_acc:
            best_acc, best_hp = score.mean(), hp

    print(f"Best ACCURACY: {best_acc}, hp: {best_hp}")
    return np.asarray(accuracies), best_hp
