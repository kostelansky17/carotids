import os

import numpy as np
from sklearn.linear_model import LogisticRegression, TRANSFORMATIONS


def categorize_data(data_path, crop=False):
    clf = train_classifier()
    file_names = np.asarray(os.listdir(data_path))
    X = load_dir(data_path, crop)
    y = clf.predict(X)
    print(y)
    return file_names[y == 1], file_names[y == 0]


def train_classifier():
    X, y = create_categorization_features()
    clf = LogisticRegression()
    clf.fit(X, y)
    print(np.sum(clf.predict(X) == y)/len(y))
    return  clf
