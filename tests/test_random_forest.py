"""
Programmer: Cooper Braun & Sam Vanturennout
Class: CPSC 322-01, Fall 2025
Final Project
12/1/25
Description: Tests for the MyRandomForestClassifier class.
"""

from collections import Counter

from mysklearn import evaluation
from mysklearn.random_forest import MyRandomForestClassifier


def _toy_dataset():
    X = []
    y = []
    colors = ["red", "blue"]
    sizes = ["small", "medium", "large"]
    textures = ["rough", "smooth"]

    for color in colors:
        for size in sizes:
            for texture in textures:
                X.append([color, size, texture])
                label = "hot" if color == "red" else "cold"
                y.append(label)

    X = X * 2
    y = y * 2
    return X, y


def _stratified_thirds(X, y, seed=0):
    folds = evaluation.stratified_kfold_split(
        X, y, n_splits=3, random_state=seed, shuffle=True
    )
    train_indices, test_indices = folds[0]
    X_train = [X[i] for i in train_indices]
    y_train = [y[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_test = [y[i] for i in test_indices]
    return X_train, X_test, y_train, y_test


def test_random_forest_stratified_split_preserves_ratio():
    X, y = _toy_dataset()
    X_train, X_test, y_train, y_test = _stratified_thirds(X, y, seed=3)

    assert len(X_test) == len(X) // 3
    train_dist = Counter(y_train)
    test_dist = Counter(y_test)

    for label in train_dist:
        train_ratio = train_dist[label] / len(y_train)
        test_ratio = test_dist[label] / len(y_test)
        assert abs(train_ratio - test_ratio) <= 0.1


def test_random_forest_fit_selects_top_m():
    X, y = _toy_dataset()
    X_train, _, y_train, _ = _stratified_thirds(X, y, seed=5)

    clf = MyRandomForestClassifier(n_trees=7, forest_size=3, random_state=11)
    clf.fit(X_train, y_train)

    assert len(clf.forest) == 3
    for tree_info in clf.forest:
        assert "tree" in tree_info and "default" in tree_info
        assert tree_info["tree"][0] in {"Leaf", "Attribute"}


def test_random_forest_predict_majority_vote():
    X, y = _toy_dataset()
    X_train, X_test, y_train, y_test = _stratified_thirds(X, y, seed=7)

    clf = MyRandomForestClassifier(n_trees=15, forest_size=5, random_state=21)
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)

    assert len(predictions) == len(y_test)
    accuracy = evaluation.accuracy_score(y_test, predictions)
    assert accuracy >= 0.8