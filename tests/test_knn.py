"""
Programmer: Cooper Braun & Sam Vanturennout
Class: CPSC 322-01, Fall 2025
Final Project
12/1/25
Description: Tests for the KNeighborsClassifier class.
"""

import pytest
from mysklearn.knn import KNeighborsClassifier

# example #1  (4 instances)
X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
y_train_class_example1 = ["bad", "bad", "good", "good"]

# example #2 (8 instances)
X_train_class_example2 = [
    [3, 2],
    [6, 6],
    [4, 1],
    [4, 4],
    [1, 2],
    [2, 0],
    [0, 3],
    [1, 6]
]
y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]

# Bramer example (figure 3.5)
header_bramer_example = ["Attribute 1", "Attribute 2"]
X_train_bramer_example = [
    [0.8, 6.3],
    [1.4, 8.1],
    [2.1, 7.4],
    [2.6, 14.3],
    [6.8, 12.6],
    [8.8, 9.8],
    [9.2, 11.6],
    [10.8, 9.6],
    [11.8, 9.9],
    [12.4, 6.5],
    [12.8, 1.1],
    [14.0, 19.9],
    [14.2, 18.5],
    [15.6, 17.4],
    [15.8, 12.2],
    [16.6, 6.7],
    [17.4, 4.5],
    [18.2, 6.9],
    [19.0, 3.4],
    [19.6, 11.1]
]
y_train_bramer_example = [
    "-", "-", "-", "+", "-", "+", "-", "+", "+", "+",
    "-", "-", "-", "-", "-", "+", "+", "+", "-", "+"
]

def test_kneighbors_classifier_kneighbors():
    """
    kneighbors(): distances and indices for multiple datasets.
    """
    # example #1: X_test = [[0.33, 0]], k=3
    knn1 = KNeighborsClassifier(n_neighbors=3)
    knn1.fit(X_train_class_example1, y_train_class_example1)
    dists1, idxs1 = knn1.kneighbors([[0.33, 0]])

    # expect nearest three: itself [0.33,0] (idx=2, dist=0.0),
    # then [0,0] (idx=3, dist=0.33), then [1,0] (idx=1, dist=0.67)
    assert len(dists1) == 1 and len(idxs1) == 1
    assert dists1[0] == pytest.approx([0.0, 0.33, 0.67], abs=1e-3)
    assert idxs1[0] == [2, 3, 1]

    # example #2: X_test = [[6, 6]], k=3
    knn2 = KNeighborsClassifier(n_neighbors=3)
    knn2.fit(X_train_class_example2, y_train_class_example2)
    dists2, idxs2 = knn2.kneighbors([[6, 6]])

    # distances to the 3 closest: [6,6] (idx=1, 0.0),
    # [4,4] (idx=3, sqrt(8) ~ 2.828),
    # [3,2] (idx=0, 5.0)
    assert len(dists2) == 1 and len(idxs2) == 1
    assert pytest.approx(dists2[0][0], rel=1e-9, abs=1e-9) == 0.0
    assert pytest.approx(dists2[0][1], rel=1e-6) == 2.8284271247461903
    assert pytest.approx(dists2[0][2], rel=1e-9, abs=1e-9) == 5.0
    assert idxs2[0] == [1, 3, 0]

    # Bramer example 2: X_test = [[9.1, 11.0]], k=5
    knn3 = KNeighborsClassifier(n_neighbors=5)
    knn3.fit(X_train_bramer_example, y_train_bramer_example)
    dists3, idxs3 = knn3.kneighbors([[9.1, 11.0]])

    # from the book's table, the 5 nearest (in ascending distance) are:
    # idx 6 (0.608), idx 5 (1.237), idx 7 (2.202), idx 4 (2.802), idx 8 (2.915)
    expected_idxs = [6, 5, 7, 4, 8]
    expected_dists = [0.608, 1.237, 2.202, 2.802, 2.915]

    assert len(dists3) == 1 and len(idxs3) == 1
    # check each with small tolerance
    for got, exp in zip(dists3[0], expected_dists):
        assert pytest.approx(got, rel=1e-3, abs=1e-3) == exp
    assert idxs3[0] == expected_idxs


def test_kneighbors_classifier_predict():
    """
    predict(): majority vote among k neighbors.
    """
    # example #1: X_test = [[0.33, 0]], k=3 => labels: ["good","good","bad"] -> "good"
    knn1 = KNeighborsClassifier(n_neighbors=3)
    knn1.fit(X_train_class_example1, y_train_class_example1)
    assert knn1.predict([[0.33, 0]]) == ["good"]

    # example #2: X_test = [[6, 6]], k=3
    # neighbors labels expected: [ "yes", "no", "no"] -> "no"
    knn2 = KNeighborsClassifier(n_neighbors=3)
    knn2.fit(X_train_class_example2, y_train_class_example2)
    assert knn2.predict([[6, 6]]) == ["no"]

    # Bramer example 2: X_test = [[9.1, 11.0]], k=5
    # neighbor labels at indices [6,5,7,4,8] are: ["-", "+", "+", "-", "+"]
    # majority "+" -> predict ["+"]
    knn3 = KNeighborsClassifier(n_neighbors=5)
    knn3.fit(X_train_bramer_example, y_train_bramer_example)
    assert knn3.predict([[9.1, 11.0]]) == ["+"]