"""
Programmer: Cooper Braun & Sam Vanturennout
Class: CPSC 322-01, Fall 2025
Final Project
12/1/25
Description: Random forest classifier that wraps the custom decision tree
builder with bootstrap aggregation and random attribute sub-sampling.
"""

import random
from typing import List, Tuple

from mysklearn import evaluation, utils


class MyRandomForestClassifier:
    """
    Random forest implementation using custom decision trees.
    """

    def __init__(
        self,
        n_trees: int = 10,
        max_features: int | None = None,
        forest_size: int | None = None,
        random_state: int | None = None
    ):
        self.n_trees = max(1, n_trees)
        self.max_features = max_features
        self.forest_size = forest_size if forest_size is not None else self.n_trees
        self.random_state = random_state

        self._rng = random.Random(random_state)
        self.forest: List[dict] = []
        self.classes_: List = []

        self.forest_size = max(1, min(self.forest_size, self.n_trees))

    def fit(self, X_train: List[List], y_train: List):
        """
        Fit the random forest using bootstrap aggregation.
        
        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        if not X_train or not y_train:
            self.forest = []
            self.classes_ = []
            return

        self.classes_ = sorted(set(y_train))
        models: List[Tuple[float, dict]] = []

        for _ in range(self.n_trees):
            seed = None
            if self.random_state is not None:
                seed = self._rng.randint(0, 10**6)

            X_sample, X_val, y_sample, y_val = evaluation.bootstrap_sample(
                X_train, y_train, random_state=seed
            )

            if not X_sample or not y_sample:
                continue

            tree, default_label = utils.build_random_tree(
                X_sample,
                y_sample,
                rng=self._rng,
                max_features=self.max_features
            )

            if y_val:
                y_pred_val = [
                    utils.predict_instance(row, tree, default_label) for row in X_val
                ]
                accuracy = evaluation.accuracy_score(y_val, y_pred_val)
            else:
                accuracy = 0.0

            models.append(
                (
                    accuracy,
                    {"tree": tree, "default": default_label}
                )
            )

        models.sort(key=lambda item: item[0], reverse=True)
        selected = models[: self.forest_size]
        self.forest = [entry for _, entry in selected]

    def predict(self, X_test: List[List]):
        """
        Predict test labels via majority vote across the fitted forest.
        
        Args:
            X_test(list of list of obj): The list of test instances (samples).
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            list of obj: The predicted labels for the test instances
        """
        if not self.forest or not X_test:
            return []

        predictions = []
        default_vote = self.classes_[0] if self.classes_ else None
        for instance in X_test:
            votes = [
                utils.predict_instance(instance, tree_info["tree"], tree_info["default"])
                for tree_info in self.forest
            ]
            label, _ = utils.majority_label(votes, default_vote)
            predictions.append(label)
        return predictions