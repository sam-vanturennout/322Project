"""
Programmer: Cooper Braun
Class: CPSC 322-01, Fall 2025
Final Project
12/1/25
Description: Decision tree classifier for our final project. Currently includes
an entropy-based decision tree with rule printing.
"""

from mysklearn import utils

class DecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None
        self._attribute_domains = None
        self._default_label = None

    def fit(self, X_train, y_train):
        """
        Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        if not X_train or not y_train:
            self.X_train = []
            self.y_train = []
            self.tree = None
            self._attribute_domains = None
            self._default_label = None
            return

        self.X_train = [list(row) for row in X_train]
        self.y_train = list(y_train)
        num_attributes = len(self.X_train[0])
        self._attribute_domains = utils.build_attribute_domains(self.X_train)
        self._default_label, _ = utils.majority_label(self.y_train)

        available_attributes = list(range(num_attributes))
        self.tree = utils.tdidt(
            self.X_train,
            self.y_train,
            available_attributes,
            self._attribute_domains,
            self._default_label,
            len(self.y_train)
        )

    def predict(self, X_test):
        """
        Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        if self.tree is None:
            return []

        predictions = []
        for instance in X_test:
            predictions.append(
                utils.predict_instance(instance, self.tree, self._default_label)
            )
        return predictions

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """
        Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        if self.tree is None:
            return

        def traverse(node, conditions):
            if not node:
                return
            node_type = node[0]
            if node_type == "Leaf":
                condition_str = " AND ".join(conditions) if conditions else "TRUE"
                print(f"IF {condition_str} THEN {class_name} = {node[1]}")
                return

            attribute_label = node[1]
            attribute_index = int(attribute_label.replace("att", ""))
            display_name = (
                attribute_names[attribute_index]
                if attribute_names and attribute_index < len(attribute_names)
                else f"att{attribute_index}"
            )
            for child in node[2:]:
                value = child[1]
                new_condition = f"{display_name} == {value}"
                traverse(child[2], conditions + [new_condition])

        traverse(self.tree, [])