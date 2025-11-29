"""
Programmer: Cooper Braun
Class: CPSC 322-01, Fall 2025
Final Project
12/1/25
Description: Shared utilities for custom classifiers and data tools. Includes
decision-tree helpers (entropy, TDIDT support, column projection), and 
evaluation shortcuts like reusable cross-validation logic for feature subsets.
"""

import math
from collections import Counter
from mysklearn import evaluation


def project_columns(rows, column_indexes):
    """
    Project a dataset onto the specified column indexes.
    """
    return [[row[idx] for idx in column_indexes] for row in rows]


def build_attribute_domains(X):
    """
    Build sorted domains for each attribute in a dataset.
    """
    domains = {}
    if not X:
        return domains
    num_attributes = len(X[0])
    for index in range(num_attributes):
        values = sorted({row[index] for row in X})
        domains[index] = values
    return domains


def majority_label(labels, default_label=None):
    """
    Return the majority label and its count (alphabetical tie-break).
    """
    if not labels:
        return default_label, 0
    counts = Counter(labels)
    max_count = max(counts.values())
    candidates = sorted([label for label, cnt in counts.items() if cnt == max_count])
    chosen_label = candidates[0]
    return chosen_label, counts[chosen_label]


def partition_dataset(X, y, attribute_index, attribute_value):
    """
    Partition X and y by a specific attribute value.
    """
    subset_X = []
    subset_y = []
    for row, label in zip(X, y):
        if row[attribute_index] == attribute_value:
            subset_X.append(row)
            subset_y.append(label)
    return subset_X, subset_y


def entropy(labels):
    """
    Compute entropy of a list of labels.
    """
    total = len(labels)
    if total == 0:
        return 0
    entropy_value = 0
    counts = Counter(labels)
    for count in counts.values():
        probability = count / total
        entropy_value -= probability * math.log2(probability)
    return entropy_value


def attribute_entropy(X, y, attribute_index, attribute_domains):
    """
    Calculate the entropy for a specific attribute split.
    """
    total = len(y)
    entropy_sum = 0
    for value in attribute_domains.get(attribute_index, []):
        subset_labels = [
            label
            for row, label in zip(X, y)
            if row[attribute_index] == value
        ]
        if not subset_labels:
            continue
        weight = len(subset_labels) / total
        entropy_sum += weight * entropy(subset_labels)
    return entropy_sum


def select_attribute(X, y, available_attributes, attribute_domains):
    """
    Select the attribute with the highest information gain.
    """
    base_entropy = entropy(y)
    best_attribute = available_attributes[0]
    best_gain = -math.inf
    for attribute in available_attributes:
        attr_entropy = attribute_entropy(X, y, attribute, attribute_domains)
        information_gain = base_entropy - attr_entropy
        if information_gain > best_gain + 1e-9 or (
            abs(information_gain - best_gain) <= 1e-9 and attribute < best_attribute
        ):
            best_gain = information_gain
            best_attribute = attribute
    return best_attribute


def tdidt(X, y, available_attributes, attribute_domains, default_label, parent_attribute_size):
    """
    Recursively builds a decision tree using TDIDT.
    """
    if not y:
        return ["Leaf", default_label, 0, parent_attribute_size]

    unique_labels = set(y)
    if len(unique_labels) == 1:
        label = y[0]
        return ["Leaf", label, len(y), parent_attribute_size]

    if not available_attributes:
        label, label_count = majority_label(y, default_label)
        return ["Leaf", label, label_count, parent_attribute_size]

    best_attribute = select_attribute(X, y, available_attributes, attribute_domains)
    node: list = ["Attribute", f"att{best_attribute}"]
    new_available = [att for att in available_attributes if att != best_attribute]
    current_total = len(y)
    majority_lbl, majority_count = majority_label(y, default_label)

    for value in attribute_domains.get(best_attribute, []):
        subset_X, subset_y = partition_dataset(X, y, best_attribute, value)
        if not subset_y:
            leaf = ["Leaf", majority_lbl, majority_count, current_total]
            node.append(["Value", value, leaf])
        else:
            subtree = tdidt(
                subset_X,
                subset_y,
                new_available,
                attribute_domains,
                default_label,
                current_total
            )
            node.append(["Value", value, subtree])
    return node


def predict_instance(instance, node, default_label):
    """
    Traverse a tree to predict a single instance.
    """
    if node is None:
        return default_label

    if node[0] == "Leaf":
        return node[1]

    attribute_name = node[1]
    attribute_index = int(attribute_name.replace("att", ""))
    attribute_value = instance[attribute_index]

    for child in node[2:]:
        if child[1] == attribute_value:
            return predict_instance(instance, child[2], default_label)
    return default_label


def evaluate_feature_subset_cv(
    X,
    y,
    header,
    feature_subset,
    classifier_builder,
    n_splits=10,
    stratify=False,
    positive_label=None,
    class_labels=None
):
    """
    Evaluate a feature subset via cross-validation and return common metrics.
    """
    if not feature_subset:
        raise ValueError("feature_subset must contain at least one attribute.")

    name_to_index = {name: idx for idx, name in enumerate(header)}
    subset_indexes = [name_to_index[name] for name in feature_subset]
    projected_X = project_columns(X, subset_indexes)

    classifier = classifier_builder()
    accuracy, error_rate, y_true, y_pred = cross_val_predict(
        classifier,
        projected_X,
        y,
        n_splits=n_splits,
        random_state=None,
        stratify=stratify
    )

    labels = class_labels or sorted(set(y))
    pos_label = positive_label or labels[0]

    precision = evaluation.binary_precision_score(
        y_true, y_pred, labels=labels, pos_label=pos_label
    )
    recall = evaluation.binary_recall_score(
        y_true, y_pred, labels=labels, pos_label=pos_label
    )
    f1 = evaluation.binary_f1_score(
        y_true, y_pred, labels=labels, pos_label=pos_label
    )
    matrix = evaluation.confusion_matrix(y_true, y_pred, labels)

    return {
        "features": feature_subset,
        "accuracy": accuracy,
        "error_rate": error_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": matrix,
        "labels": labels,
        "y_true": y_true,
        "y_pred": y_pred
    }

def random_subsample(classifier, X, y, k=10, test_size=0.33, random_state=None):
    """
    Perform random subsampling evaluation on a classifier.
    
    Args:
        classifier: A classifier object with fit() and predict() methods
        X(list of list of numeric vals): The feature data
        y(list of obj): The target labels
        k(int): Number of times to repeat random subsampling
        test_size(float or int): Size of test set for each split
        random_state(int): Random seed for reproducibility
    
    Returns:
        tuple: (average_accuracy, average_error_rate, all_accuracies)
    """
    from mysklearn import evaluation
    
    accuracies = []
    
    for i in range(k):
        # set different random state for each iteration if provided
        curr_random_state = None if random_state is None else random_state + i
        
        # split the data
        X_train, X_test, y_train, y_test = evaluation.train_test_split(
            X, y, test_size=test_size, random_state=curr_random_state, shuffle=True
        )
        
        # train and predict
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        # calculate accuracy for this iteration
        accuracy = evaluation.accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    
    # calculate average accuracy and error rate
    avg_accuracy = sum(accuracies) / len(accuracies)
    avg_error_rate = 1.0 - avg_accuracy
    
    return avg_accuracy, avg_error_rate, accuracies

def cross_val_predict(classifier, X, y, n_splits=10, random_state=None, stratify=False):
    """
    Perform k-fold cross validation on a classifier.
    
    Args:
        classifier: A classifier object with fit() and predict() methods
        X(list of list of numeric vals): The feature data
        y(list of obj): The target labels
        n_splits(int): Number of folds for cross validation
        random_state(int): Random seed for reproducibility
        stratify(bool): If True, use stratified k-fold split
    
    Returns:
        tuple: (average_accuracy, average_error_rate, all_predictions)
    """
    from mysklearn import evaluation
    
    # get folds
    if stratify:
        folds = evaluation.stratified_kfold_split(X, y, n_splits=n_splits, 
                                                     random_state=random_state, shuffle=False)
    else:
        folds = evaluation.kfold_split(X, n_splits=n_splits, 
                                         random_state=random_state, shuffle=False)
    
    all_y_true = []
    all_y_pred = []
    
    for train_indices, test_indices in folds:
        # build train and test sets for this fold
        X_train = [X[i] for i in train_indices]
        y_train = [y[i] for i in train_indices]
        X_test = [X[i] for i in test_indices]
        y_test = [y[i] for i in test_indices]
        
        # train and predict
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        # collect predictions
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
    
    # calculate overall accuracy and error rate
    accuracy = evaluation.accuracy_score(all_y_true, all_y_pred)
    error_rate = 1.0 - accuracy
    
    return accuracy, error_rate, all_y_true, all_y_pred

def bootstrap_method(classifier, X, y, k=10, random_state=None):
    """
    Perform bootstrap method evaluation on a classifier.
    
    Args:
        classifier: A classifier object with fit() and predict() methods
        X(list of list of numeric vals): The feature data
        y(list of obj): The target labels
        k(int): Number of bootstrap samples to generate
        random_state(int): Random seed for reproducibility
    
    Returns:
        tuple: (average_accuracy, average_error_rate, all_accuracies)
    """
    from mysklearn import evaluation
    
    accuracies = []
    
    for i in range(k):
        # set different random state for each iteration if provided
        curr_random_state = None if random_state is None else random_state + i
        
        # create bootstrap sample
        X_train, X_test, y_train, y_test = evaluation.bootstrap_sample(
            X, y, random_state=curr_random_state
        )
        
        # train and predict on out-of-bag samples
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        # calculate accuracy for this iteration
        accuracy = evaluation.accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    
    # calculate average accuracy and error rate
    avg_accuracy = sum(accuracies) / len(accuracies)
    avg_error_rate = 1.0 - avg_accuracy
    
    return avg_accuracy, avg_error_rate, accuracies