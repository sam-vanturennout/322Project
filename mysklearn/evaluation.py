"""
Programmer: Cooper Braun Sam Vanturennout
Class: CPSC 322-01, Fall 2025
Final Project
12/1/25
Description: Evaluation helpers for our classifiers. Provides train/test split,
standard and stratified k-fold utilities, bootstrap sampling, confusion matrices,
and binary metrics (accuracy, precision, recall, F1) to summarize experiments.
"""

import numpy as np

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """
    Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    
    # determine the number of test samples
    if isinstance(test_size, float):
        n_test = int(np.ceil(n_samples * test_size))
    else:
        n_test = test_size
    
    # create indices
    indices = list(range(n_samples))
    
    # shuffle if needed
    if shuffle:
        np.random.shuffle(indices)
        # when shuffled, test comes from beginning
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
    else:
        # when not shuffled, test comes from end
        train_indices = indices[:n_samples - n_test]
        test_indices = indices[n_samples - n_test:]
    
    # build train and test sets
    X_train = [X[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_train = [y[i] for i in train_indices]
    y_test = [y[i] for i in test_indices]
    
    return X_train, X_test, y_train, y_test

def kfold_split(X, n_splits=5, random_state=None, shuffle=False):
    """
    Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    indices = list(range(n_samples))
    
    # shuffle if needed
    if shuffle:
        np.random.shuffle(indices)
    
    # calculate fold sizes
    fold_size = n_samples // n_splits
    remainder = n_samples % n_splits
    
    folds = []
    start = 0
    
    for i in range(n_splits):
        # first 'remainder' folds get an extra element
        current_fold_size = fold_size + (1 if i < remainder else 0)
        end = start + current_fold_size
        
        # test indices for this fold
        test_indices = indices[start:end]
        
        # train indices are all other indices
        train_indices = indices[:start] + indices[end:]
        
        folds.append((train_indices, test_indices))
        start = end
    
    return folds

def stratified_kfold_split(X, y, n_splits=5, random_state=None, shuffle=False):
    """
    Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    
    # create list of (index, label) pairs
    indices_with_labels = list(enumerate(y))
    
    # shuffle all indices together if needed
    if shuffle:
        np.random.shuffle(indices_with_labels)
    
    # group indices by class label after shuffling
    label_to_indices = {}
    for i, label in indices_with_labels:
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(i)
    
    # split each class's indices into n_splits folds
    label_folds = {}
    for label, indices in label_to_indices.items():
        label_folds[label] = []
        n_label_samples = len(indices)
        fold_size = n_label_samples // n_splits
        remainder = n_label_samples % n_splits
        
        start = 0
        for i in range(n_splits):
            current_fold_size = fold_size + (1 if i < remainder else 0)
            end = start + current_fold_size
            label_folds[label].append(indices[start:end])
            start = end
    
    # combine folds across all classes
    folds = []
    for i in range(n_splits):
        test_indices = []
        for label in label_folds:
            test_indices.extend(label_folds[label][i])
        
        train_indices = []
        for j in range(n_splits):
            if j != i:
                for label in label_folds:
                    train_indices.extend(label_folds[label][j])
        
        folds.append((train_indices, test_indices))
    
    return folds

def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """
    Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results

    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
        Sample indexes of X with replacement, then build X_sample and X_out_of_bag
            as lists of instances using sampled indexes (use same indexes to build
            y_sample and y_out_of_bag)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    if n_samples is None:
        n_samples = len(X)
    
    # sample indices with replacement
    sampled_indices = np.random.choice(len(X), size=n_samples, replace=True)
    
    # track which indices were sampled
    sampled_set = set(sampled_indices)
    
    # build sample and out-of-bag sets
    X_sample = [X[i] for i in sampled_indices]
    X_out_of_bag = [X[i] for i in range(len(X)) if i not in sampled_set]
    
    if y is not None:
        y_sample = [y[i] for i in sampled_indices]
        y_out_of_bag = [y[i] for i in range(len(y)) if i not in sampled_set]
    else:
        y_sample = None
        y_out_of_bag = None
    
    return X_sample, X_out_of_bag, y_sample, y_out_of_bag

def confusion_matrix(y_true, y_pred, labels):
    """
    Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    # initialize matrix with zeros
    n_labels = len(labels)
    matrix = [[0 for _ in range(n_labels)] for _ in range(n_labels)]
    
    # create label to index mapping
    label_to_index = {label: i for i, label in enumerate(labels)}
    
    # fill the matrix
    for true_label, pred_label in zip(y_true, y_pred):
        true_index = label_to_index[true_label]
        pred_index = label_to_index[pred_label]
        matrix[true_index][pred_index] += 1
    
    return matrix

def accuracy_score(y_true, y_pred, normalize=True):
    """
    Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    
    if normalize:
        return correct / len(y_true)
    else:
        return correct

def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    # determine labels if not provided
    if labels is None:
        labels = list(set(y_true))
    
    # determine positive label if not provided
    if pos_label is None:
        pos_label = labels[0]
    
    # calculate TP and FP
    tp = 0  # true positives
    fp = 0  # false positives
    
    for true_val, pred_val in zip(y_true, y_pred):
        if pred_val == pos_label:
            if true_val == pos_label:
                tp += 1
            else:
                fp += 1
    
    # precision = tp / (tp + fp)
    if tp + fp == 0:
        return 0.0
    
    return tp / (tp + fp)

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    # determine labels if not provided
    if labels is None:
        labels = list(set(y_true))
    
    # determine positive label if not provided
    if pos_label is None:
        pos_label = labels[0]
    
    # calculate TP and FN
    tp = 0  # true positives
    fn = 0  # false negatives
    
    for true_val, pred_val in zip(y_true, y_pred):
        if true_val == pos_label:
            if pred_val == pos_label:
                tp += 1
            else:
                fn += 1
    
    # recall = tp / (tp + fn)
    if tp + fn == 0:
        return 0.0
    
    return tp / (tp + fn)

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    # calculate precision and recall
    precision = binary_precision_score(y_true, y_pred, labels=labels, pos_label=pos_label)
    recall = binary_recall_score(y_true, y_pred, labels=labels, pos_label=pos_label)
    
    # F1 = 2 * (precision * recall) / (precision + recall)
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)

def cross_val_predict_labels(X, y, clf, n_splits=10, stratified = True, random_state=None):
    """
    Perform k-fold cross validation and return the predicted labels for each instance.

   """
    n_samples = len(X)
    y_pred = [None] * n_samples
    if stratified:
        folds = stratified_kfold_split(X, y, n_splits=n_splits, random_state=random_state, shuffle=True)
    else:
        folds = kfold_split(X, n_splits=n_splits, random_state=random_state, shuffle=True)

    for train_indices, test_indices in folds:
        X_train = [X[i] for i in train_indices]
        y_train = [y[i] for i in train_indices]
        X_test = [X[i] for i in test_indices]

        clf.fit(X_train, y_train)
        y_test_pred = clf.predict(X_test)

        for i, index in enumerate(test_indices):
            y_pred[index] = y_test_pred[i]

    return y_pred
