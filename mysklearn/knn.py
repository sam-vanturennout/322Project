class KNeighborsClassifier:
    """
    Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """
        Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """
        Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def euclidean(self, a, b):
        """
        Compute Euclidean distance between two equal-length vectors."""
        # No imports; do it manually
        s = 0.0
        for i in range(len(a)):
            diff = (a[i] - b[i])
            s += diff * diff
        # Using exponent 0.5 instead of math.sqrt to avoid extra import
        return s ** 0.5

    def kneighbors(self, X_test):
        """
        Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model not fitted. Call fit() before kneighbors().")

        distances_all = []
        indices_all = []

        for x in X_test:
            # compute all distances to training points
            dists = []
            for idx, xtr in enumerate(self.X_train):
                d = self.euclidean(x, xtr)
                dists.append((d, idx))

            # Sort by distance, then by index for determinism
            dists.sort(key=lambda t: (t[0], t[1]))

            # Take the first k neighbors
            k = min(self.n_neighbors, len(dists))
            k_dists = [dists[i][0] for i in range(k)]
            k_idxs = [dists[i][1] for i in range(k)]

            distances_all.append(k_dists)
            indices_all.append(k_idxs)

        return distances_all, indices_all

    def predict(self, X_test):
        """
        Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Must call fit() before predict().")
        distances_all, indices_all = self.kneighbors(X_test)
        y_pred = []

        for neighbor_indices, neighbor_distances in zip(indices_all, distances_all):
            # collect neighbor labels in order (closest first)
            labels = [self.y_train[i] for i in neighbor_indices]

            # majority vote
            counts = {}
            for lab in labels:
                counts[lab] = counts.get(lab, 0) + 1

            # determine the winner with deterministic tie-breaking:
            # 1) higher count, 2) smaller summed distances among its neighbors,
            # 3) first appearance in labels (closest neighbor)
            best_label = None
            best_tuple = None  # (-count, sum_dist, first_pos) for min()
            for lab in counts:
                count = counts[lab]
                # sum of distances for this label among the neighbors
                sum_dist = 0.0
                first_pos = None
                for pos, (lab_i, dist_i) in enumerate(zip(labels, neighbor_distances)):
                    if lab_i == lab:
                        sum_dist += dist_i
                        if first_pos is None:
                            first_pos = pos
                # sort by: (-count) to prefer higher counts,
                # then by (sum_dist) lower is better,
                # then by (first_pos) lower is better.
                key = (-count, sum_dist, first_pos)
                if best_tuple is None or key < best_tuple:
                    best_tuple = key
                    best_label = lab

            y_pred.append(best_label)

        return y_pred