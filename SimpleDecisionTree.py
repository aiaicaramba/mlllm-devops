import numpy as np
from collections import Counter


class SimpleDecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes = len(set(y))
        self.n_features = X.shape[1]
        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(set(y))

        # Stopping criteria
        if depth >= self.max_depth or n_labels == 1 or n_samples < 2:
            leaf_value = Counter(y).most_common(1)[0][0]
            return {"value": leaf_value}

        # Find the best split
        best_feature, best_threshold = self._best_split(X, y)

        # Split the data
        left_idxs = X[:, best_feature] < best_threshold
        right_idxs = ~left_idxs

        # Grow the children that result from the split
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return {
            "feature_index": best_feature,
            "threshold": best_threshold,
            "left": left,
            "right": right,
        }

    def _best_split(self, X, y):
        m = self.n_features
        best_gain = -1
        split_idx, split_thresh = None, None

        for idx in range(m):
            thresholds = np.unique(X[:, idx])
            for thresh in thresholds:
                gain = self._information_gain(y, X[:, idx], thresh)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = idx
                    split_thresh = thresh

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        # Parent entropy
        parent_entropy = self._entropy(y)

        # Generate split
        left_idxs = X_column < split_thresh
        right_idxs = ~left_idxs

        if len(y[left_idxs]) == 0 or len(y[right_idxs]) == 0:
            return 0

        # Weighted avg. of children entropy
        n = len(y)
        n_l, n_r = len(y[left_idxs]), len(y[right_idxs])
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # Return information gain
        return parent_entropy - child_entropy

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _predict(self, inputs):
        node = self.tree
        while "value" not in node:
            if inputs[node["feature_index"]] < node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["value"]


# Example usage
if __name__ == "__main__":
    # Prepare data
    X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    y = np.array([0, 1, 1, 0])

    # Create and train the model
    tree = SimpleDecisionTree(max_depth=3)
    tree.fit(X, y)

    # Make predictions
    test_X = np.array([[1, 1], [1, 0], [0, 0]])
    predictions = tree.predict(test_X)
    print("Predictions:", predictions)
