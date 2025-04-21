import numpy as np
import random

def variance(y):
    mean = np.mean(y)
    return np.mean((y - mean) ** 2)

def variance_reduction(X_column, y, split_value):
    parent_variance = variance(y)
    left_indices = X_column <= split_value
    right_indices = X_column > split_value

    y = np.array(y)
    n = len(y)

    n_left, n_right = left_indices.sum(), right_indices.sum()
    if n_left == 0 or n_right == 0:
        return 0

    left_variance = variance(y[left_indices])
    right_variance = variance(y[right_indices])

    child_variance = (n_left / n) * left_variance + (n_right / n) * right_variance
    vr = parent_variance - child_variance
    return vr

def best_split(features, labels, feature_indices):
    best_vr = -float('inf')
    best_split_value = None
    best_feature = None
    features = np.array(features)
    labels = np.array(labels)

    for feature_index in feature_indices:
        feature_column = features[:, feature_index]
        possible_splits = np.unique(feature_column)

        for split_value in possible_splits:
            vr = variance_reduction(feature_column, labels, split_value)
            if vr > best_vr:
                best_vr = vr
                best_split_value = split_value
                best_feature = feature_index

    return best_feature, best_split_value

def build_tree(X, y, feature_indices):
    X = np.array(X)
    y = np.array(y)

    if len(y) <= 1:
        return np.mean(y)

    if np.all(y == y[0]):
        return y[0]

    best_feature, best_split_value = best_split(X, y, feature_indices)
    if best_feature is None:
        return np.mean(y)

    left_indices = X[:, best_feature] <= best_split_value
    right_indices = X[:, best_feature] > best_split_value

    left_subtree = build_tree(X[left_indices], y[left_indices], feature_indices)
    right_subtree = build_tree(X[right_indices], y[right_indices], feature_indices)

    return (best_feature, best_split_value, left_subtree, right_subtree)

def predict_tree(tree, x):
    if not isinstance(tree, tuple):
        return tree

    feature_index, split_value, left_subtree, right_subtree = tree
    if x[feature_index] <= split_value:
        return predict_tree(left_subtree, x)
    else:
        return predict_tree(right_subtree, x)

class RandomForest:
    def __init__(self, n_estimators=5, max_features=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape

        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))

        self.trees = []

        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            feature_indices = random.sample(range(n_features), self.max_features)

            tree = build_tree(X_sample, y_sample, feature_indices)
            self.trees.append((tree, feature_indices))

    def predict(self, X):
        X = np.array(X)
        tree_preds = np.array([self._predict_tree(tree, features, X) for tree, features in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [np.mean(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)

    def _predict_tree(self, tree, features, X):
        return [predict_tree(tree, x) for x in X]

features = [
    [5, 2, 9, 1, 6, 3],
    [8, 7, 1, 4, 5, 2],
    [3, 5, 7, 6, 2, 8],
    [9, 1, 4, 7, 8, 5],
    [2, 8, 6, 3, 7, 1],
    [4, 3, 2, 8, 5, 7],
    [7, 6, 5, 2, 1, 9],
    [1, 9, 8, 5, 4, 3],
    [6, 2, 3, 7, 9, 1],
    [5, 7, 4, 2, 3, 6]
]

labels = [5.5, 8.0, 3.2, 9.1, 2.5, 4.7, 7.3, 1.8, 6.4, 5.2]

rf = RandomForest(n_estimators=5, max_features=3)
rf.fit(features, labels)
predictions = rf.predict(features)

print("Predictions:", predictions)