import numpy as np
import random

def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    prob = counts / counts.sum()
    return -np.sum(prob * np.log2(prob))

def information_gain(X_column, y, split_value):
    parent_entropy = entropy(y)
    left_indices = X_column <= split_value
    right_indices = X_column > split_value

    y = np.array(y)
    n = len(y)

    n_left, n_right = left_indices.sum(), right_indices.sum()
    if n_left == 0 or n_right == 0:
        return 0

    left_entropy = entropy(y[left_indices])
    right_entropy = entropy(y[right_indices])

    child_entropy = (n_left / n) * left_entropy + (n_right / n) * right_entropy
    ig = parent_entropy - child_entropy
    return ig

def best_split(features, labels, feature_indices):
    best_ig = -float('inf')
    best_split_value = None
    best_feature = None
    features = np.array(features)
    labels = np.array(labels)

    for feature_index in feature_indices:
        feature_column = features[:, feature_index]
        possible_splits = np.unique(feature_column)

        for split_value in possible_splits:
            ig = information_gain(feature_column, labels, split_value)
            if ig > best_ig:
                best_ig = ig
                best_split_value = split_value
                best_feature = feature_index

    return best_feature, best_split_value

def build_tree(X, y, feature_indices):
    X = np.array(X)
    y = np.array(y)

    if len(set(y)) == 1:
        return y[0]

    if X.shape[0] == 0:
        return None

    best_feature, best_split_value = best_split(X, y, feature_indices)
    if best_feature is None:
        return np.bincount(y).argmax()

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
        y_pred = [np.bincount(tree_pred).argmax() for tree_pred in tree_preds]
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

labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

rf = RandomForest(n_estimators=5, max_features=3)
rf.fit(features, labels)
predictions = rf.predict(features)

print("Predictions:", predictions)