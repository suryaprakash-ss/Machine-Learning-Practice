import numpy as np

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

def gini(y):
    classes, counts = np.unique(y, return_counts=True)
    prob = counts / counts.sum()

    return 1 - np.sum(prob**2)

def variance(y):
    mean = np.mean(y)

    return np.mean((y - mean) ** 2)