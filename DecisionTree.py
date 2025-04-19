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

def best_split(features, labels, criterion='information_gain'):
    best_ig = -float('inf')
    best_split_value = None
    best_feature = None
    features = np.array(features)

    for feature_index in range(features.shape[1]):
        feature_column = features[:, feature_index]
        possible_splits = np.unique(feature_column)

        for split_value in possible_splits:
            if criterion == 'information_gain':
                ig = information_gain(feature_column, labels, split_value)
            elif criterion == 'gini':
                g = gini(feature_column)
                ig = g
            elif criterion == 'variance_reduction':
                vr = variance_reduction(feature_column, labels, split_value)
                ig = vr
            if ig > best_ig:
                best_ig = ig
                best_split_value = split_value
                best_feature = feature_index

    return best_feature, best_split_value, best_ig

def make_prediction(features, labels):
    best_feature, best_split_value, best_ig = best_split(features, labels)
    if best_feature is None:
        return None, None
    
    print(f"Best Feature: {best_feature}, \nBest Split: {best_split_value}, \nInformation Gain: {best_ig}")
    
    features = np.array(features)
    left_indices = features[:, best_feature] <= best_split_value
    right_indices = features[:, best_feature] > best_split_value

    labels = np.array(labels)
    left_labels = labels[left_indices]
    right_labels = labels[right_indices]

    left_prediction = np.bincount(left_labels).argmax()
    right_prediction = np.bincount(right_labels).argmax()
    return left_prediction, right_prediction

features = [[5, 2, 9, 1, 6, 3],
            [8, 7, 1, 4, 5, 2],
            [3, 5, 7, 6, 2, 8],
            [9, 1, 4, 7, 8, 5],
            [2, 8, 6, 3, 7, 1],
            [4, 3, 2, 8, 5, 7],
            [7, 6, 5, 2, 1, 9],
            [1, 9, 8, 5, 4, 3],
            [6, 2, 3, 7, 9, 1],
            [5, 7, 4, 2, 3, 6]]

labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

left_pred, right_pred = make_prediction(features, labels)

print(f"Left Prediction: {left_pred} \nRight Prediction: {right_pred}")