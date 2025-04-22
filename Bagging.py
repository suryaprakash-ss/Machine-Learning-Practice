import numpy as np
from collections import Counter

def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[indices], y[indices]

class SimpleDecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left_class = None
        self.right_class = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        best_gini = 1.0
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left = y[X[:, feature] <= threshold]
                right = y[X[:, feature] > threshold]
                gini = self.gini_index(left, right)
                if gini < best_gini:
                    best_gini = gini
                    self.feature_index = feature
                    self.threshold = threshold
                    self.left_class = Counter(left).most_common(1)[0][0] if len(left) > 0 else None
                    self.right_class = Counter(right).most_common(1)[0][0] if len(right) > 0 else None

    def predict(self, X):
        predictions = []
        for sample in X:
            if sample[self.feature_index] <= self.threshold:
                predictions.append(self.left_class)
            else:
                predictions.append(self.right_class)
        return np.array(predictions)

    def gini_index(self, left, right):
        def gini(group):
            classes = np.unique(group)
            score = 1.0
            for c in classes:
                p = np.sum(group == c) / len(group)
                score -= p * p
            return score
        n = len(left) + len(right)
        return (len(left) / n) * gini(left) + (len(right) / n) * gini(right)

def train_bagging(X_train, y_train, base_model, n_estimators):
    models = []
    for _ in range(n_estimators):
        X_sample, y_sample = bootstrap_sample(X_train, y_train)
        model = base_model()
        model.fit(X_sample, y_sample)
        models.append(model)
    return models

def predict_bagging(models, X_test):
    predictions = []
    for model in models:
        pred = model.predict(X_test)
        predictions.append(pred)
    predictions = np.array(predictions).T
    final_predictions = []
    for preds in predictions:
        most_common = Counter(preds).most_common(1)[0][0]
        final_predictions.append(most_common)
    return np.array(final_predictions)

X = np.array([
    [2.7, 2.5],
    [1.3, 1.5],
    [3.0, 3.7],
    [2.0, 1.0],
    [1.0, 1.0],
    [2.0, 2.5]
])

y = np.array([0, 0, 1, 1, 0, 1])


X_train, X_test = X[:4], X[4:]
y_train, y_test = y[:4], y[4:]


models = train_bagging(X_train, y_train, SimpleDecisionStump, n_estimators=5)

y_pred = predict_bagging(models, X_test)

accuracy = np.sum(y_pred == y_test) / len(y_test)
print(f"{accuracy:.2f}")