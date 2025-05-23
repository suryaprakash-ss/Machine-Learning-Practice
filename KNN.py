import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        preds = [self._predict(x) for x in X]
        return np.array(preds)

    def _predict(self, x):
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

# Example dataset
X_train = np.array([
    [1, 2],
    [2, 3],
    [3, 1],
    [6, 5],
    [7, 7],
    [8, 6]
])
y_train = np.array([0, 0, 0, 1, 1, 1])

X_test = np.array([
    [5, 5],
    [2, 2]
])

model = KNN(k=3)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(f"Predictions: {predictions}")