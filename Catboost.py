import numpy as np

class SimpleCatBoost:
    def __init__(self, n_estimators=5, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.gammas = []

    def fit(self, X, y):
        y_pred = np.zeros(len(y))
        
        for i in range(self.n_estimators):
            residuals = y - self.sigmoid(y_pred)
            stump = DecisionStump()
            stump.fit(X, residuals)
            preds = stump.predict(X)
            gamma = self.learning_rate
            y_pred += gamma * preds
            self.models.append(stump)
            self.gammas.append(gamma)
            print(f"Tree {i+1}: \n Residuals = {residuals}, \nPrediction Update = {preds}")

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for gamma, model in zip(self.gammas, self.models):
            y_pred += gamma * model.predict(X)
        return np.sign(self.sigmoid(y_pred) - 0.5)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

class DecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left_value = None
        self.right_value = None

    def fit(self, X, residuals):
        n_samples, n_features = X.shape
        best_mse = float('inf')
        
        for feature_i in range(n_features):
            thresholds = np.unique(X[:, feature_i])
            for threshold in thresholds:
                left = residuals[X[:, feature_i] <= threshold]
                right = residuals[X[:, feature_i] > threshold]
                mse = (left.var() * len(left) + right.var() * len(right)) / n_samples
                
                if mse < best_mse:
                    best_mse = mse
                    self.feature_index = feature_i
                    self.threshold = threshold
                    self.left_value = np.mean(left) if len(left) > 0 else 0
                    self.right_value = np.mean(right) if len(right) > 0 else 0

    def predict(self, X):
        feature = X[:, self.feature_index]
        preds = np.where(feature <= self.threshold, self.left_value, self.right_value)
        return preds

# Example dataset
X = np.array([
    [1, 2],
    [2, 1],
    [2, 3],
    [3, 2],
    [3, 4],
    [4, 3],
    [5, 5]
])

y = np.array([0, 0, 1, 1, 1, 1, 0])  # Labels must be 0 or 1 for sigmoid

model = SimpleCatBoost(n_estimators=3, learning_rate=0.5)
model.fit(X, y)
final_preds = model.predict(X)
print(f"Final Prediction: {final_preds}")