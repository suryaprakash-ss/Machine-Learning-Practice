import numpy as np

class AdaBoost:
    def __init__(self, n_estimators=5):
        self.n_estimators = n_estimators
        self.alphas = []
        self.models = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        w = np.full(n_samples, (1 / n_samples))
        
        for i in range(self.n_estimators):
            stump = DecisionStump()
            stump.fit(X, y, sample_weight=w)
            predictions = stump.predict(X)
            err = np.sum(w * (predictions != y))
            alpha = 0.5 * np.log((1 - err) / (err + 1e-10))
            w *= np.exp(-alpha * y * predictions)
            w /= np.sum(w)
            self.models.append(stump)
            self.alphas.append(alpha)
            print(f"Stump {i+1}: Predictions = {predictions}, Alpha = {alpha:.4f}")

    def predict(self, X):
        final_pred = np.zeros(X.shape[0])
        for alpha, model in zip(self.alphas, self.models):
            final_pred += alpha * model.predict(X)
        return np.sign(final_pred)

class DecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = 1

    def fit(self, X, y, sample_weight):
        n_samples, n_features = X.shape
        min_error = float('inf')
        for feature_i in range(n_features):
            feature_values = X[:, feature_i]
            thresholds = np.unique(feature_values)
            for threshold in thresholds:
                for polarity in [1, -1]:
                    pred = np.ones(n_samples)
                    pred[polarity * feature_values < polarity * threshold] = -1
                    error = np.sum(sample_weight[pred != y])
                    if error < min_error:
                        min_error = error
                        self.polarity = polarity
                        self.threshold = threshold
                        self.feature_index = feature_i

    def predict(self, X):
        feature_values = X[:, self.feature_index]
        predictions = np.ones(X.shape[0])
        predictions[self.polarity * feature_values < self.polarity * self.threshold] = -1
        return predictions

# 2D Input Dataset
X = np.array([
    [1, 2],
    [2, 1],
    [2, 3],
    [3, 2],
    [3, 4],
    [4, 3],
    [5, 5]
])

y = np.array([-1, -1, 1, 1, 1, 1, -1])

model = AdaBoost(n_estimators=4)
model.fit(X, y)
final_preds = model.predict(X)
print(f"Final Prediction: {final_preds}")