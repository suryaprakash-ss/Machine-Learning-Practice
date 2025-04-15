import numpy as np
from sklearn.datasets import load_iris

def findSigmoid(z):
    return 1 / (1 + np.exp(-z))

def findLoss(x, y, weights):
    n = len(y)
    pred = findSigmoid(np.dot(x, weights))
    loss = - (1 / n) * np.sum(y * np.log(pred) + (1 - y) * np.log(1 - pred))
    return loss

def computeGradient(x, y, weights):
    n = len(y)
    pred = findSigmoid(np.dot(x, weights))
    gradient = (1 / n) * np.dot(x.T, (pred - y))
    return gradient

def gradientDescent(x, y, lr=0.1, iter=1000):
    weights = np.zeros(x.shape[1])
    for i in range(iter):
        gradient = computeGradient(x, y, weights)
        weights -= lr * gradient
    return weights

df = load_iris()
x = df["data"]
y = df["target"]

mask = y < 2
x = x[mask]
y = y[mask]

weights = gradientDescent(x, y)

input = np.array([[1.2, 6.2, 3.4, 5.4]])
response = findSigmoid(np.dot(input, weights))
predClass = (response >= 0.5).astype(int)

print("Prediction (probability):", response)
print("Predicted Class:", predClass)