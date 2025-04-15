import numpy as np

def calculateSigmoid(z):
    return 1 / (1 + np.exp(-z))

def costFunction(x, y, weights):
    n = len(y)
    prediction = calculateSigmoid(np.dot(x, weights))
    cost = - (1 / n) * np.sum(y * np.log(prediction) + (1 -y) * np.log(1 - prediction))
    return cost

def findGradientDescent(x, y, weights, learningRate, nIter):
    n = len(y)
    costHistory = []

    for i in range(nIter):
        pred = calculateSigmoid(np.dot(x , weights))
        gradient = (1/n) * np.dot(x.T, (pred - y))
        weights = weights - learningRate * gradient

        costHistory.append(costFunction(x, y, weights))
    return weights, costHistory

def predict(x, theta):
    pred = calculateSigmoid(np.dot(x, theta))
    predClass = [1 if p >= 0.5 else 0 for p in pred]
    return predClass

x = np.array([[1, 2, 4], [2, 3, 6], [3, 4, 7], [4, 5, 6], [5, 6, 9]])  
y = np.array([0, 0, 1, 1, 1]) 
weights = np.zeros(x.shape[1])
learningRate = 0.1
nIter = 1000

weights, costHistory = findGradientDescent(x, y, weights, learningRate, nIter)

input = np.array([[14, 1, 12]])
response = predict(input, weights)

print(f"Logistic Regression Classification: {response}")