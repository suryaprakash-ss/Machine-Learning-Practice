import numpy as np

def findMean(arr):
    return sum(arr) / len(arr)

def findSlopeAndIntercept(x, y):
    x = np.array(x)
    y = np.array(y).reshape(-1, 1)
    n = len(y)

    intercept = np.c_[np.ones((n, 1)), x]

    slope = np.linalg.inv(intercept.T @ intercept) @ (intercept.T @ y)

    return slope

def LinearRegression(input, slope):
    input = np.array([1] + input)
    return (input @ slope).item()


x = [
    [1, 2],
    [2, 1],
    [3, 4],
    [4, 3],
    [5, 5]
]

y = [2, 3, 6, 7, 8]

slope = findSlopeAndIntercept(x, y)

input = [3, 7]
response = LinearRegression(input, slope)


print(f"Predicted value at x={input} is {response}")