def findMean(arr):
    return sum(arr) / len(arr)

def findSlopeAndIntercept(x, y):
    xMean = findMean(x)
    yMean = findMean(y)

    numerator = sum((x[i] - xMean) * (y[i] - yMean) for i in range(len(x)))
    denominator = sum((x[i] - xMean) ** 2 for i in range(len(x)))

    m = numerator / denominator
    c = yMean - m * xMean

    return m, c

def LinearRegression(x, m, c):
    return m * x + c

x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]

slope, intercept = findSlopeAndIntercept(x, y)

input = 6
response = LinearRegression(input, slope, intercept)

print(f"Predicted value at x={input} is {response}")