import numpy as np
import matplotlib.pyplot as plt

def gradientDescent(weights, x, y, alpha):
    m = len(y)
    hypothesis = np.dot(x, weights)
    gradient = (1 / m) * np.dot(x.T, (hypothesis - y))
    new_weights = weights - alpha * gradient
    return new_weights

def costFunction(weights, x, y):
    m = len(y)
    hypothesis = np.dot(x, weights)
    cost = (1 / (2 * m)) * np.sum(np.square(hypothesis - y))
    return cost

x = np.array([1, 2, 3, 4, 4, 5])
y = np.array([2, 3, 4, 4, 5, 1])

x = x.reshape(-1, 1)
weights = np.zeros(x.shape[1])
learningRate = 0.01

all_weights = []
all_costs = []

for i in range(100):
    weights = gradientDescent(weights, x, y, learningRate)
    all_weights.append(weights.copy())
    all_costs.append(costFunction(weights, x, y))
    print(f"Iteration {i+1}: Weights: {weights}, Cost: {all_costs[-1]}")

all_weights = np.array(all_weights).flatten()
all_costs = np.array(all_costs)

theta_vals = np.linspace(-2, 2, 100)
cost_vals = []

for t in theta_vals:
    cost = costFunction(np.array([t]), x, y)
    cost_vals.append(cost)

theta_vals, cost_vals = np.array(theta_vals), np.array(cost_vals)

plt.figure(figsize=(8,6))
plt.plot(theta_vals, cost_vals, label="Cost Function")
plt.scatter(all_weights, all_costs, color='red', label="Gradient Descent Path")
plt.xlabel("Theta")
plt.ylabel("Cost")
plt.title("Cost vs Theta")
plt.legend()
plt.grid(True)
plt.show()