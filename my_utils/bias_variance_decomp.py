import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import seaborn as sns

biases, variances, errors = [], [], []


def bias_variance_decomp(model, X, y, n_rounds=50, test_size=0.25):
    predictions = []
    y_tests = []

    for _ in range(n_rounds):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=True
        )
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        predictions.append(preds)
        y_tests.append(y_test)

    predictions = np.array(predictions)
    y_tests = np.array(y_tests)

    avg_preds = predictions.mean(axis=0)

    bias = np.mean((y_tests.mean(axis=0) - avg_preds) ** 2)
    variance = np.mean(np.var(predictions, axis=0))
    error = bias + variance
    
    return bias, variance, error


alpha_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]


# for alpha in alpha_range:
#     bias, variance, error = bias_variance_decomp(Ridge(alpha), features, target, test_size=0.25)
#     biases.append(bias)
#     variances.append(variance)
#     errors.append(error)
#     print(f'alpha = {alpha} \t Bias: {bias:.0f} | Variance: {variance:.0f} | Error: {error:.0f}')


sns.lineplot(x=alpha_range, y=biases,  marker='o', label='Bias')
sns.lineplot(x=alpha_range, y=variances, marker='o', label="Variance")
sns.lineplot(x=alpha_range, y=errors, marker='o', label="Total Error", linewidth=2)
plt.xscale('log')  # log scale helps since alphas vary exponentially
plt.xlabel("Alpha (log scale)")
plt.ylabel("Error")
plt.title("Bias-Variance Tradeoff for Ridge Regression")
plt.legend()
plt.grid(True)
plt.show()
