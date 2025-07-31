import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from my_utils.regression_eval_plots import EvaluationPlots

class CustomLinearRegression:
    def __init__(self, number_of_iterations=1000, learning_rate=0.1, regularize=False, regularization_strength=0.1, verbose=False, log_every=100, tolerance = 0.001):
        self.weights : np.ndarray = None  # (n_features, )
        self.bias : float = None  # scalar
        self.number_of_iterations : int = number_of_iterations
        self.learning_rate : float = learning_rate
        self.regularize = regularize
        self.regularization_strength = regularization_strength
        self.final_cost : float = 0.0
        self.cost_history : list[float] = []
        self.verbose : bool = verbose
        self.log_every : int = log_every
        self.tolerance : float = tolerance

    def _init_parameters(self, number_of_features):
        self.weights = np.random.uniform(-0.01, 0.01, number_of_features)
        self.bias = 0

    def _compute_loss(self, y, y_hat) -> float:
        number_of_samples = y.shape[0]
        error = y - y_hat
        mean_squared_error = np.mean(error ** 2)

        if self.regularize:
            l2 = self.regularization_strength / number_of_samples * np.sum(self.weights ** 2) 
            return 1/2 * (mean_squared_error + l2)
        
        return 1/2 * mean_squared_error
    
    def _forward_propagation(self, X, y):
        y_hat = X @ self.weights + self.bias
        cost = self._compute_loss(y, y_hat)
        self.cost_history.append(cost)
        return y_hat, cost
    
    def _backward_propagation(self, X, y, y_hat):
        number_of_samples = X.shape[0]
        error = y_hat - y

        bias_derivative = np.mean(error)

        # weights gradient without regulariztion
        weights_gradient = 1 / number_of_samples * (X.T @ error)

        # add L2 regularization if enabled
        if self.regularize:
            weights_gradient += self.regularization_strength / number_of_samples * self.weights
        
        return weights_gradient, bias_derivative
    
    def _update(self, weights_gradient, bias_derivative):
        self.weights = self.weights - self.learning_rate * weights_gradient
        self.bias = self.bias - self.learning_rate * bias_derivative

    def fit(self, X, y):
        self._init_parameters(X.shape[1])
        previous_cost = np.inf

        for iteration in range(self.number_of_iterations):
            y_hat, cost = self._forward_propagation(X, y)
            
            if abs(previous_cost - cost) < self.tolerance:
                print(f'Early stopping at iteration: {iteration} due to ΔCost being less {self.tolerance}')
                break

            if self.verbose and (iteration % self.log_every == 0 or iteration == self.number_of_iterations - 1):
                print(f'Iteration: {iteration} | Cost: {cost:.5f} | Weights Norm: {np.linalg.norm(self.weights)}')
            
            previous_cost = cost
            weights_derivative, bias_derivative = self._backward_propagation(X, y, y_hat)
            self._update(weights_derivative, bias_derivative)

 
        self.final_cost = cost
        return self

    def predict(self, X):
        predictions = X @ self.weights + self.bias
        return predictions
    
    def get_parameters(self):
        return self.weights, self.bias
    
    def plot_losses(self, ax=None):
        if ax is None:
            ax = plt.gca()
        
        sns.lineplot(x=range(len(self.cost_history)), y=self.cost_history, label=f'Final cost: {self.final_cost:.5f}', ax=ax)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Cost')
        ax.set_title('Training Cost Vs Iteration')
        ax.grid()
        ax.legend()

    # Not a useful plot unless data is a time series

    # @staticmethod
    # def plot_residuals(residuals, ax=None):
    #     if ax is None:
    #         ax = plt.gca()

    #     ax.scatter(x=np.arange(len(residuals)), y=residuals, alpha=0.7)
    #     ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
        
    #     ax.set_xlabel('X axis')
    #     ax.set_ylabel('Residuals')
    #     ax.set_title('Residuals plot')
    #     ax.grid()

    