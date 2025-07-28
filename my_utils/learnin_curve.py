from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(model, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=3, scoring='neg_mean_squared_error', ax=None):
    if ax is None:
        ax = plt.gca()

    train_sizes, train_scores, val_scores = learning_curve(
        estimator=model,
        X=X,
        y=y,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        shuffle=True,
        random_state=42
    )

    # Convert negative MSE to positive
    train_errors = -np.mean(train_scores, axis=1)
    val_errors = -np.mean(val_scores, axis=1)

    ax.plot(train_sizes, train_errors, label='Training MSE', marker='o')
    ax.plot(train_sizes, val_errors, label='Validation MSE', marker='s')
    ax.set_title(f'Learning Curve\n{str(model)}')
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Mean Squared Error')
    ax.legend()
    ax.grid()