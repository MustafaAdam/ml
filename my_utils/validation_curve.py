from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
import numpy as np

def plot_validation_curve(model, X, y, param_name, param_range, cv=5, ax=None, scoring='accuracy'):
    """
        Plot a validation curve to evaluate model performance across different values of a specified hyperparameter.

        Parameters
        ----------
            model : estimator object
                The machine learning model (must implement `fit` and `predict` methods).
                
            X : array-like of shape (n_samples, n_features)
                The input training data.
                
            y : array-like of shape (n_samples,)
                The target values.
                
            param_name : str
                The name of the hyperparameter to vary (must match the model's parameter name).
                
            param_range : array-like
                The range of values for the hyperparameter.
                
            cv : int, default=5
                Number of cross-validation folds to use.
                
            ax : matplotlib.axes.Axes, optional
                Axes object to draw the plot onto. If None, the current active axes will be used.
                
            scoring : str, default='accuracy'
                Performance metric to evaluate. Commonly used options in practice are:

                For regression:
                    - 'r2' : R² score (variance explained, default for regression)
                    - 'neg_mean_squared_error' : Negative MSE (lower is better)
                    - 'neg_mean_absolute_error' : Negative MAE (robust to outliers)

                For classification:
                    - 'accuracy' : Proportion of correct predictions (default for classification)
                    - 'f1' : F1 score (harmonic mean of precision and recall)
                    - 'roc_auc' : Area Under ROC Curve (ranking quality, especially for imbalanced data)

            Returns
        -------
            None
                The function plots the validation curve on the provided or active Matplotlib axes.

        Examples
        --------
            >>> from sklearn.datasets import load_digits
            >>> from sklearn.linear_model import Ridge
            >>> import matplotlib.pyplot as plt
            >>> X, y = load_digits(return_X_y=True)
            >>> fig, ax = plt.subplots()
            >>> plot_validation_curve(
            ...     model=Ridge(),
            ...     X=X,
            ...     y=y,
            ...     param_name='alpha',
            ...     param_range=[0.001, 0.01, 0.1, 1, 10, 100],
            ...     cv=5,
            ...     ax=ax,
            ...     scoring='r2'
            ... )
            >>> plt.show()
    """
    
    if ax is None:
        ax = plt.gca()

    train_scores, validation_scores = validation_curve(
        model, X, y, param_name=param_name, param_range=param_range, cv=cv, scoring=scoring)

    training_means = np.mean(train_scores, axis=1)
    training_stds = np.std(train_scores, axis=1)
    validation_means = np.mean(validation_scores, axis=1)
    validation_stds = np.std(validation_scores, axis=1)

    # Handle negative scoring metrics
    if scoring.startswith('neg_'):
        training_means, validation_means = -training_means, -validation_means
        training_stds, validation_stds = training_stds, validation_stds
        ylabel = scoring.replace("neg_", "").replace("_", " ").title()
    else:
        ylabel = scoring.replace("_", " ").title()


    # plot training scores
    ax.plot(param_range, training_means, label='Training score', color='blue')
    ax.fill_between(param_range, training_means - training_stds, training_means + training_stds, 
                    alpha=0.1, color='blue')


    # Plot validation scores
    ax.plot(param_range, validation_means, label='Validation Score', color='orange')
    ax.fill_between(param_range, validation_means - validation_stds, validation_means + validation_stds, 
                    alpha=0.1, color='orange')

    # log scale is useful for parameters like 'C' and 'alpha', but not all
    log_scale = param_range[0] > 0 and max(param_range) / min(param_range) > 100
    if log_scale:
        ax.set_xscale('log')

    # these scoring methods only go from 0 to 1. so keep the the y axis limited to this range
    if scoring in ['accuracy', 'f1', 'roc_auc', 'r2']:
        ax.set_ylim(0, 1)

    # turn the scientific notation (10-3) into regular floats (0.001)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:g}'))
    ax.set_xlabel(f'{param_name}')
    ax.set_ylabel(f"{ylabel}")
    ax.set_title('Validation Curve')
    ax.legend()
    ax.grid()
    
    ### values to return if needed (optional)
    # Best parameter (max validation mean score)
    best_index = np.argmax(validation_means)
    best_param = param_range[best_index]

    results = {
        "param_range": param_range,
        "train_mean": training_means,
        "train_std": training_stds,
        "val_mean": validation_means,
        "val_std": validation_stds,
    }

    return best_param, results

