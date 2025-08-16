from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(model, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=3,
                        scoring='neg_mean_squared_error', ax=None, show_error_bands=False,
                         label='', model_name=''
                        ):

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
        random_state=42,
        n_jobs=-1,
    )
    
    training_scores_mean = np.mean(train_scores, axis=1)
    validation_scores_mean = np.mean(val_scores, axis=1)

    if 'neg' in scoring:
        training_scores_mean = -training_scores_mean
        validation_scores_mean = -validation_scores_mean
    
    ax.plot(train_sizes, training_scores_mean, label=f'Training {label}', marker='o')
    ax.plot(train_sizes, validation_scores_mean, label=f'Validation {label}', marker='s')

    if show_error_bands:
        trainining_scores_std = np.std(train_scores, axis=1)
        validation_scores_std = np.std(val_scores, axis=1)

        ax.fill_between(train_sizes, (training_scores_mean - trainining_scores_std), (training_scores_mean + trainining_scores_std), alpha=0.2)
        ax.fill_between(train_sizes, (validation_scores_mean - validation_scores_std), (validation_scores_mean + validation_scores_std), alpha=0.2)
    
    # if model_name is None:
    #     ax.set_title(f'Learning Curve\n{str(model)}')
    # else:
    #     ax.set_title(f'Learning Curve\n{model_name}')
    ax.set_title(f'Learning Curve\n{model_name}')
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel(f'{label}')
    ax.legend()
    ax.grid(alpha=0.6)

    return training_scores_mean, validation_scores_mean