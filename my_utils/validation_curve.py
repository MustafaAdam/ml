from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
import numpy as np

def plot_validation_curve(model, X, y, param_name, param_range, cv=5, ax=None, scoring='accuracy'):
    train_scores, validation_scores = validation_curve(
        model, X, y, param_name=param_name, param_range=param_range, cv=cv, scoring=scoring)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(validation_scores, axis=1)
    val_std = np.std(validation_scores, axis=1)

    if ax is None:
        ax = plt.gca()

    
    ax.plot(param_range, train_mean, label='Training score')
    ax.fill_between(param_range, train_mean - train_std, train_mean + train_std, 
                    alpha=0.1, color='blue')

    # Plot validation scores
    # sns.lineplot(x=param_range, y=val_mean, label='Validation Score', ax=ax)
    ax.plot(param_range, val_mean, label='Validation Score', color='blue')
    ax.fill_between(param_range, val_mean - val_std, val_mean + val_std, 
                    alpha=0.1, color='red')

    # Formatting
    ax.set_xscale('log')  # Important for C values!
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:g}'))
    ax.set_xlabel(f'{param_name}')
    ax.set_ylabel(f'{scoring.replace('_', ' ').title()} ')
    ax.set_title('Validation Curve')
    ax.legend()
    ax.grid()