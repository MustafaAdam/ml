import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


class EvaluationPlots: 
    @staticmethod
    def plot_residuals_histogram(residuals, bins=20, ax=None):
        '''
            1. A good residuals model should be centered around 0
                If residuals aren't centered around 0, it means your model has systematic bias
                Meaning that the model is consistently over-predicting or under-predicting
                This violates the assumption that E(residuals) = 0
            2. Linear regression assumes errors are normally distributed: ε ~ N(0, σ²)
                If residuals ARE normal:
                    Your model is missing randomness only - the patterns you haven't captured are just noise
                    The unexplained variation behaves like pure random error
                    Your model has captured the systematic relationships in the data
                If residuals are NOT normal:
                    Your model is missing systematic patterns - there are still discoverable relationships you haven't captured
                    The errors aren't just random noise - they have structure
            3. Possible residuals that are not good are:
                a. Skewed residuals: Model systematically worse for high/low values
                b. Heavy tails: Model occasionally makes huge mistakes (outliers)
                c. Bimodal: Model works differently for different groups in your data
        '''
        if ax is None:
            ax = plt.gca()

        sns.histplot(residuals, kde=True, bins=bins, alpha=0.7, ax=ax)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1)

        ax.set_xlabel('Residuals')
        ax.set_title('Residuals histogram')
        ax.grid()
    
    @staticmethod
    def plot_residuals_vs_predictions(predictions, residuals, ax=None):
        '''
            A good residuals vs predictions plot should be a random scatter, with not discernible shape around y = 0 line
            The spread of residuals should be constant across all predicted values.
            If there's an expanding or contracting residuals shape as predictions values increase, this indicates non-constant variance.
            This violates the Homoscedasticity(constant variance) assumption of linear regression.
            Systematic patterns in the scatter plot (curve, smiles, frowns), indicates that there exists a nonlinear relationship between
                the features and target. This violates the Linearity assumption(features should be have a linear relationship with the target)
                This indicates that linear regression is not a good model for the dataset.
            If the average residual is not close to zero, this indicates the existence of systematic bias in the predictions.
                If most residuals are under zero, the model tends to under predict. If most residuals are above zero, the model tends to over predict.

            Used to detect:
                Heteroscedasticity: residual spread changes with predicted value.
                Nonlinearity: curve or pattern suggests missing nonlinear component.
                Bias: residuals systematically above or below 0.
        '''
        if ax is None:
            ax = plt.gca()

        sns.scatterplot(x=predictions, y=residuals, alpha=0.7, ax=ax)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
        
        ax.set_xlabel('Predictions')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals vs Predictions')
        ax.grid()

    @staticmethod
    def plot_predictions_vs_true(true_values, predictions, ax=None):
        '''
            A good true values vs predictions plot should show a tight scatter around the diagonal line.
            This indicates that our predictions are very close to the true value (indicated by the diagonal line)
            The tighter the clustering of the scatters around the diagonal, the better our model is.
            Also, the higher R2 score, the tighter the clustering of the scatters around the diagonal is.

            Used to detect:
                Bias: whether the model is systematically over or under predicting and at which range of values (low true values or high ones)
        '''
        if ax is None:
            ax = plt.gca()

        sns.scatterplot(x=true_values, y=predictions, alpha=0.7, ax=ax)
        
        # Reference line y = x
        min_val = min(true_values.min(), predictions.min())
        max_val = max(true_values.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

        ax.set_xlabel('True values')
        ax.set_ylabel('Predictions')
        ax.set_title('Predictions vs true plot')
        ax.grid()
    
    @staticmethod
    def plot_qq(residuals, ax=None):
        """
            Generate a Q-Q plot for the given residuals.

            Parameters:
            residuals (array-like): The residuals to plot.

                        | What You See                  | What It Means                                                              |
            | ----------------------------- | -------------------------------------------------------------------------- |
            | Points lie on a straight line | Your data is well-modeled by the chosen distribution (often normal)        |
            | S-curve shape                 | Your data has **heavier or lighter tails** than the reference distribution |
            | Upward curve at the ends      | Your data has **heavy tails** (outliers in both directions)                |
            | Downward curve at the ends    | Your data has **light tails** (less spread than normal)                    |
            | Asymmetry in curve            | Indicates **skewness** (left or right skew)                                |

        """
        if ax is None:
            ax = plt.gca()

        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title("Q-Q Plot of Residuals")
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")
        ax.grid()


        ### This is not important to learn.
        ### It only serves to match the colors of other plots. Not important at all
        # Change scatter points color
        for line in ax.get_lines():
            # line[0] is usually points, line[1] is the 45-degree line
            line.set_color('#1f77b4')
            line.set_markersize(5)
            line.set_alpha(1)
        # Change line color (if needed)
        # Usually the last line is the fit line:
        ax.get_lines()[-1].set_color('red')
        ax.get_lines()[-1].set_linestyle('--')
        ax.get_lines()[-1].set_linewidth(2)