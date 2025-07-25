import matplotlib.pyplot as plt
import scipy.stats as stats


def qq_plot(residuals):
    """
    Generate a Q-Q plot for the given residuals.

    Parameters:
    residuals (array-like): The residuals to plot.
    """
    plt.figure()
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("Q-Q Plot of Residuals")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    plt.grid()
    plt.show() 
