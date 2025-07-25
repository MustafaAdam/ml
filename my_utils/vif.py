import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

def get_vif(df: pd.DataFrame) -> pd.DataFrame:
    """
        get Variance Inflation Factor
        VIF is a measure used in statistics to detect multicollinearity in regression models
        Multicollinearity occurs when independent variables in a regression model are highly correlated with each other.
        A high VIF indicates that the variance of a regression coefficient is inflated due to this correlation.
        
        VIF = 1: No multicollinearity. 
        VIF between 1 and 5: Moderate multicollinearity. 
        VIF greater than 5 or 10: High multicollinearity, potentially problematic for the model.

        Args:
            df: pd.Dataframe
        
        Returns:
            pd.Dataframe: A dataframe with 2 columns. First column is the names of features. Second column is the VIF value
    """

    vif_data = pd.DataFrame()
    column_names = df.columns
    number_of_columns = df.shape[1]
    vif_data['Features'] = column_names
    vif_data['VIF'] = [variance_inflation_factor(df, i) for i in range(number_of_columns)]
    return vif_data