import pandas as pd

def replace_outliers_with_median(df : pd.DataFrame, column: str):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    median_value = df[column].median()
    
    # Replace outliers with median
    df.loc[df[column] < lower_bound, column] = median_value
    df.loc[df[column] > upper_bound, column] = median_value
    
    return df


# Example: apply to all numeric columns in California housing data
# for col in remove_outliers_copy.select_dtypes(include='number').columns:
    # remove_outliers_copy = replace_outliers_with_median(remove_outliers_copy, col)