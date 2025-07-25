import pandas as pd
import seaborn as sns
import numpy as np

def get_correlation_matrix(df: pd.DataFrame):
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', mask=np.triu(np.ones_like(df.corr())))

