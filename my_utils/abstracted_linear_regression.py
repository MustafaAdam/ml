import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split



class CustomLinearRegression:
    def __init__(self, features : pd.DataFrame = None, target: np.typing.NDArray = None):
        """
            A convenience wrapper class for performing Linear Regression with train-test split,
            evaluation metrics, coefficient extraction, and residual analysis plots.

            Parameters:
            -----------
            features : pd.DataFrame
                Input feature dataset used for training and testing the model.
            target : pd.Series or np.ndarray
                Target variable array or series corresponding to the features.

            Attributes:
            -----------
            X_train, X_test : pd.DataFrame
                Training and testing splits of the features.
            y_train, y_test : np.ndarray or pd.Series
                Training and testing splits of the target.
            lr : sklearn.linear_model.LinearRegression
                The fitted Linear Regression model.
            predictions : np.ndarray
                Predictions made on the test set.
            r_squared : float
                R² score on the test set.
            mse : float
                Mean Squared Error on the test set.
            rmse : float
                Root Mean Squared Error on the test set.

            Methods:
            --------
            fit_and_predict():
                Fits the linear regression model on the training data and generates predictions on the test set.
            weights_and_intercept() -> pd.DataFrame:
                Returns a DataFrame containing the model coefficients sorted by magnitude and the intercept.
            metrics() -> dict[str, str]:
                Returns R², MSE, and RMSE scores as formatted strings for the test set.
            get_predictions() -> np.ndarray:
                Returns the model predictions on the test set.
            check_overfit() -> dict[str, str]:
                Compares training and testing metrics to assess overfitting.
            get_residuals() -> np.ndarray:
                Returns residuals (actual - predicted) for the test set.
            plot_residuals(x_size=12, y_size=5):
                Plots a histogram of residuals with KDE and zero reference line.
            plot_residuals_vs_predictions(x_size=12, y_size=5):
                Plots residuals versus predicted values scatter plot with zero horizontal line.

            Usage:
            ------
            model = CustomLinearRegression(features_df, target_array)
            model.fit_and_predict()
            print(model.metrics())
            model.plot_residuals()
            model.plot_residuals_vs_predictions()
        """


        # make sure it's a Pandas Series or Numpy NDArray
        if not (isinstance(target, pd.Series) or isinstance(target, np.ndarray)):
            raise Exception('target must be an NDArray, not a DataFrame')
        
        self.X_train , self.X_test, self.y_train, self.y_test = train_test_split(
            features,
            target,
            test_size=0.2,
            random_state=100
        )
        self.lr : LinearRegression | None = None
        self.predictions : np.typing.NDArray = None
        self.r_squared : float = None
        self.mse : float = None
        self.rmse : float= None
    
    def fit_and_predict(self) -> None:
        self.lr = LinearRegression().fit(self.X_train, self.y_train)
        self.predictions = self.lr.predict(self.X_test)
        self.r_squared = r2_score(self.y_test, self.predictions)
        self.mse = mean_squared_error(self.y_test, self.predictions)
        self.rmse = root_mean_squared_error(self.y_test, self.predictions)

    def weights_and_intercept(self) -> pd.DataFrame:
        coefficients = self.lr.coef_ if self.lr.coef_.ndim == 1 else self.lr.coef_.flatten()
        df = pd.DataFrame(data={'Coefficients and Intercept': coefficients}, index=self.X_train.columns)
        df.sort_values(
            by='Coefficients and Intercept', ascending=False, inplace=True
        )
        df.loc[''] = ''
        df.loc['Intercept'] = self.lr.intercept_
        return df

    def metrics(self) -> dict[str, str]:
       return {
            'R2:': f'{self.r_squared:.2f}',
            'Mean Squared Error:': f'{self.mse:.2f}',
            'Root Mean Squared Error:': f'{self.rmse:.2f}',
        }
    
    def get_predictions(self) -> np.typing.NDArray:
        return self.predictions
    

    
    def check_overfit(self) -> dict[str, str]:
        training_predictions = self.lr.predict(self.X_train)
        training_r2 = r2_score(self.y_train, training_predictions)
        training_mse = mean_squared_error(self.y_train, training_predictions)
        training_rmse = root_mean_squared_error(self.y_train, training_predictions)

        return {
            'Training R2:': f'{training_r2:.2f}',
            'Testing R2:': f'{self.r_squared:.2f}',
            'Training Mean Squared Error:': f'{training_mse:.2f}',
            'Testing Mean Squared Error:': f'{self.mse:.2f}',
            'Training Root Mean Squared Error:': f'{training_rmse:.2f}',
            'Testing Root Mean Squared Error:': f'{self.rmse:.2f}',
        }
    
    def get_residuals(self) -> np.typing.NDArray:
        # return self.y_test.flatten() - self.predictions.flatten()
        return np.ravel(self.y_test) - np.ravel(self.predictions)
    
    def plot_residuals(self, x_size = 12, y_size = 5):
        '''
            A histogram of the residuals checks if the residuals are normally distributed or not.
            In other words, the residuals of the model are randomally distributed
            This is an assumption of linear regression.

            It should show the residuals are centered on or around zero.
            This shows that the model is not consistently predicting over or under the true value.

            Skewed left            -> model is over-predicting
            Skewed right           -> model is under-predicting
            Long tails             -> outliers are present
            Shifted away from zero -> model shows a systematic bias in its predictions

        '''
        plt.figure(figsize=(x_size, y_size))
        plt.title('Residuals Plot')
        plt.xlabel('Residuals')
        plt.ylabel('Count')
        sns.histplot(self.get_residuals(), kde=True)
        plt.axvline(x=0, color='red', linestyle='--')
        plt.show()

    def plot_residuals_vs_predictions(self, x_size=12, y_size=5):
        plt.figure(figsize=(x_size, y_size))
        sns.scatterplot(x=self.predictions, y=self.get_residuals(), alpha=0.6)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Residuals vs Predicted Values")
        plt.show()