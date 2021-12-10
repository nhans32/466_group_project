import numpy as np
from sklearn.base import BaseEstimator
import pandas as pd
from sklearn.metrics import r2_score

class LinearRegressor(BaseEstimator):
    """LinearRegressor model."""

    def __init__(self, iterations: int, learning_rate: float, lambda_coefficient: float) -> None:
        """init for LinearRegressor model.

        Args:
            iterations:         number of steps to take for gradient descent
            learning_rate:      learning rate for gradient descent
            lambda_coefficient: lambda (complexity weight) for gradient descent

        Returns:
            None
        """
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.lambda_coefficient = lambda_coefficient
        self.alpha = 0
        self.betas = np.array([0])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fits the Linear Regression model.

        Args:
            X: explanatory variables
            y: target

        Returns:
            None
        """
        n, p = X.shape

        # initialize alpha and betas
        alpha = 0
        betas = np.zeros(p)

        y_bar = y.mean()
        x_bar = X.mean()

        for _ in range(self.iterations):
            grad_alpha = -2 * (y_bar - alpha - x_bar @ betas)
            grad_beta = -2 * X.T @ (y - alpha - X @ betas) / X.shape[1] + 2 * self.lambda_coefficient * betas
            alpha = alpha - self.learning_rate * grad_alpha
            betas = betas - self.learning_rate * grad_beta

        self.alpha = alpha
        self.betas = betas

    def predict(self, X: pd.DataFrame) -> np.array:
        """Gets the prediction of the Linear Regression model.

        Args:
            X: explanatory variables

        Returns:
            np.array: the prediction of the model (floats)
        """
        return self.alpha + X @ self.betas

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """R^2 measurement for the Linear Regression model.

        Args:
            X: explanatory variables
            y: target

        Returns:
            float: the accuracy of the model (# correct / # total)
        """
        y_pred = self.alpha + X @ self.betas
        # return np.sqrt(np.square(y - y_pred).sum() / X.shape[0])
        return r2_score(y, y_pred)
