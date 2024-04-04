import pandas as pd

class DataImputation():
    """
    A class used to represent data imputation methods - methods used to fill in missing values in a dataset.

    Methods
    -------
    mode_imputation(column : pd.Series) -> pd.Series
        Imputes missing values using the mode of the given column.
        Most suitable for categorical data.

    mean_imputation(column : pd.Series) -> pd.Series
        Imputes missing values using the mean of the given column.
        Most suitable for numeric data without outliers.

    median_imputation(column : pd.Series) -> pd.Series
        Imputes missing values using the median of the given column.
        Most suitable for numeric data with outliers.
    """

    def mode_imputation(self, column: pd.Series):
        mode = column.mode()[0]
        return column.fillna(mode)

    def mean_imputation(self, column: pd.Series):
        """Imputes missing values using the mean of the given column.
           Most suitable for numeric data without outliers.
        """
        mean = column.mean()
        return column.fillna(mean)

    def median_imputation(self, column: pd.Series):
        """Imputes missing values using the median of the given column.
           Most suitable for numeric data with outliers.
        """
        median = column.median()
        return column.fillna(median)