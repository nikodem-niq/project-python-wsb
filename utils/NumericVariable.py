import numpy as np
import pandas as pd


class NumericVariable() :
    """
    A class used to represent a numeric variable from some dataset.

    Attributes
    ----------
    column : pandas Series.
        A column from a pandas DataFrame or a standalone column.

    Methods
    -------
    detect_outlier_iqr() -> list
        Detects outliers using the interquartile range method.
    """

    # popraw proszę konstruktor zgodnie z pkt 1 polecenia
    def __init__(self, column: pd.Series):

        """
        Initializes the instance of the class.

        Parameters
        ----------
        column : pd.Series
            The column of data in pd.Series format.
        """

        self.column = column

    def detect_outlier_iqr(self) -> list:

        """
        Detects outliers using the interquartile range method.
        Returns a list of indices of the outliers in the given column.
        The method uses the 1.5 * IQR rule to detect outliers.


        Returns
        -------
        list
            A list of indices of the outliers in the given column.
        """

        # a) Obliczanie kwartyli
        q1 = np.percentile(self.column, 25)
        q3 = np.percentile(self.column, 75)

        # b) Obliczanie IQR
        iqr = q3 - q1

        # c) Określenie granic
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # d) Identyfikacja i zwracanie indeksów odstających obserwacji
        outliers_indices = []
        for i in range(len(self.column)):
            if self.column[i] < lower_bound or self.column[i] > upper_bound:
                outliers_indices.append(i)

        return outliers_indices