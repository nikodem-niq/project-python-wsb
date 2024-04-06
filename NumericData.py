from DataSet import DataSet
from utils.NumericVariable import NumericVariable


class NumericData(DataSet) :

    """
    A class used to represent numeric data from some dataset.
    Inherits from the DataSet class (which can consist of both numeric and categorical data).

    Attributes
    ----------
    path : str, optional. Defaults to None.
        The path to the data file.
    data : pandas DataFrame, optional. Defaults to None.
        The data already in DataFrame form.

    Methods
    -------
    detect_outliers(method='iqr', by_column=False) -> list or dict
        Detects outliers using the given method.
        Defaults to the interquartile range method.
    """

    def __init__(self, path = None, data=None) :

        super().__init__(path, data)
        self.num_data = self.data.select_dtypes(include='number')

    def detect_outliers(self, method = 'iqr', by_column = False) :

        """
        Detects outliers using the given method.
        By default uses interquartile range method.
        Under the hood, it applies NumericVariable.detect_outlier_iqr() to each numeric column,
        and returns combined data as list / dictionary.

        Parameters
        ----------
        method : str, optional. Defaults to 'iqr'.
            The method to use for outlier detection.

        by_column : bool, optional. Defaults to False.
            If True, returns a dictionary of outliers for each column.
            If False, returns a list of indices of the outliers in the dataset (which is a set of all columns).

        Returns
        -------
        list or dict
            A list of indices of the outliers in the dataset (if by_column is False).
            A dictionary of outliers for each column (if by_column is True).
        """

        outliers = {}

        for c in self.num_data.columns :

            numeric_col = NumericVariable(self.data[c])

            if method == 'iqr' :
                indices_outliers_iterab = numeric_col.detect_outlier_iqr()

                if indices_outliers_iterab != [] :
                    outliers[c] = indices_outliers_iterab


            elif method == 'z_score' :
                pass
            else :
                raise ValueError(f"Outlier detection method {method} not recognized.")

        if by_column :
            return outliers

        outlier_indices = []

        for v in outliers.values() :
            outlier_indices += v

        return outlier_indices