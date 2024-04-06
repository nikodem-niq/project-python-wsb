import pandas as pd
from PreparingDataset import PreparingDataset


class CleanDataset(PreparingDataset) :

    """
    A class used to represent a clean dataset which is ready for predictive modelling.

    Attributes
    ----------
    path : str, optional. Defaults to None.
        The path to the data file.
    data : pandas DataFrame, optional. Defaults to None.
        The data already in DataFrame form.
    date_col_name : str, optional. Defaults to None.
        The name of the column that contains date data.

    Methods
    -------
    get_data(encoding_method='one_hot', outlier_method='iqr', remove_outliers=True, impute_missing=False) -> pd.DataFrame
        Returns the clean dataset ready for predictive modelling.

    """

    def __init__(self, path = None, data = None, date_col_name = None) :
        super().__init__(path = path, data = data, date_col_name = date_col_name)
        self.date_col_name = date_col_name

    def get_data(self, encoding_method = 'one_hot', outlier_method = 'iqr', remove_outliers = True, impute_missing = False) :

        categorical = self.prepare_categoricl_data(method = encoding_method, impute_missing = impute_missing)
        numeric = self.prepare_numeric_data(method=outlier_method, remove_outliers = remove_outliers, impute_missing=impute_missing)

        data_parts = [categorical, numeric]

        if self.date_col_name is not None :
            date_calendar = self.prepare_date_data()
            data_parts.append(date_calendar)

        return pd.concat(data_parts, axis=1)