import pandas as pd
from CategoricalVariable import CategoricalVariable
from DataSet import DataSet


class CategoricalData(DataSet) :
    """
    A class used to represent a set of categorical variables from some dataset.
    Inherits from the DataSet class.

    Attributes
    ----------
    path : str, optional. Defaults to None.
        The path to the data file.
    data : pandas DataFrame, optional. Defaults to None.
        The data already in DataFrame form.
    max_uniq_vals : int, optional. Defaults to 10.
        The maximum number of unique values a column can have.
        If a column has more unique values than this, it will not be encoded.
        It is useful for avoiding computational overhead when one-hot encoding.

    Methods
    -------
    encode_data(method, show_mapping=False) -> pd.DataFrame
        Encodes the categorical data using the given method.

    """

    def __init__(self, path = None, data = None, max_uniq_vals=10) :
        super().__init__(path, data)
        self.cat_data = self.data.select_dtypes(include='object')
        self.unique_values = self.cat_data.nunique()
        self.cols_to_encode = self.unique_values[self.unique_values <= max_uniq_vals].index.tolist()
        self.cat_data = self.data[self.cols_to_encode]

    def encode_data(self, method, show_mapping=False) -> pd.DataFrame:

        encoded_data = {}

        for column in self.cat_data.columns:
            categorical_col = CategoricalVariable(self.cat_data[column])
            encoded_data[column] = categorical_col.encode_data(method, show_mapping)

        df = pd.DataFrame()

        for k, v in encoded_data.items():
            if method == 'ordinal':
                df[k] = v
            elif method == 'one_hot':
                df = pd.concat([df, v], axis=1)
            else:
                raise ValueError(f"Encoding method {method} not recognized.")

        return df