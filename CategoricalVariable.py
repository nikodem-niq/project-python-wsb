import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

class CategoricalVariable() :

    """
    A class used to represent a categorical variable from some dataset.

    Attributes
    ----------
    column : pandas Series.
        A column from a pandas DataFrame or a standalone column.

    Methods
    -------
    ordinal_encode(column: pd.Series, show_mapping=False) -> pd.Series
        Encodes the given column using ordinal encoding.

    """


    def __init__(self, column : pd.Series) :
        self.column = column


    @staticmethod
    def ordinal_encode(column: pd.Series, show_mapping=False) -> pd.Series:
        encoder = OrdinalEncoder()
        encoder_fitted = encoder.fit(pd.DataFrame(column))
        encoded_data = encoder_fitted.transform(pd.DataFrame(column))
        inverse_transformation = encoder_fitted.inverse_transform(encoded_data)

        if show_mapping:
            values_mapping = { e.tolist()[0] : t.tolist() for t, e in\
                                        zip(encoded_data, inverse_transformation) }
            return values_mapping

        return pd.Series(encoded_data.flatten(), index=column.index, name=column.name)



    def encode_data(self, method, show_mapping=False) -> pd.DataFrame:
        if method == 'ordinal':
            encoded_df = CategoricalVariable.ordinal_encode(self.column, show_mapping=show_mapping)
        elif method == 'one_hot':
            encoded_df = CategoricalVariable.one_hot_encode(self.column, show_mapping=show_mapping)
        else:
            raise ValueError(f"Encoding method {method} not recognized.")
        return encoded_df