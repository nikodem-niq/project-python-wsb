from CategoricalData import CategoricalData
from NumericData import NumericData
from utils.DateVariable import DateVariable


class PreparingDataset(CategoricalData, NumericData) :

    """
    A class used to describe how to prepare data for predictive modelling.
    Inherits from both CategoricalData and NumericData classes.

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
    prepare_categoricl_data(method='one_hot', impute_missing=False) -> pd.DataFrame
        Prepares the categorical data for predictive modelling, meaning it
        fills in the missing values with the mode (most common value)
        and encodes values such as 'green' as numbers."""

    def __init__(self, path = None, data = None, date_col_name = None) :

        CategoricalData.__init__(self, path = path, data = data)
        NumericData.__init__(self, path = path, data = data)

        if date_col_name is not None :
            self.date_data = self.data[date_col_name]

    def prepare_categoricl_data(self, method = 'one_hot', impute_missing = False) :

        if impute_missing :
            for c in self.cat_data.columns :
                most_common = self.cat_data[c].mode()[0]
                self.cat_data[c] = self.cat_data[c].fillna(most_common)

        return self.encode_data(method = method)

    def prepare_numeric_data(self, method = 'iqr', remove_outliers = False, impute_missing = False) :

        outliers = self.detect_outliers(method)
        outliers_by_column = self.detect_outliers(method, by_column=True)

        if impute_missing :
            for c in self.num_data.columns :
                if c in outliers_by_column.keys() :
                    median = self.num_data[c].median()
                    self.num_data[c] = self.num_data[c].fillna(median)
                else :
                    mean = self.num_data[c].mean()
                    self.num_data[c] = self.num_data[c].fillna(mean)

        if remove_outliers :
            indices_keep = [i for i in self.num_data.index if i not in outliers]
            self.num_data = self.num_data.iloc[indices_keep]

            return self.num_data

        else :

            return self.num_data

    def prepare_date_data(self) :
        return DateVariable(self.date_data).encode_as_number()