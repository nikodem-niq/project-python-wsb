import pandas as pd

class DateVariable() :

    """
    A class used to represent a date variable from some dataset.

    Attributes
    ----------
    column : pandas Series.
        A column from a pandas DataFrame or a standalone column.

    Static Methods
    -------
    parse_string_date(string_date : str) -> pd.Timestamp
        Parses a string date into a pandas Timestamp object.

    Methods
    -------
    encode_as_number() -> pd.Series
        Encodes the date as a number, where each day is represented by a unique number.
    """

    def __init__(self, column : pd.Series) :
        self.column = column

    @staticmethod
    def parse_string_date(string_date: str) -> pd.Timestamp:

        """
        Parses a string date into a pandas Timestamp object.

        Parameters
        ----------
        string_date : str
            The date in string format.

        Returns
        -------
        pd.Timestamp
            The parsed date as a Timestamp object.
        """

        timestamp = pd.to_datetime(string_date, dayfirst=True)

        formatted_date = timestamp.strftime('%Y-%m-%d')

        return pd.Timestamp(formatted_date)
    
    def encode_as_number(self) -> pd.Series:
        if self.column.dtype != 'datetime64[ns]':
            self.column = self.column.apply(DateVariable.parse_string_date)

        year = self.column.dt.year
        month = self.column.dt.month
        day = self.column.dt.day

        encoded_column = year * 365 + (month - 1) * 30 + day

        return encoded_column