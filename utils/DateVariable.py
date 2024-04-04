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

        timestamp = pd.to_datetime(string_date)

        formatted_date = timestamp.strftime('%Y-%m-%d')

        return pd.Timestamp(formatted_date)
    
    def encode_as_number(self) -> pd.Series:
        """
        Encodes the date as a number, where each day is represented by a unique number.

        Returns
        -------
        pd.Series
            A pandas Series with the date encoded as a number.
        """

        # Ensure self.column is in datetime format
        self.column = pd.to_datetime(self.column, dayfirst=True) 

        # Convert the date to days since 1970-01-01
        days_since_epoch = (self.column - pd.Timestamp('1970-01-01')).dt.days

        return days_since_epoch