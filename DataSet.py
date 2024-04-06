import pandas as pd

class DataSet:

    """
    A class used to represents tabular data.
    which can be used in down-stream predictive modelling.

    Attributes
    ----------
    path : str
        The path to the data file.
    data : pandas DataFrame.
        The data loaded from the given path.

    """

    def __init__(self, path=None, data=None):

        """
        Loads the data from the given path
        and stores path and data as instance attributes.
        The data attribute is of type pandas DataFrame.

        Parameters
        ----------
        path  : str, optional. Defaults to None.
            The path to the data file.
        data : pandas DataFrame, optional. Defaults to None.
            The data already in DataFrame form.
        """

        if path is None and data is None:
            raise ValueError("Musimy skądś wziąć dane!")
        elif path is not None and data is not None:
            print(f"Podano ścieżkę: {path}")
            print(f"Podano dane: {data}")
            raise ValueError("Należy podać tylko jeden argument: ścieżkę lub dane.")
        elif path is not None:
            self.path = path
            self.data = pd.read_csv(path, delimiter=';')
        else:
            self.path = None
            self.data = data