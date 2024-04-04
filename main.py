# -*- coding: utf-8 -*-
import pandas as pd
import glob
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import manipulating_data as md
from scipy.stats import iqr
import numpy as np
from dateutil import parser
from sklearn.linear_model import LinearRegression

from utils.DateVariable import DateVariable

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

"""# Zadanie 1.5 - zadanie na 5 dla ambitnych

### Rozszerzenie klasy CategoricalVariable
Cel:
Zadaniem jest stworzenie klasy NewCategoricalVariable, która dziedziczy po klasie CategoricalVariable i dodaje nową metodę one_hot_encode, która będzie realizować kodowanie one-hot dla zmiennej kategorycznej.

Ogólnie chodzi o to, że możemy mieć zmienną taką jak kolor, a modele machine learning / statystyczne rozumieją tylko liczby. Więc jak mamy zmienną kolor, o wartościach zielony i niebieski, to tworzymy 2 nowe zmienne, z których jedna odpowiada na pytanie "Czy kolor jest niebieski?" i przyjmuje wartość 1 jeśli tak i 0 jeśli nie, i drugą zmienną, która odpowiada na pytanie "Czy kolor jest zielony?" i również przyjmuje wartość 1, jeśli tak, i 0, jeśli nie.
"""

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

"""- a) Zdefiniuj nową klasę NewCategoricalVariable, która dziedziczy po CategoricalVariable. To oznacza, że przejmie ona wszystkie atrybuty i metody klasy nadrzędnej.

- b) Napisz metodę one_hot_encode wewnątrz klasy NewCategoricalVariable. Ta metoda powinna przyjmować jedną kolumnę typu pd.Series tak jak metoda ordinal_encode w powyższej klasie CategoricalVariable. Nie musisz podawać żadnych innych argumentów. Metoda powinna zwracać DataFrame po kodowaniu one-hot.

Twoja nowa metoda w klasie NewCategoricalVariable ma wyglądać jakoś tak :
"""

@staticmethod
def one_hot_encode(column : pd.Series) -> pd.DataFrame:

    # wewnątrz NewCategoricalVariable zdefiniuj tą metodę, a w środku
    # napisz kod który zamieni każdą unikalną wartość
    # zmiennej kategorialnej (takiej jak kolor czy nazwa miasta) na nową kolumnę
    # o wartościach 0 i 1 a następnie połączy te kolumny w jedną pd.DataFrame
    # o nazwie frame

    """
    Encodes the given column using one-hot encoding.

    Parameters
    ----------
    column : pd.Series
        The column to be encoded.

    Returns
    -------
    pd.DataFrame
        The one-hot encoded column, which becomes a DataFrame (table of columns).

    Example
    -------
    input :
        city
        sj
        sj
        iq
        iq

    output :
        city_iq  city_sj
            0.0      1.0
            0.0      1.0
            1.0      0.0
            1.0      0.0
    """

    return #frame


"""Na koniec zobacz czy metoda encode_as_number poprawnie koduje daty jako liczby.

# Zadanie 3

Klasa NumericVariable jest przeznaczona do reprezentowania zmiennej numerycznej z jakiegoś zbioru danych. Będziecie tworzyć konstruktor oraz metodę detect_outlier_iqr. Oto jak to zrobić:
1. Konstruktor __init__(self, column: pd.Series):
Konstruktor ma za zadanie inicjalizować instancję klasy, przyjmując jedną zmienną: kolumnę danych w formacie pd.Series. Oto jakie kroki należy wykonać:

- a) W definicji konstruktora określ, że przyjmuje on jeden parametr column, który jest obiektem typu pd.Series.

- b) W ciele konstruktora przypisz przekazany argument column do atrybutu instancji o tej samej nazwie column.

2. Metoda detect_outlier_iqr(self) -> list:
Ta metoda ma za zadanie wykrywać obserwacje odstające (outliers) w przekazanej kolumnie numerycznej, korzystając z metody zakresu międzykwartylowego (IQR). Realizacja tej metody powinna obejmować następujące kroki:

- a) Oblicz pierwszy (Q1) i trzeci (Q3) kwartyl danych za pomocą funkcji np.percentile z biblioteki numpy.

- b) Oblicz zakres międzykwartylowy (IQR) jako różnicę między Q3 a Q1.

- c) Określ wartości graniczne dla wykrywania odstających obserwacji: 1.5 * IQR powyżej Q3 i 1.5 * IQR poniżej Q1.

- d) Użyj tych granic do identyfikacji indeksów obserwacji odstających, które są mniejsze niż dolna granica lub większe niż górna granica.

- e) Zwróć listę indeksów tych obserwacji odstających.

Przykład szablonu metody:
"""

"""*** Początek wskazówki do podpunktu d) ***
Pamiętaj, że Twój atrybut instancji klasy NumericVariable jest typu pd.Series.
"""

# przykładowy obiekt typu pd.Series
seria = pd.Series([1, 2 ,3])
seria

# subsetting obiektu pd.Series
# wyświetlamy elementy mniejsze od 3
seria[seria < 3]

# w zadaniu będziemy bazować na indeksach
# poniżej przykład jak znależć indeksy danych mniejszych od 3 i dostać je jako listę
seria[seria < 3].index.tolist()


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



"""Kiedy skończysz uruchom poniższy kod :"""

# NumericVariable(pd.Series( [ 1, 2, 3, 10 ** 7 ] ) ).detect_outlier_iqr()

"""Powinien zwrócić 3, czyli indeks liczby nie pasującej wielkością do reszty - 10 ^ 7."""

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

path = glob.glob('**/*dengue_features_train.csv', recursive=True) [0]

deng_train = pd.read_csv(path, delimiter=';')

"""## Przykład danych wejściowych"""

deng_train.head()

deng_train[['city', 'week_start_date']].head()

# df = CleanDataset(data = deng_train, date_col_name = 'week_start_date').get_data(impute_missing=True, remove_outliers=True, encoding_method='ordinal')

# """## Przykład danych wyjściowych"""

# df.head()

# df[['city', 'week_start_date']].head()