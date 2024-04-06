import pandas as pd
from CategoricalVariable import CategoricalVariable

class NewCategoricalVariable(CategoricalVariable):

    @staticmethod
    def one_hot_encode(column: pd.Series) -> pd.DataFrame:

        # Stwórz słownik, który będzie mapował unikalne wartości kolumny na listy binarne
        dummies = {}
        for value in column.unique():
            dummies[value] = [1 if x == value else 0 for x in column]

        # Przekształć słownik w DataFrame
        frame = pd.DataFrame(dummies)

        # Dodaj przedrostek "city_" do nazw kolumn
        frame.columns = [f"city_{col}" for col in frame.columns]

        return frame

