import pandas as pd
from CategoricalVariable import CategoricalVariable

class NewCategoricalVariable(CategoricalVariable):

    @staticmethod
    def one_hot_encode(column: pd.Series) -> pd.DataFrame:
        dummies = {}
        for value in column.unique():
            dummies[value] = [1 if x == value else 0 for x in column]

        frame = pd.DataFrame(dummies)

        frame.columns = [f"city_{col}" for col in frame.columns]

        return frame

