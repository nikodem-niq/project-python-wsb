# -*- coding: utf-8 -*-
import pandas as pd
import glob

from CleanDataset import CleanDataset
from DataSet import DataSet
from utils.DateVariable import DateVariable
from utils.NumericVariable import NumericVariable
from CategoricalData import CategoricalData
from NumericData import NumericData


"""Kiedy skończysz uruchom poniższy kod :"""

# x = NumericVariable(pd.Series( [ 1, 2, 3, 10 ** 7 ] ) ).detect_outlier_iqr()
# print(x)

"""Powinien zwrócić 3, czyli indeks liczby nie pasującej wielkością do reszty - 10 ^ 7."""

path = glob.glob('**/*dengue_features_train.csv', recursive=True) [0]

deng_train = pd.read_csv(path, delimiter=';')

"""## Przykład danych wejściowych"""
inData = deng_train.head()
inHead = deng_train[['city', 'week_start_date']].head()
print(inData, inHead)

df = CleanDataset(data = deng_train, date_col_name = 'week_start_date').get_data(impute_missing=True, remove_outliers=True, encoding_method='ordinal')

"""## Przykład danych wyjściowych"""

outHead = df.head()
outData = df[['city', 'week_start_date']].head()
print(outHead, outData)