# -*- coding: utf-8 -*-
import pandas as pd
import glob
from CleanDataset import CleanDataset

# x = NumericVariable(pd.Series( [ 1, 2, 3, 10 ** 7 ] ) ).detect_outlier_iqr()
# print(x)
# Should return [3]

path = glob.glob('**/*dengue_features_train.csv', recursive=True) [0]

deng_train = pd.read_csv(path, delimiter=';')

"""## Przykład danych wejściowych"""
deng_train.head()
deng_train[['city', 'week_start_date']].head()

df = CleanDataset(data = deng_train, date_col_name = 'week_start_date').get_data(impute_missing=True, remove_outliers=True, encoding_method='ordinal')

"""## Przykład danych wyjściowych"""

outHead = df.head()
outData = df[['city', 'week_start_date']].head()
print(outData)