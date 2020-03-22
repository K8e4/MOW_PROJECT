import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

wine = pd.read_csv("C:/Users/Kasia/Desktop/data_science/MOW/winequality-white.csv", ';')
print(wine.isnull().sum())
print(wine.describe())
#test_df = pd.DataFrame(wine)
#print(test_df)