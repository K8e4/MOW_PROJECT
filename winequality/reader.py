import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def dataset():
    return pd.read_csv("./resources/winequality-white.csv", ';')
