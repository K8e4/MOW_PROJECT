import os
import numpy as np
import winequality.analysis as analysis
import winequality.reader as r
import winequality.transform as t
from termcolor import colored

df = r.df()

## Analyze the data
if os.environ.get('RUN_ANALYSIS') == 1:
    print(colored('\n-------------------------------------------------------------------\n', 'red'))
    print(colored('Analyze the raw data:\n', 'red'))

    analysis.run(df)

    print(colored('\n-------------------------------------------------------------------\n', 'red'))

    analysis.run(t.transform(df))

## Join the class 3/4 to one class and 8/9 to one class
df = t.merge_classes_by_vals(t.merge_classes_by_vals(df, 'quality', 6, 7, 6), 'quality', 3, 4, 3)

print(df.quality.unique())

## After joining we have only 5 classes, therefore we change their values to labels A, B, C, D, E
df = t.rename_vals_from_col()
