import os
import numpy as np
import winequality.analysis as analysis
import winequality.reader as r
import winequality.transform as t
from termcolor import colored

df = r.df()
gd_col = 'quality'

#--------------------------------------------------------------------------------------------------
# START: Analysis
#--------------------------------------------------------------------------------------------------
if os.environ.get('RUN_ANALYSIS') == '1':
    print(colored('\n-------------------------------------------------------------------\n', 'red'))
    print(colored('Analyze the raw data:\n', 'red'))

    analysis.run(df)

    print(colored('\n-------------------------------------------------------------------\n', 'red'))

    analysis.run(t.transform(df))
#--------------------------------------------------------------------------------------------------
# END: Analysis
#--------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------
# START: Classification Algorithm
#--------------------------------------------------------------------------------------------------


## Join the class 3/4 to one class and 8/9 to one class
df = t.merge_classes_by_vals(t.merge_classes_by_vals(df, gd_col, 6, 7, 6), gd_col, 3, 4, 3)
print('\nNUMERIC VALUES OF COLUMN `QUALITY`: ' + str(df.quality.unique()))



## After joining we have only 5 classes, therefore we change their values to labels A, B, C, D, E
df = t.rename_vals_from_col(df, gd_col, [3, 5, 6, 8, 9], ['E', 'D', 'C', 'B', 'A'])
print('\nLABELS OF COLUMN `QUALITY`: ' + str(df.quality.unique()))



## New dataset in which the outliers in particular column are replaced with the STD of all the values in column
df1 = t.outliers_to_std(df.copy(), df.columns[:-1])


## Select the features which are useful
print('\nBefore selection: ' + str(df.shape) + '\n')
df = t.select_with_chi(df, gd_col)

## What we can observe is that the dataset from which the outliners has been removed
## shows different feature which have to be removed
df1 = t.select_with_chi(df1, gd_col)
print('\nAfter selection: ' + str(df.shape))



## split dataset into [90% of df = train, 10% of df = test]
## stratiffy according to ground truth classes
## example [train[1,2,3], test[2,3,4,5]]
df_train, df_test = t.train_test_pair(df, gd_col)
df1_train, df1_test = t.train_test_pair(df, gd_col)


#--------------------------------------------------------------------------------------------------
# END: Classification Algorithm
#--------------------------------------------------------------------------------------------------
