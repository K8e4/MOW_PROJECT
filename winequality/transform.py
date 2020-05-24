import numpy as np
from winequality.debug import dprint
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

@dprint
def transform(df):
    ## Standarize the df
    df_standard = (df - df.mean()) / df.std()

    ## Normalize to 0 - 1
    df_norm = (df_standard - df_standard.min()) / (df_standard.max() - df_standard.min())

    return df_norm

@dprint
def merge_classes_by_vals(df, col, v1, v2, nval):
    df.loc[(df[col] == v1) | (df[col] == v2), col] = nval

    return df

@dprint
def rename_vals_from_col(df, col, match, replacement):
    if len(match) != len(replacement):
        raise Exception('Both match & replacement vectors should have size equal')

    for m, r in zip(match, replacement):
        df.loc[df[col] == m, col] = r

    return df

@dprint
def outliers_to_std(df, cols, m = 2):

    for col in cols:
        z_scores = zscore(df[col])
        abs_z_scores = np.abs(z_scores)
        df.loc[(abs_z_scores < m), col] = np.std(df[col])

    return df

def cols_except_gd(df, gd_col):
    return list(filter(lambda x: x != gd_col, df.columns.values))

def features_gd_pair(df, gd_col):
    return df.loc[:, cols_except_gd(df, gd_col)], df.loc[:, gd_col]

def train_test_pair(df, gd_col, test_size=0.1):
    feature_cols = cols_except_gd(df, gd_col)

    data = df.loc[:, feature_cols]
    gd_data = df[gd_col]

    d_train, d_test, _, __ = train_test_split(data, gd_data, test_size=test_size, random_state=11, stratify=gd_data)

    return d_train, d_test

def select_with_chi(df, gd_col, remove_above_significance_lvl=0.80):
    x, y = features_gd_pair(df, gd_col)

    # significance level
    # the lower the more significant the feature is
    _, p_vals = chi2(x, y)

    unimportant_indices = [i for i, v in enumerate(p_vals) if v > remove_above_significance_lvl]
    feature_cols = cols_except_gd(df, gd_col)
    unimportant_features = np.array(feature_cols)[unimportant_indices]
    significent_features = [x for x in feature_cols if x not in unimportant_features]

    print('Those are unimportant features which gonna be removed: ' + str(unimportant_features))
    print('Those are significient features: ' + str(significent_features))

    gd_col_with_significient_features = significent_features + [gd_col]

    return df.loc[:, gd_col_with_significient_features]
