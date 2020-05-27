import pandas as pd
from deap import base
from deap import creator
from deap import tools
from sklearn.preprocessing import LabelEncoder
import random
from sklearn import metrics
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

def TreeParametersFeatures(numberFeatures, icls):
    genome = list()

    max_depth = random.randint(1, 50)
    genome.append(max_depth)

    criterion = ["gini", "entropy"]
    genome.append(criterion[random.randint(0, 1)])

    splitter = ["best", "random"]
    genome.append(splitter[random.randint(0, 1)])

    for i in range(0, numberFeatures):
        genome.append(random.randint(0, 1))
    return icls(genome)


def TreeParametersFeatureFitness(y, df, numberOfAtributtes, individual):
    split = 5
    cv = StratifiedKFold(n_splits=split)

    listColumnsToDrop = []
    for i in range(numberOfAtributtes, len(individual)):
        if individual[i] == 0:
            listColumnsToDrop.append(i - numberOfAtributtes)
    dfSelectedFeatures = df.drop(df.columns[listColumnsToDrop], axis=1, inplace=False)

    mms = MinMaxScaler()
    df_norm = mms.fit_transform(dfSelectedFeatures)
    estimator = DecisionTreeClassifier(max_depth=individual[0], criterion=individual[1], splitter=individual[2], random_state=0)
    resultSum = 0
    for train, test in cv.split(df_norm, y):
        estimator.fit(df_norm[train], y[train])
        predicted = estimator.predict(df_norm[test])
        expected = y[test]
        tn, fp, fn, tp = metrics.confusion_matrix(expected, predicted).ravel()
        result = (tp + tn) / (tp + fp + tn + fn)
        resultSum = resultSum + result
    return resultSum / split,


def mutationTree(individual):
    numberParamer = random.randint(0, len(individual) - 1)
    if numberParamer == 0:
        max_depth = random.randint(1, 50)
        individual[0] = max_depth

    elif numberParamer == 1:
        criterion = ["gini", "entropy"]
        individual[1] = criterion[random.randint(0, 1)]

    elif numberParamer == 2:
        splitter = ["best", "random"]
        individual[2] = splitter[random.randint(0, 1)]

    else:
        if individual[numberParamer] == 0:
            individual[numberParamer] = 1
        else:
            individual[numberParamer] = 0
