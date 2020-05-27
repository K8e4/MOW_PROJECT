import pandas as pd
from deap import base
from deap import creator
from deap import tools
from sklearn.preprocessing import LabelEncoder
import random
from sklearn import metrics
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier


def KNNParametersFeatures(numberFeatures, icls):
    genome = list()

    # n_neighbors
    n_neighbors = random.randint(1, 50)
    genome.append(n_neighbors)

    # weights
    weights = ["uniform", "distance"]
    genome.append(weights[random.randint(0, 1)])

    # algorithms
    algorithms = ["auto", "ball_tree", "kd_tree", "brute"]
    genome.append(algorithms[random.randint(0, 3)])

    for i in range(0, numberFeatures):
        genome.append(random.randint(0, 1))
    return icls(genome)


def KNNParametersFeatureFitness(y, df, numberOfAtributtes, individual):
    split = 5
    cv = StratifiedKFold(n_splits=split)

    listColumnsToDrop = []
    for i in range(numberOfAtributtes, len(individual)):
        if individual[i] == 0:
            listColumnsToDrop.append(i - numberOfAtributtes)
    dfSelectedFeatures = df.drop(df.columns[listColumnsToDrop], axis=1, inplace=False)

    mms = MinMaxScaler()
    df_norm = mms.fit_transform(dfSelectedFeatures)
    estimator = KNeighborsClassifier(n_neighbors=individual[0], weights=individual[1], algorithm=individual[2])
    resultSum = 0
    for train, test in cv.split(df_norm, y):
        estimator.fit(df_norm[train], y[train])
        predicted = estimator.predict(df_norm[test])
        expected = y[test]
        tn, fp, fn, tp = metrics.confusion_matrix(expected, predicted).ravel()
        result = (tp + tn) / (tp + fp + tn + fn)
        resultSum = resultSum + result
    return resultSum / split,


def mutationKNN(individual):
    numberParamer = random.randint(0, len(individual) - 1)
    if numberParamer == 0:
        n_neighbors = random.randint(1, 50)
        individual[0] = n_neighbors

    elif numberParamer == 1:
        weights = ["uniform", "distance"]
        individual[1] = weights[random.randint(0, 1)]

    elif numberParamer == 2:
        algorithms = ["auto", "ball_tree", "kd_tree", "brute"]
        individual[2] = algorithms[random.randint(0, 3)]

    else:
        if individual[numberParamer] == 0:
            individual[numberParamer] = 1
        else:
            individual[numberParamer] = 0