import pandas as pd
from deap import base
from deap import creator
from deap import tools
from sklearn.preprocessing import LabelEncoder
import random
from sklearn import metrics
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

def SVCParametersFeatures(numberFeatures, icls):
    genome = list()
    # kernel
    listKernel = ["linear", "rbf", "poly", "sigmoid"]
    genome.append(listKernel[random.randint(0, 3)])
    # c
    k = random.uniform(0.1, 100)
    genome.append(k)
    # degree
    genome.append(random.uniform(0.1, 5))
    # gamma
    gamma = random.uniform(0.001, 5)
    genome.append(gamma)
    # coeff
    coeff = random.uniform(0.01, 10)
    genome.append(coeff)
    for i in range(0, numberFeatures):
        genome.append(random.randint(0, 1))
    return icls(genome)


def SVCParametersFeatureFitness(y, df, numberOfAtributtes, individual):
    #cross validation
    split = 10
    cv = StratifiedKFold(n_splits=split)

    #attribute selection
    listColumnsToDrop = []
    for i in range(numberOfAtributtes, len(individual)):
        if individual[i] == 0:
            listColumnsToDrop.append(i - numberOfAtributtes)
    dfSelectedFeatures = df.drop(df.columns[listColumnsToDrop], axis=1, inplace=False)

    mms = MinMaxScaler()
    df_norm = mms.fit_transform(dfSelectedFeatures)
    estimator = SVC(kernel=individual[0], C=individual[1], degree=individual[2], gamma=individual[3],
                    coef0=individual[4], random_state=101)
    resultSum = 0
    for train, test in cv.split(df_norm, y):
        estimator.fit(df_norm[train], y[train])
        predicted = estimator.predict(df_norm[test])
        expected = y[test]
        #print(metrics.confusion_matrix(expected, predicted))

        ####rozwiazanie dzialajace dla binarnych danych w quality####

        tn, fp, fn, tp = metrics.confusion_matrix(expected, predicted).ravel()

        #############################################################

        result = (tp + tn) / (tp + fp + tn + fn)
        resultSum = resultSum + result
    return resultSum / split,


def mutationSVC(individual):
    numberParamer = random.randint(0, len(individual) - 1)
    if numberParamer == 0:
        # kernel
        listKernel = ["linear", "rbf", "poly", "sigmoid"]
        individual[0] = listKernel[random.randint(0, 3)]
    elif numberParamer == 1:
        # C
        k = random.uniform(0.1, 100)
        individual[1] = k
    elif numberParamer == 2:
        # degree
        individual[2] = random.uniform(0.1, 5)
    elif numberParamer == 3:
        # gamma
        gamma = random.uniform(0.01, 1)
        individual[3] = gamma
    elif numberParamer == 4:
        # coeff
        coeff = random.uniform(0.1, 1)
        individual[2] = coeff
    else:  #genetic attribute selection
        if individual[numberParamer] == 0:
            individual[numberParamer] = 1
        else:
            individual[numberParamer] = 0

#config
#adding file with binary quality values
numberIteration = 10
numberElitism = 5
sizePopulation = 100
df = pd.read_csv('winequality-white-binary.csv')
y = df['quality']
df.drop('quality', axis=1, inplace=True)
numberOfAtributtes = len(df.columns)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("select", tools.selRoulette)
toolbox.register("individual", SVCParametersFeatures, numberOfAtributtes, creator.Individual)
toolbox.register("evaluate", SVCParametersFeatureFitness, y, df, numberOfAtributtes)
toolbox.register("mutate", mutationSVC)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
pop = toolbox.population(n=sizePopulation)
fitnesses = list(map(toolbox.evaluate, pop))

for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

g = 0
#main genetic loop
while g < numberIteration:
    g += 1

    offspring = toolbox.select(pop, sizePopulation - numberElitism)
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))

    listElitism = tools.selBest(pop, numberElitism)

    for mutant in offspring:
        # mutate an individual with probability MUTPB
        toolbox.mutate(mutant)
        del mutant.fitness.values

    # Evaluate the individuals with fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    pop[:] = offspring + listElitism
    # Gather all the fitnesses in one list and print the stats
    fits = [ind.fitness.values[0] for ind in pop]
    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x * x for x in fits)
    std = abs(sum2 / length - mean ** 2) ** 0.5
    print("Generation " + str(g))
    print(" Min %s" % min(fits))
    print(" Max %s" % max(fits))
    print(" Avg %s" % mean)
    print(" Std %s" % std)
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

print("End of evolution")