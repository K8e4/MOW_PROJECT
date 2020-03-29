from termcolor import colored
import pandas as pd
import matplotlib.pyplot as plt

def plot_var_distribution(dataset, varname, fig):
    fig.plot(
        dataset[varname].index,
        dataset[varname].values,
        label=varname
    )

    fig.title.set_text(varname)

def plot_in_single_frame(title, dataset, afn):
    fig = plt.figure()
    fig.suptitle(title, fontsize=16)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1, hspace=1)

    for i, column_name in enumerate(dataset.columns):
        sub = plt.subplot(4,3,i+1)
        afn(dataset, column_name, sub);

    plt.show()


def boxplot(dataset, varname, fig):
   dataset.boxplot(column=varname)


def run(dataset):
    print(colored('Sum of missing data in dataset:\n', 'green'))
    print(dataset.isnull().sum())

    print(colored('\n------------------------------------------\n', 'green'))

    print(colored('Describe the dataset:\n', 'green'))
    print(dataset.describe())

    print(colored('\n------------------------------------------\n', 'green'))

    plot_in_single_frame('Dataset', dataset, plot_var_distribution)

    plot_in_single_frame('Boxplots of the dataset', dataset, boxplot)
