import numpy as np
import winequality.analysis as analysis
import winequality.reader as r
import winequality.steps.transform as t
from termcolor import colored

raw_dataset = r.dataset()

print(colored('\n-------------------------------------------------------------------\n', 'red'))
print(colored('Analyze the raw data:\n', 'red'))

analysis.run(raw_dataset)

print(colored('\n-------------------------------------------------------------------\n', 'red'))

analysis.run(t.transform(raw_dataset))
