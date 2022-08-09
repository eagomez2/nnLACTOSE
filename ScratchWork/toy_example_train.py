#%%
from datagen import DataGenerator
from conditional_models import ConditionalDenseModel

import tensorflow as tf
import numpy as np
import pandas as pd

printCoeffs = False
printConds = True

MyDataGenerator = DataGenerator(48000, 0.1)
Input, Output, t, Conds, Coeffs = MyDataGenerator.GenerateData()
NewOutput = MyDataGenerator.GenerateDataAgain(Output)
MyDataGenerator.PlotTransferFunction(NewOutput)
Conds = np.sort(Conds)
if printCoeffs:
    print(Coeffs)
if printConds:
    print("The conditions are as follows")
    print(Conds)

#%%
Dataset = []
for i in range(len(Input)):
    Dataset.append([Input[i], NewOutput[i]])

# print(Dataset[0])
myModel = ConditionalDenseModel()
(
    model,
    loss_fn,
    optimizer,
    metric,
    SavedWeights1,
    SavedWeights2,
    SavedWeights3,
    SavedWeights4,
) = myModel.GetModel()

myModel.CustomTrainingLoop(10, Dataset, model, loss_fn, optimizer, metric, Conds)
Output, TruePlot = myModel.PlotOutput(model, Dataset)

# %%
import matplotlib.pyplot as plt

Output = np.array(Output).flatten()
TruePlot = np.array(TruePlot).flatten()
plt.plot(Output)
plt.plot(TruePlot)

# %%
