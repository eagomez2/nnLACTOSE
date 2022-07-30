# %%
import sys

sys.path.insert(0, "./Data_128x_downsample_shifted")

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

input_to_model_path = "./Data_128x_downsample_shifted/001_Exhale1_voc.csv"
model_in_df = pd.read_csv(input_to_model_path)
Input = model_in_df.to_numpy()

# input_to_model_path3 = "Data_128x_downsample_shifted_dcshift_all/001_Exhale1_bcg.csv"
# model_in_df3 = pd.read_csv(input_to_model_path3)
# model_in_np3 = model_in_df3.to_numpy()

# model_in_np = np.concatenate((model_in_np, model_in_np3), axis=1)
# print(model_in_np)
truth_path = "Data_128x_downsample_shifted/001_Exhale1_ecg.csv"
truth_df = pd.read_csv(truth_path)
Output = truth_df.to_numpy()
# print(truth_np)

plt.figure()
plt.plot(Input)
plt.plot(Output)

plt.figure()
plt.plot(Input, Output)

# %%
Dataset = []
for i in range(len(Input) - 1):
    Dataset.append([Input[i], Input[i + 1], Output[i]])
# %%
Conds = [-1, -0.5, 0, 0.5, 1]

from conditional_models import ConditionalDenseModel
import tensorflow as tf

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

myModel.CustomTrainingLoop(50, Dataset, model, loss_fn, optimizer, metric, Conds)
Output, TruePlot = myModel.PlotOutput(model, Dataset)
myModel.SaveModelWeights(model, "./Model1/")

# %%
import matplotlib.pyplot as plt

Output = np.array(Output).flatten()
Output = Output / np.max(np.abs(Output))
TruePlot = np.array(TruePlot).flatten()
plt.plot(Output)
plt.plot(TruePlot)

# %%
plt.figure()
plt.plot(Input)
plt.plot(Output)
# %%
