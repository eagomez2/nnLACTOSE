#%%
import sys

sys.path.insert(0, "./Data_128x_downsample_shifted")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

input_to_model_path = "./Data_128x_downsample_shifted/001_Exhale1_voc.csv"
model_in_df = pd.read_csv(input_to_model_path)
Input = model_in_df.to_numpy()

truth_path = "Data_128x_downsample_shifted/001_Exhale1_ecg.csv"
truth_df = pd.read_csv(truth_path)
Output = truth_df.to_numpy()

Input = Input / np.max(np.abs(Input))
Output = Output / np.max(np.abs(Output))


def CreateMemoryDataset(Input, Output, MemorySize):
    Dataset = np.array([])
    FeatureAndAnswerVector = np.array([])
    for i in range(len(Input)):
        if i > MemorySize:
            for j in range(MemorySize, 0, -1):
                FeatureAndAnswerVector = np.append(FeatureAndAnswerVector, Input[i - j])
            FeatureAndAnswerVector = np.append(FeatureAndAnswerVector, Output[i])
            if i == MemorySize + 1:
                Dataset = FeatureAndAnswerVector[None, :]
            else:
                Dataset = np.concatenate(
                    (Dataset, FeatureAndAnswerVector[None, :]), axis=0
                )
            FeatureAndAnswerVector = np.array([])
    return Dataset


Dataset = CreateMemoryDataset(Input, Output, MemorySize=5)
Dataset_DF = pd.DataFrame(Dataset)
Dataset_DF.head()

# %%
import nnLACTOSE as nnLACTOSE

LayerInfoDict = {
    "0": {"size": (1, 5), "type": "input"},
    "1": {"size": 200, "type": "dense"},
    "2": {"size": 200, "type": "gru"},
    "3": {"size": 200, "type": "dense"},
    "4": {"size": 1, "type": "dense"},
}
ConditionArray = [-1, -0.6, -0.2, 0.2, 0.5, 0.8, 1.0]
Model = nnLACTOSE.LactoseModel(LayerInfoDict, ConditionArray, DisplayPlot=False)

Model.Train(Dataset=Dataset, Epochs=50)

Model.SaveModelWeights("./Model1/WeightOfModel")

# %%
Model.Predict(Input, Plot=True)
