import sys

sys.path.insert(0, "./Data")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

GLOBAL_FS = 44100 / 256
Coeff_B, Coeff_A = signal.butter(8, 9, "low", fs=GLOBAL_FS)


def CreateMemoryDataset(Input, Output, MemorySize, WithOutput=True):
    Dataset = np.array([])
    FeatureAndAnswerVector = np.array([])
    for i in range(len(Input)):
        if i > MemorySize:
            for j in range(MemorySize, 0, -1):
                FeatureAndAnswerVector = np.append(FeatureAndAnswerVector, Input[i - j])
            if WithOutput:
                FeatureAndAnswerVector = np.append(FeatureAndAnswerVector, Output[i])
            if i == MemorySize + 1:
                Dataset = FeatureAndAnswerVector[None, :]
            else:
                Dataset = np.concatenate(
                    (Dataset, FeatureAndAnswerVector[None, :]), axis=0
                )
            FeatureAndAnswerVector = np.array([])
    return Dataset


def FFTPlot(Input):
    Input_FFT = np.fft.fft(Input)
    N = len(Input)
    n = np.arange(N)
    T = N / GLOBAL_FS
    Freqs = n / T

    plt.plot(Freqs, np.abs(Input_FFT))
    plt.xlim((0, GLOBAL_FS / 2))


NameArray = ["Normal", "Inhale", "Exhale"]
for Name in range(len(NameArray)):
    for i in range(4):
        InputPath = f"./Data/001_{NameArray[Name]}{i+1}_bcg.csv"
        Input_df = pd.read_csv(InputPath)
        InputTemp = Input_df.to_numpy() - 1
        InputTemp_Filtered = signal.filtfilt(
            Coeff_B, Coeff_A, InputTemp.flatten(), padlen=0
        )
        InputTemp_Filtered = InputTemp_Filtered[10:-1]
        InputTemp_Filtered = InputTemp_Filtered / np.max(np.abs(InputTemp_Filtered))

        OutputPath = f"./Data/001_{NameArray[Name]}{i+1}_ecg.csv"
        Output_df = pd.read_csv(OutputPath)
        OutputTemp = Output_df.to_numpy() - 1
        OutputTemp = OutputTemp[10:-1]
        OutputTemp = OutputTemp / np.max(np.abs(OutputTemp))

        DatasetTemp = CreateMemoryDataset(
            InputTemp_Filtered, OutputTemp, MemorySize=400
        )
        # Dataset_DF = pd.DataFrame(Dataset)
        # Dataset_DF.head()

        if i == 0 and Name == 0:
            plt.plot(InputTemp_Filtered)
            plt.plot(OutputTemp)
            Dataset = DatasetTemp
        else:
            Dataset = np.concatenate((Dataset, DatasetTemp), axis=0)

print(f"The shape of the dataset is {Dataset.shape}")
#%%
plt.close()


# %%
import nnLACTOSE as nnLACTOSE

LayerInfoDict = {
    "0": {"size": (1, 400), "type": "input"},
    "1": {"size": 1000, "type": "dense"},
    "2": {"size": 300, "type": "gru"},
    "3": {"size": 200, "type": "dense"},
    "4": {"size": 1, "type": "dense"},
}
ConditionArray = [-1, -0.6, -0.2, 0.2, 0.5, 0.8, 1.0]
Model = nnLACTOSE.LactoseModel(LayerInfoDict, ConditionArray, DisplayPlot=False)

Model.Train(Dataset=Dataset, Epochs=200)

Model.SaveModelWeights("./Model1/WeightOfModel")

Model.ExportLossDictionary("./Model1/LossDictionary")
