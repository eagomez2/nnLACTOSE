#%%
import sys

sys.path.insert(0, "./Data")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from time import time
import tensorflow as tf

GLOBAL_FS = 44100 / 256
Coeff_B, Coeff_A = signal.butter(8, 9, "low", fs=GLOBAL_FS)
plt.figure(figsize=(10, 5), dpi=300, facecolor="w")


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
for Name in range(
    len(NameArray) - 2
):  # 3 collections (10secs) of normal breathing while ECG
    for i in range(1):
        InputPath = f"./Data/TrainData/001_{NameArray[Name]}{i+1}_voc.csv"
        Input_df = pd.read_csv(InputPath)
        InputTemp = Input_df.to_numpy() - 1
        InputTemp_Filtered = signal.filtfilt(
            Coeff_B, Coeff_A, InputTemp.flatten(), padlen=0
        )
        InputTemp_Filtered = InputTemp_Filtered[10:-1]
        InputTemp_Filtered = InputTemp_Filtered / np.max(np.abs(InputTemp_Filtered))

        OutputPath = f"./Data/TrainData/001_{NameArray[Name]}{i+1}_ecg.csv"
        Output_df = pd.read_csv(OutputPath)
        OutputTemp = Output_df.to_numpy() - 1
        OutputTemp = OutputTemp[10:-1]
        OutputTemp = OutputTemp / np.max(np.abs(OutputTemp))

        DatasetTemp = CreateMemoryDataset(InputTemp_Filtered, OutputTemp, MemorySize=80)

        if i == 0 and Name == 0:
            Dataset_DF = pd.DataFrame(DatasetTemp)
            Dataset_DF.head()
            plt.figure(1)
            plt.plot(InputTemp_Filtered)
            plt.plot(OutputTemp)
            plt.xlabel("Time (Samples)")
            plt.ylabel("Normalised Arbitrary Units")
            plt.legend(["Input (Audio)", "Output (ECG)"])
            plt.figure(2)
            plt.figure(figsize=(10, 5), dpi=300, facecolor="w")
            plt.plot(InputTemp_Filtered, OutputTemp, ".")
            plt.xlabel("Input (Audio)")
            plt.ylabel("Output (ECG)")
            plt.title("Collapsed Time Domain Transfer Function")
            Dataset = DatasetTemp
        else:
            Dataset = np.concatenate((Dataset, DatasetTemp), axis=0)

print(f"The shape of the dataset is {Dataset.shape}")

TrainDataset = Dataset[: int(Dataset.shape[0] * 0.7)]
TestDataset = Dataset[int(Dataset.shape[0] * 0.7) :]

TestDatasetModelInput = TestDataset[:, :-1]
TestDatasetModelOutput = TestDataset[:, -1]

print(f"The shape of the train dataset is {TrainDataset.shape}")
print(f"The shape of the test dataset is {TestDataset.shape}")
#%%
plt.close()


# %%
import nnLACTOSE as nnLACTOSE

GoodModelDirectory = "./Model3"
LayerInfoDict = {
    "0": {
        "size": (80, 1),
        "type": "input",
    },
    "1": {"size": 8, "type": "dense"},
    "2": {"size": 8, "type": "dense"},
    "3": {"size": 8, "type": "cnn"},
    "4": {"size": 8, "type": "lstm"},
    "5": {"size": 8, "type": "dense"},
    "6": {"size": 8, "type": "dense"},
    "7": {"size": 8, "type": "dense"},
    "8": {"size": (1), "type": "dense"},
}
ConditionArray = [-1, -0.6, -0.25, 0.25, 0.6, 1.0]
Model = nnLACTOSE.LactoseModel(
    LayerInfoDict,
    ConditionArray,
    GoodModelDirectory,
    DisplayPlot=False,
    UseGoodModel=False,
)
ModelName = str(time())

Model.Train(Dataset=TrainDataset, ModelName=ModelName, Epochs=1000)

Model.SaveModelWeights(f"./Model{ModelName}/WeightOfModel")

Model.ExportLossDictionary(f"./Model{ModelName}/LossDictionary")

LossDictionary = Model.ModelLosses

plt.close()
NumberOfPlots = 0
plt.rcParams["figure.figsize"] = (9, 9)
for i in range(len(ConditionArray) - 1):
    PlotArray = LossDictionary[f"loss_history{i}"]
    if len(PlotArray) > 5:
        NumberOfPlots += 1
        print(len(PlotArray))
fig, ax = plt.subplots(NumberOfPlots)
PlotInNumber = 0
for i in range(len(ConditionArray) - 1):
    PlotArray = LossDictionary[f"loss_history{i}"]
    if len(PlotArray) > 5:
        ax[PlotInNumber].plot(PlotArray)
        ax[PlotInNumber].set(xlabel="Network Training Epochs", ylabel="Loss")
        ax[PlotInNumber].set_title(f"Network of Condition {i}")
        PlotInNumber += 1
plt.subplots_adjust(wspace=1, hspace=1)
plt.savefig(f"imgs/Model{ModelName}/IndivModelLoss.png", dpi=300)

#%%
TestOutput = Model.Predict(TestDatasetModelInput)

plt.close()
plt.figure(figsize=(10, 5), dpi=300, facecolor="w")
plt.plot(TestDatasetModelOutput)
plt.plot(TestOutput)
plt.xlabel("Time")
plt.ylabel("Normalised Arbitrary Units")
plt.legend(["Output (ECG)", "Predicted Output (ECG)"])
plt.title("Test Data")
plt.savefig(f"imgs/Model{ModelName}/TestData.png", dpi=300)

# %%
plt.close()


TrainDataSequentialModel = tf.data.Dataset.from_tensor_slices(
    (TrainDataset[:, :-1], TrainDataset[:, -1])
)

OtherModels = ["Dense", "DenseGRU", "GRU", "LSTM", "Conv1D", "ConvGRU", "ConvLSTM"]


def ModelFactory(ModelNumber):
    i = ModelNumber
    Sequential = tf.keras.Sequential()
    Sequential.add(tf.keras.layers.InputLayer(input_shape=(80, 1)))

    Sequential.add(tf.keras.layers.Dense(units=8, activation="gelu"))
    if i == 0:
        Sequential.add(tf.keras.layers.Dense(units=80, activation="gelu"))
        Sequential.add(tf.keras.layers.Dense(units=80, activation="gelu"))
    elif i == 1:
        Sequential.add(tf.keras.layers.Reshape((80, 8)))
        Sequential.add(tf.keras.layers.GRU(units=80))
    elif i == 2:
        Sequential.add(tf.keras.layers.Reshape((80, 8)))
        Sequential.add(tf.keras.layers.GRU(units=80))
        Sequential.add(tf.keras.layers.Dense(units=80, activation="gelu"))
    elif i == 3:
        Sequential.add(tf.keras.layers.Reshape((80, 8)))
        Sequential.add(tf.keras.layers.LSTM(units=8))
        Sequential.add(tf.keras.layers.Dense(units=8, activation="gelu"))
    elif i == 4:
        Sequential.add(tf.keras.layers.Reshape((80, 8)))
        Sequential.add(
            tf.keras.layers.Conv1D(filters=2, kernel_size=4, activation="gelu")
        )
        # Sequential.add(tf.keras.layers.Reshape((1,)))
        Sequential.add(tf.keras.layers.Dense(units=8, activation="gelu"))
    elif i == 5:
        Sequential.add(tf.keras.layers.Reshape((80, 8)))
        Sequential.add(
            tf.keras.layers.Conv1D(filters=2, kernel_size=4, activation="gelu")
        )
        Sequential.add(tf.keras.layers.GRU(units=8))
        Sequential.add(tf.keras.layers.Dense(units=8, activation="gelu"))
    elif i == 6:
        Sequential.add(tf.keras.layers.Reshape((80, 8)))
        Sequential.add(
            tf.keras.layers.Conv1D(filters=2, kernel_size=4, activation="gelu")
        )
        Sequential.add(tf.keras.layers.LSTM(units=8))
        Sequential.add(tf.keras.layers.Dense(units=8, activation="gelu"))

    Sequential.add(tf.keras.layers.Flatten())
    Sequential.add(tf.keras.layers.Dense(units=1, activation="gelu"))

    Sequential.compile(
        optimizer=tf.keras.optimizers.RMSprop(),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=tf.keras.metrics.Accuracy(),
    )
    Sequential.build()
    Sequential.summary()

    Sequential.fit(TrainDataset[:, :-1], TrainDataset[:, -1], epochs=151, batch_size=30)
    TrainPredict = Sequential.predict(TrainDataset[:, :-1])
    TestPredict = Sequential.predict(TestDatasetModelInput)
    return TrainPredict, TestPredict


Train0, Test0 = ModelFactory(0)
Train1, Test1 = ModelFactory(1)
Train2, Test2 = ModelFactory(2)
Train3, Test3 = ModelFactory(3)
Train4, Test4 = ModelFactory(4)
Train5, Test5 = ModelFactory(5)
Train6, Test6 = ModelFactory(6)

# %%
plt.figure(figsize=(10, 5), dpi=300, facecolor="w")
plt.plot(TrainDataset[:, -1], label="True")
plt.plot(Train0, "--", label=OtherModels[0])
plt.plot(Train1, "--", label=OtherModels[1])
plt.plot(Train2, "--", label=OtherModels[2])
plt.plot(Train3, "--", label=OtherModels[3])
plt.plot(Train4, "--", label=OtherModels[4])
plt.plot(Train5, "--", label=OtherModels[5])
plt.plot(Train6, "--", label=OtherModels[6])
plt.legend()
plt.xlabel("Time")
plt.ylabel("Normalised Arbitrary Units")
plt.savefig(f"imgs/Model{ModelName}/CompareTrain.png", dpi=300)


# %%
plt.close()
plt.figure(figsize=(10, 5), dpi=300, facecolor="w")
plt.plot(TestDatasetModelOutput, label="True")
plt.plot(Test0, "--", label=OtherModels[0])
plt.plot(Test1, "--", label=OtherModels[1])
plt.plot(Test2, "--", label=OtherModels[2])
plt.plot(Test3, "--", label=OtherModels[3])
plt.plot(Test4, "--", label=OtherModels[4])
plt.plot(Test5, "--", label=OtherModels[5])
plt.plot(Test6, "--", label=OtherModels[6])
plt.legend()
plt.xlabel("Time")
plt.ylabel("Normalised Arbitrary Units")
plt.savefig(f"imgs/Model{ModelName}/CompareTest.png", dpi=300)

# %%
plt.close()
plt.figure(figsize=(10, 5), dpi=300, facecolor="w")
ActualOutput = TrainDataset[:, -1]
LACTOSEOutput = Model.Predict(TrainDataset[:, :-1])
plt.plot(np.abs(ActualOutput - LACTOSEOutput), label="LACTOSE")
plt.plot(np.abs(ActualOutput - Train0.flatten()), label=OtherModels[0])
plt.plot(np.abs(ActualOutput - Train1.flatten()), label=OtherModels[1])
plt.plot(np.abs(ActualOutput - Train2.flatten()), label=OtherModels[2])
plt.plot(np.abs(ActualOutput - Train3.flatten()), label=OtherModels[3])
plt.plot(np.abs(ActualOutput - Train4.flatten()), label=OtherModels[4])
plt.plot(np.abs(ActualOutput - Train5.flatten()), label=OtherModels[5])
plt.plot(np.abs(ActualOutput - Train6.flatten()), label=OtherModels[6])
plt.legend()
plt.xlabel("Time")
plt.ylabel("Error")
plt.savefig(f"imgs/Model{ModelName}/ErrorTrain.png", dpi=300)

#%%
plt.close()
from scipy.signal import savgol_filter

LACTOSETestOutput = TestOutput
Print0 = np.abs(savgol_filter((TestDatasetModelOutput - LACTOSETestOutput), 150, 3))
Print1 = np.abs(savgol_filter((TestDatasetModelOutput - Test0.flatten()), 150, 3))
Print2 = np.abs(savgol_filter((TestDatasetModelOutput - Test1.flatten()), 150, 3))
Print3 = np.abs(savgol_filter((TestDatasetModelOutput - Test2.flatten()), 150, 3))
Print4 = np.abs(savgol_filter((TestDatasetModelOutput - Test3.flatten()), 150, 3))
Print5 = np.abs(savgol_filter((TestDatasetModelOutput - Test4.flatten()), 150, 3))
Print6 = np.abs(savgol_filter((TestDatasetModelOutput - Test5.flatten()), 150, 3))
Print7 = np.abs(savgol_filter((TestDatasetModelOutput - Test6.flatten()), 150, 3))
print(Print0.shape)
#%%
plt.close()
plt.figure(figsize=(10, 5), dpi=300, facecolor="w")
plt.plot(Print0.flatten(), label="LACTOSE")
plt.plot(Print1.flatten(), label=OtherModels[0])
plt.plot(Print2.flatten(), label=OtherModels[1])
plt.plot(Print3.flatten(), label=OtherModels[2])
plt.plot(Print4.flatten(), label=OtherModels[3])
plt.plot(Print5.flatten(), label=OtherModels[4])
plt.plot(Print6.flatten(), label=OtherModels[5])
plt.plot(Print7.flatten(), label=OtherModels[6])
plt.legend()
plt.xlabel("Time")
plt.ylabel("Error")
plt.savefig(f"imgs/Model{ModelName}/ErrorTest.png", dpi=300)
#%%
