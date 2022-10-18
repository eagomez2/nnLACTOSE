# %%
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
plt.close()
#%%

SequentialModel = tf.keras.Sequential()
SequentialModel.add(tf.keras.layers.Input(shape=(80,)))
SequentialModel.add(tf.keras.layers.Dense(2000, activation="gelu"))
SequentialModel.add(tf.keras.layers.Reshape((2000, 1)))
SequentialModel.add(tf.keras.layers.LSTM(800, activation="gelu"))
SequentialModel.add(tf.keras.layers.Dense(600, activation="gelu"))
SequentialModel.add(tf.keras.layers.Dense(200, activation="gelu"))
SequentialModel.add(tf.keras.layers.Dense(50, activation="gelu"))
SequentialModel.add(tf.keras.layers.Dense(1, activation="gelu"))


SequentialModel.compile(
    optimizer=tf.keras.optimizers.RMSprop(),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=tf.keras.metrics.Accuracy(),
)
SequentialModel.build()
SequentialModel.summary()

SequentialModel.fit(TrainDataset[:, :-1], TrainDataset[:, -1], epochs=10, batch_size=32)
TrainPredict = SequentialModel.predict(TrainDataset[:, :-1])
TestPredict = SequentialModel.predict(TestDatasetModelInput)


plt.figure(figsize=(10, 5), dpi=300, facecolor="w")
plt.plot(TrainDataset[:, -1], label="True")
plt.plot(TrainPredict, label="Predicted")
plt.close()

plt.figure(figsize=(10, 5), dpi=300, facecolor="w")
plt.plot(TestDatasetModelOutput, label="True")
plt.plot(TestPredict, label="Predicted")
# %%
