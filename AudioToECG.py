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


def CreateDataset(Path):  # "./Data/TrainData/001"
    NameArray = ["Normal", "Inhale", "Exhale"]
    for Name in range(len(NameArray)):  # 4 collections (10secs) of breathing while ECG
        for i in range(4):
            InputPath = f"{Path}_{NameArray[Name]}{i+1}_voc.csv"
            Input_df = pd.read_csv(InputPath)
            InputTemp = Input_df.to_numpy() - 1
            InputTemp_Filtered = signal.filtfilt(
                Coeff_B, Coeff_A, InputTemp.flatten(), padlen=0
            )
            InputTemp_Filtered = InputTemp_Filtered[10:-1]
            InputTemp_Filtered = InputTemp_Filtered / np.max(np.abs(InputTemp_Filtered))

            OutputPath = f"{Path}_{NameArray[Name]}{i+1}_ecg.csv"
            Output_df = pd.read_csv(OutputPath)
            OutputTemp = Output_df.to_numpy() - 1
            OutputTemp = OutputTemp[10:-1]
            OutputTemp = OutputTemp / np.max(np.abs(OutputTemp))

            DatasetTemp = CreateMemoryDataset(
                InputTemp_Filtered, OutputTemp, MemorySize=80
            )

            if i == 0 and Name == 0:
                Dataset_DF = pd.DataFrame(DatasetTemp)
                Dataset_DF.head()
                plt.figure(1)
                plt.figure(figsize=(10, 5), dpi=300, facecolor="w")
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

    return Dataset


TrainDataset = CreateDataset("./Data/TrainData/001")
#%%

plt.figure(1)
plt.close()
plt.figure(2)
plt.close()

TestDataset = CreateDataset("./Data/TestData/020")
#%%

plt.figure(1)
plt.close()
plt.figure(2)
plt.close()

print(f"The shape of the train dataset is {TrainDataset.shape}")
print(f"The shape of the test dataset is {TestDataset.shape}")
#%%
TestDatasetModelInput = TestDataset[:, :-1]
TestDatasetModelOutput = TestDataset[:, -1]
plt.close()


# %%
import nnLACTOSE as nnLACTOSE

GoodModelDirectory = "./Model3"
LayerInfoDict = {
    "0": {
        "size": (80, 1),
        "type": "input",
    },
    "1": {"size": 100, "type": "dense"},
    "2": {"size": 100, "type": "dense"},
    "3": {"size": 100, "type": "cnn"},
    "4": {"size": 100, "type": "lstm"},
    "5": {"size": 100, "type": "lstm"},
    "6": {"size": 100, "type": "dense"},
    "7": {"size": 100, "type": "dense"},
    "8": {"size": (1), "type": "dense"},
}
ConditionArray = [-1, -0.8, -0.6, -0.4, -0.25, 0.0, 0.25, 0.4, 0.6, 0.8, 1.0]
Model = nnLACTOSE.LactoseModel(
    LayerInfoDict,
    ConditionArray,
    GoodModelDirectory,
    DisplayPlot=False,
    UseGoodModel=False,
)
ModelName = str(time())

Model.Train(Dataset=TrainDataset, ModelName=ModelName, Epochs=151)

Model.SaveModelWeights(f"./ModelWeights/Model{ModelName}/WeightOfModel")

Model.ExportLossDictionary(f"./ModelWeights/Model{ModelName}/LossDictionary")

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

OtherModels = [
    "Dense",
    "DenseGRU",
    "GRU",
    "LSTM",
    "Conv1D",
    "ConvGRU",
    "ConvLSTM",
    "SequentialModel",
]

#%%
def ModelFactory(ModelNumber):
    i = ModelNumber
    Sequential = tf.keras.Sequential()
    Sequential.add(tf.keras.layers.InputLayer(input_shape=(80, 1)))

    Sequential.add(tf.keras.layers.Dense(units=8, activation="gelu"))
    if i == 0:
        Sequential.add(tf.keras.layers.Dense(units=1, activation="gelu"))
        Sequential.add(tf.keras.layers.Dense(units=1, activation="gelu"))
    elif i == 1:
        Sequential.add(tf.keras.layers.Reshape((80, 8)))
        Sequential.add(tf.keras.layers.GRU(units=1))
    elif i == 2:
        Sequential.add(tf.keras.layers.Reshape((80, 8)))
        Sequential.add(tf.keras.layers.GRU(units=1))
        Sequential.add(tf.keras.layers.Dense(units=1, activation="gelu"))
    elif i == 3:
        Sequential.add(tf.keras.layers.Reshape((80, 8)))
        Sequential.add(tf.keras.layers.LSTM(units=100))
        Sequential.add(tf.keras.layers.Dense(units=100, activation="gelu"))
    elif i == 4:
        Sequential.add(tf.keras.layers.Reshape((80, 8)))
        Sequential.add(
            tf.keras.layers.Conv1D(filters=20, kernel_size=4, activation="gelu")
        )
        # Sequential.add(tf.keras.layers.Reshape((1,)))
        Sequential.add(tf.keras.layers.Dense(units=100, activation="gelu"))
    elif i == 5:
        Sequential.add(tf.keras.layers.Reshape((80, 8)))
        Sequential.add(
            tf.keras.layers.Conv1D(filters=20, kernel_size=4, activation="gelu")
        )
        Sequential.add(tf.keras.layers.GRU(units=100))
        Sequential.add(tf.keras.layers.Dense(units=100, activation="gelu"))
    elif i == 6:
        Sequential.add(tf.keras.layers.Reshape((80, 8)))
        Sequential.add(
            tf.keras.layers.Conv1D(filters=20, kernel_size=4, activation="gelu")
        )
        Sequential.add(tf.keras.layers.LSTM(units=100))
        Sequential.add(tf.keras.layers.Dense(units=100, activation="gelu"))
    elif i == 7:
        Sequential.add(tf.keras.layers.Dense(units=100, activation="gelu"))
        Sequential.add(tf.keras.layers.Dense(units=100, activation="gelu"))
        Sequential.add(
            tf.keras.layers.Conv1D(filters=20, kernel_size=4, activation="gelu")
        )
        Sequential.add(tf.keras.layers.LSTM(units=100))
        Sequential.add(tf.keras.layers.Dense(units=100, activation="gelu"))
        Sequential.add(tf.keras.layers.Dense(units=100, activation="gelu"))
        Sequential.add(tf.keras.layers.Dense(units=100, activation="gelu"))

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


# Train0, Test0 = ModelFactory(0)
# Train1, Test1 = ModelFactory(1)
# Train2, Test2 = ModelFactory(2)
# Train3, Test3 = ModelFactory(3)
# Train4, Test4 = ModelFactory(4)
# Train5, Test5 = ModelFactory(5)
# Train6, Test6 = ModelFactory(6)
Train7, Test7 = ModelFactory(7)

# %%
plt.figure(figsize=(10, 5), dpi=300, facecolor="w")
plt.plot(TrainDataset[:, -1], label="True")
# plt.plot(Train0, "--", label=OtherModels[0])
# plt.plot(Train1, "--", label=OtherModels[1])
# plt.plot(Train2, "--", label=OtherModels[2])
# plt.plot(Train3, "--", label=OtherModels[3])
# plt.plot(Train4, "--", label=OtherModels[4])
# plt.plot(Train5, "--", label=OtherModels[5])
# plt.plot(Train6, "--", label=OtherModels[6])
plt.plot(Train7, "--", label=OtherModels[7])
plt.legend()
plt.xlabel("Time")
plt.ylabel("Normalised Arbitrary Units")
plt.savefig(f"imgs/Model{ModelName}/CompareTrain.png", dpi=300)


# %%
plt.close()
plt.figure(figsize=(10, 5), dpi=300, facecolor="w")
plt.plot(TestDatasetModelOutput, label="True")
# plt.plot(Test0, "--", label=OtherModels[0])
# plt.plot(Test1, "--", label=OtherModels[1])
# plt.plot(Test2, "--", label=OtherModels[2])
# plt.plot(Test3, "--", label=OtherModels[3])
# plt.plot(Test4, "--", label=OtherModels[4])
# plt.plot(Test5, "--", label=OtherModels[5])
# plt.plot(Test6, "--", label=OtherModels[6])
plt.plot(Test7, "--", label=OtherModels[7])
plt.legend()
plt.xlabel("Time")
plt.ylabel("Normalised Arbitrary Units")
plt.savefig(f"imgs/Model{ModelName}/CompareTest.png", dpi=300)

# %%
plt.close()
plt.figure(figsize=(10, 5), dpi=300, facecolor="w")
ActualOutput = TrainDataset[:, -1]
LACTOSEOutput = Model.Predict(TrainDataset[:, :-1])
# plt.plot(np.abs(ActualOutput - LACTOSEOutput), label="LACTOSE")
# plt.plot(np.abs(ActualOutput - Train0.flatten()), label=OtherModels[0])
# plt.plot(np.abs(ActualOutput - Train1.flatten()), label=OtherModels[1])
# plt.plot(np.abs(ActualOutput - Train2.flatten()), label=OtherModels[2])
# plt.plot(np.abs(ActualOutput - Train3.flatten()), label=OtherModels[3])
# plt.plot(np.abs(ActualOutput - Train4.flatten()), label=OtherModels[4])
# plt.plot(np.abs(ActualOutput - Train5.flatten()), label=OtherModels[5])
# plt.plot(np.abs(ActualOutput - Train6.flatten()), label=OtherModels[6])
plt.plot(np.abs(ActualOutput - Train7.flatten()), label=OtherModels[7])
plt.legend()
plt.xlabel("Time")
plt.ylabel("Error")
plt.savefig(f"imgs/Model{ModelName}/ErrorTrain.png", dpi=300)

#%%
plt.close()
from scipy.signal import savgol_filter

LACTOSETestOutput = TestOutput
Print0 = np.abs(savgol_filter((TestDatasetModelOutput - LACTOSETestOutput), 150, 3))
# Print1 = np.abs(savgol_filter((TestDatasetModelOutput - Test0.flatten()), 150, 3))
# Print2 = np.abs(savgol_filter((TestDatasetModelOutput - Test1.flatten()), 150, 3))
# Print3 = np.abs(savgol_filter((TestDatasetModelOutput - Test2.flatten()), 150, 3))
# Print4 = np.abs(savgol_filter((TestDatasetModelOutput - Test3.flatten()), 150, 3))
# Print5 = np.abs(savgol_filter((TestDatasetModelOutput - Test4.flatten()), 150, 3))
# Print6 = np.abs(savgol_filter((TestDatasetModelOutput - Test5.flatten()), 150, 3))
# Print7 = np.abs(savgol_filter((TestDatasetModelOutput - Test6.flatten()), 150, 3))
Print8 = np.abs(savgol_filter((TestDatasetModelOutput - Test7.flatten()), 150, 3))
print(Print0.shape)
#%%
plt.close()
plt.figure(figsize=(10, 5), dpi=300, facecolor="w")
plt.plot(Print0.flatten(), label="LACTOSE")
# plt.plot(Print1.flatten(), label=OtherModels[0])
# plt.plot(Print2.flatten(), label=OtherModels[1])
# plt.plot(Print3.flatten(), label=OtherModels[2])
# plt.plot(Print4.flatten(), label=OtherModels[3])
# plt.plot(Print5.flatten(), label=OtherModels[4])
# plt.plot(Print6.flatten(), label=OtherModels[5])
# plt.plot(Print7.flatten(), label=OtherModels[6])
plt.plot(Print8.flatten(), label=OtherModels[7])
plt.legend()
plt.xlabel("Time")
plt.ylabel("Error")
plt.savefig(f"imgs/Model{ModelName}/ErrorTest.png", dpi=300)
#%%


import optuna

BATCHSIZE = 128
CLASSES = 10
EPOCHS = 1

# edit this later


def create_model(trial):
    # We optimize the numbers of layers, their units and weight decay parameter.
    n_layers = trial.suggest_int("n_layers", 1, 3)
    weight_decay = trial.suggest_float("weight_decay", 1e-10, 1e-3, log=True)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    for i in range(n_layers):
        num_hidden = trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True)
        model.add(
            tf.keras.layers.Dense(
                num_hidden,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
            )
        )
    model.add(
        tf.keras.layers.Dense(
            CLASSES, kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
        )
    )
    return


def create_optimizer(trial):
    # We optimize the choice of optimizers as well as their parameters.
    kwargs = {}
    optimizer_options = ["RMSprop", "Adam", "SGD"]
    optimizer_selected = trial.suggest_categorical("optimizer", optimizer_options)
    if optimizer_selected == "RMSprop":
        kwargs["learning_rate"] = trial.suggest_float(
            "rmsprop_learning_rate", 1e-5, 1e-1, log=True
        )
        kwargs["decay"] = trial.suggest_float("rmsprop_decay", 0.85, 0.99)
        kwargs["momentum"] = trial.suggest_float(
            "rmsprop_momentum", 1e-5, 1e-1, log=True
        )
    elif optimizer_selected == "Adam":
        kwargs["learning_rate"] = trial.suggest_float(
            "adam_learning_rate", 1e-5, 1e-1, log=True
        )
    elif optimizer_selected == "SGD":
        kwargs["learning_rate"] = trial.suggest_float(
            "sgd_opt_learning_rate", 1e-5, 1e-1, log=True
        )
        kwargs["momentum"] = trial.suggest_float(
            "sgd_opt_momentum", 1e-5, 1e-1, log=True
        )

    optimizer = getattr(tf.optimizers, optimizer_selected)(**kwargs)
    return optimizer


def learn(model, optimizer, dataset, mode="eval"):
    accuracy = tf.metrics.Accuracy("accuracy", dtype=tf.float32)

    for batch, (images, labels) in enumerate(dataset):
        with tf.GradientTape() as tape:
            logits = model(images, training=(mode == "train"))
            loss_value = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=labels
                )
            )
            if mode == "eval":
                accuracy(
                    tf.argmax(logits, axis=1, output_type=tf.int64),
                    tf.cast(labels, tf.int64),
                )
            else:
                grads = tape.gradient(loss_value, model.variables)
                optimizer.apply_gradients(zip(grads, model.variables))

    if mode == "eval":
        return accuracy


# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial):
    # Get MNIST data.
    # train_ds, valid_ds = get_mnist()

    # Build model and optimizer.
    model = create_model(trial)
    optimizer = create_optimizer(trial)

    # Training and validating cycle.
    with tf.device("/cpu:0"):
        for _ in range(EPOCHS):
            learn(model, optimizer, train_ds, "train")

        accuracy = learn(model, optimizer, valid_ds, "eval")

    # Return last validation accuracy.
    return accuracy.result()


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
