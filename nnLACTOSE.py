# %%
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import perf_counter_ns
import sys
from IPython.display import Image, display

# %%
class LactoseModel:
    def __init__(self, LayerInfoDict, ConditionArray, DisplayPlot=True):
        StartTime = perf_counter_ns()
        self.LayerInfoDict = LayerInfoDict
        ###############################################
        # FOR LOOP TO ACCESS EACH LAYER IN DICTIONARY #
        ###############################################
        for i in range(len(self.LayerInfoDict)):
            # print(self.LayerInfoDict[f"{i+1}"])
            CurrentLayer = self.LayerInfoDict[f"{i}"]
            SizeOfCurrentLayer = CurrentLayer["size"]
            if i == 0:
                self.InputSize = SizeOfCurrentLayer
            CurrentLayerType = CurrentLayer["type"]
            ###############################
            # THIS CREATES THE MODEL DICT #
            ###############################
            if CurrentLayerType == "input":
                input = x = tf.keras.layers.Input(
                    shape=SizeOfCurrentLayer,
                    batch_size=1,
                    name="Input",
                )
            if CurrentLayerType == "dense":
                x = tf.keras.layers.Dense(
                    SizeOfCurrentLayer,
                    activation="relu",
                    kernel_initializer=tf.keras.initializers.RandomNormal(),
                )(x)
            if CurrentLayerType == "gru":
                x = GRULayer = tf.keras.layers.GRU(
                    SizeOfCurrentLayer,
                    activation="tanh",
                    recurrent_activation="sigmoid",
                    use_bias=True,
                    kernel_initializer="glorot_uniform",
                    recurrent_initializer="orthogonal",
                    bias_initializer="zeros",
                    kernel_regularizer=None,
                    recurrent_regularizer=None,
                    bias_regularizer=None,
                    activity_regularizer=None,
                    kernel_constraint=None,
                    recurrent_constraint=None,
                    bias_constraint=None,
                    dropout=0.0,
                    recurrent_dropout=0.0,
                    return_sequences=True,
                    return_state=False,
                    go_backwards=False,
                    stateful=False,
                    unroll=False,
                    time_major=False,
                    reset_after=True,
                )(x)
            if i == len(self.LayerInfoDict) - 1:
                self.OutputSize = SizeOfCurrentLayer

        output = tf.keras.layers.Flatten()(x)
        self.Model = tf.keras.models.Model(inputs=input, outputs=output)
        self.Model.summary()
        ModelPlot = tf.keras.utils.plot_model(
            self.Model,
            "./nnLactoseModel.png",
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=96,
            layer_range=None,
            show_layer_activations=True,
        )
        if DisplayPlot:
            display(Image("./nnLactoseModel.png"))

        if type(self.InputSize) is tuple:
            self.InputSize = list(self.InputSize)
            newInputSize = list()
            newInputSize.append(1)
            for i in range(len(self.InputSize)):
                newInputSize.append(self.InputSize[i])
        elif type(self.InputSize) is int:
            newInputSize = list()
            newInputSize.append(self.InputSize)
        else:
            raise Exception("Input size is not an int or tuple.")

        self.InputSize = tuple(newInputSize)

        if type(self.OutputSize) is tuple:
            self.OutputSize = list(self.OutputSize)
            newOutputSize = list()
            newOutputSize.append(1)
            for i in range(len(self.OutputSize)):
                newOutputSize.append(self.OutputSize[i])
        elif type(self.OutputSize) is int:
            newOutputSize = list()
            newOutputSize.append(self.OutputSize)
        else:
            raise Exception("Output size is not an int or tuple.")

        self.OutputSize = tuple(newOutputSize)

        self.LossFunction = tf.keras.losses.MeanAbsolutePercentageError()
        self.Optimizer = tf.keras.optimizers.RMSprop()
        self.Metric = tf.keras.metrics.Accuracy()
        self.Model(tf.ones(self.InputSize))

        self.ConditionArray = ConditionArray
        self.NumberOfModelsRequired = len(self.ConditionArray) - 1

        self.SavedWeightsDict = dict()
        for i in range(self.NumberOfModelsRequired):
            self.SavedWeightsDict[f"{i}"] = self.Model.get_weights()

        StopTime = perf_counter_ns()
        print(f"Time to create model: {(StopTime - StartTime) / 1000000000} seconds")

    def CheckInputAndReturnModel(self, Input):
        for i in range(len(self.ConditionArray)):
            if self.ConditionArray[-1] < Input:
                raise Exception(
                    "ERROR: Input is greater than largest condition. Check conditions or input."
                )
            if Input < self.ConditionArray[0]:
                raise Exception(
                    "ERROR: Input is less than smallest condition. Check conditions or input.",
                )
            if i == 0:
                if Input == self.ConditionArray[i]:
                    return self.SavedWeightsDict[f"{i}"], i
            else:
                if self.ConditionArray[i - 1] <= Input < self.ConditionArray[i]:
                    return self.SavedWeightsDict[f"{i-1}"], i
                if Input == self.ConditionArray[-1]:
                    return self.SavedWeightsDict[f"{len(self.ConditionArray)}"], i

    def CheckInputAndSaveModelWeights(self, Input, Weights):
        for i in range(len(self.ConditionArray)):
            if self.ConditionArray[-1] < Input:
                raise Exception(
                    "ERROR: Input is greater than largest condition. Check conditions or input."
                )
            if Input < self.ConditionArray[0]:
                raise Exception(
                    "ERROR: Input is less than smallest condition. Check conditions or input.",
                )
            if i == 0:
                if Input == self.ConditionArray[i]:
                    self.SavedWeightsDict[f"{i}"] = Weights
            else:
                if self.ConditionArray[i - 1] <= Input < self.ConditionArray[i]:
                    self.SavedWeightsDict[f"{i-1}"] = Weights
                if Input == self.ConditionArray[-1]:
                    self.SavedWeightsDict[f"{len(self.ConditionArray)}"] = Weights

    def Train(self, Dataset, Epochs=1, CheckInputNumber=-1):
        ModelLosses = dict()
        for i in range(self.NumberOfModelsRequired):
            ModelLosses[f"metric_history{i}"] = np.array([])
            ModelLosses[f"loss_history{i}"] = np.array([])
        for epoch in range(Epochs):
            print(f"Epoch {epoch}")

            for step, FeatureAndAnswer in enumerate(Dataset):
                print(f"Step {step} of Epoch {epoch}")

                Input = FeatureAndAnswer[0 : (len(FeatureAndAnswer) - 1)]
                Answer = FeatureAndAnswer[len(FeatureAndAnswer) - 1]
                InputCheck = Input[CheckInputNumber]
                ##########################
                # RETRIEVE MODEL WEIGHTS #
                ##########################
                ModelWeights, ModelNumber = self.CheckInputAndReturnModel(InputCheck)
                self.Model.set_weights(ModelWeights)

                Input = tf.reshape(Input, self.InputSize)
                Answer = tf.reshape(Answer, self.OutputSize)

                with tf.GradientTape() as tape:
                    Prediction = self.Model(Input)
                    Loss = self.LossFunction(Answer, Prediction)
                grads = tape.gradient(Loss, self.Model.trainable_weights)
                self.Optimizer.apply_gradients(zip(grads, self.Model.trainable_weights))
                ######################
                # SAVE MODEL WEIGHTS #
                ######################
                SavedWeight = self.Model.get_weights()
                self.CheckInputAndSaveModelWeights(InputCheck, SavedWeight)

                self.Metric.update_state(Answer, Prediction)
                for i in range(self.NumberOfModelsRequired):
                    if i == ModelNumber:
                        ModelLosses[f"{i}"] = Loss.numpy()

                        ModelLosses[f"metric_history{i}"] = np.append(
                            ModelLosses[f"metric_history{i}"],
                            self.Metric.result().numpy(),
                        )
                        ModelLosses[f"loss_history{i}"] = np.append(
                            ModelLosses[f"loss_history{i}"], Loss.numpy()
                        )

                for i in range(self.NumberOfModelsRequired):

                    if i != self.NumberOfModelsRequired - 1:
                        print(ModelLosses[f"{i}"])
                    else:
                        print(ModelLosses[f"{i}"], end="\r", flush=True)

            self.Train_Accuracy = self.Metric.result()
            self.Metric.reset_states()

    def Predict(self, Input, CheckInputNumber=-1, Plot=False):
        Output = np.array([])
        for Features in enumerate(Input):
            InputToModel = Features
            InputCheck = InputToModel[CheckInputNumber]
            ModelWeights, ModelNumber = self.CheckInputAndReturnModel(InputCheck)
            self.Model.set_weights(ModelWeights)
            InputToModel = tf.reshape(InputToModel, self.InputSize)
            Output = np.append(Output, self.Model(InputToModel).numpy())
        if Plot:
            plt.plot(Output)
            plt.show()

    def SaveModelWeights(self, FileName):
        for i in range(self.NumberOfModelsRequired):
            Weights = self.SavedWeightsDict[f"{i}"]
            self.Model.set_weights(Weights)
            Config = self.Model.to_json(f"{FileName}_{i}.json")
            # CHECK IF WORKS?


# %%
