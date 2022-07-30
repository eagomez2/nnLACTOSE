# %%
import tensorflow as tf
import numpy as np
import pandas as pd
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

        self.InputSize = list(self.InputSize)
        newInputSize = list()
        newInputSize.append(1)
        for i in range(len(self.InputSize)):
            newInputSize.append(self.InputSize[i])
        self.InputSize = tuple(newInputSize)

        self.LossFunction = tf.keras.losses.MeanAbsolutePercentageError()
        self.Optimizer = tf.keras.optimizers.RMSprop()
        self.Metric = tf.keras.metrics.Accuracy()
        # tf.print(model(tf.ones(self.InputSize)))

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
            if Input < self.ConditionArray[i]:
                raise Exception(
                    "ERROR: Input is less than smallest condition. Check conditions or input.",
                )
            if i == 0:
                if Input == self.ConditionArray[i]:
                    return self.SavedWeightsDict[f"{i}"]
            else:
                if self.ConditionArray[i - 1] <= Input < self.ConditionArray[i]:
                    return self.SavedWeightsDict[f"{i-1}"]
                if Input == self.ConditionArray[-1]:
                    return self.SavedWeightsDict[f"{len(self.ConditionArray)}"]

    def Train(self, Dataset, Epochs=1, CheckInputNumber=-1):
        for epoch in range(Epochs):
            print(f"Epoch {epoch}")

            for step, FeatureAndAnswer in enumerate(Dataset):
                Input = FeatureAndAnswer[0:-2]
                Answer = FeatureAndAnswer[-1]

                ModelWeights = self.CheckInputAndReturnModel(Input[CheckInputNumber])
                self.Model.set_weights(ModelWeights)

                Input = tf.reshape(Input, self.InputSize)
                Answer = tf.reshape(Answer, (1,))

                with tf.GradientTape() as tape:
                    Prediction = self.Model(Input)
                    Loss = self.LossFunction(Answer, Prediction)
                grads = tape.gradient(Loss, self.Model.trainable_weights)
                self.Optimizer.apply_gradients(zip(grads, self.Model.trainable_weights))

                # Self.Model.get_weights() ## SAVE THE MODEL WEIGHTS
