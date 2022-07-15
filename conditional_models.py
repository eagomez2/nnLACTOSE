#%%
import tensorflow as tf
import numpy as np
import time


#######################
# LOSS FN DOESNT WORK #
#######################
class SummedLossWithMSE(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
        self.Loss = 0

    def call(self, y_true, y_pred):
        mse = tf.square(y_pred.numpy().flatten() - y_true.numpy().flatten())
        summed = tf.abs(y_pred.numpy().flatten() - y_true.numpy().flatten())
        if self.Loss == 0:
            self.Loss = summed + mse + 1
        else:
            self.Loss = self.Loss + summed + mse
        return self.Loss


def CLoss(y_true, y_pred):
    mse = tf.square(y_pred.numpy().flatten() - y_true.numpy().flatten())
    summed = tf.abs(y_pred.numpy().flatten() - y_true.numpy().flatten())
    return summed + mse


class ConditionalDenseModel:
    ###########################################
    # CONDITIONAL MODEL RELIES ON DENSE MODEL #
    ###########################################
    def DefaultDenseModel(self):
        input = tf.keras.Input(shape=(1), name="Input")
        dense1 = tf.keras.layers.Dense(
            8, activation="relu", name="Dense1"
        )  # first layer
        x = dense1(input)
        x = tf.keras.layers.Dense(8, activation="relu", name="Dense2")(
            x
        )  # second layer
        output = tf.keras.layers.Dense(1, name="Output")(x)  # output layer

        model = tf.keras.Model(inputs=input, outputs=output)
        model.summary()

        return model

    def __init__(self):
        self.model = self.DefaultDenseModel()
        self.loss_fn = tf.keras.losses.MeanAbsolutePercentageError()
        self.optimizer = tf.keras.optimizers.RMSprop()
        self.metric = tf.keras.metrics.Accuracy()
        self.model(
            tf.ones(
                1,
            )
        )
        self.SavedWeights1 = self.model.get_weights()
        self.SavedWeights2 = self.model.get_weights()
        self.SavedWeights3 = self.model.get_weights()
        self.SavedWeights4 = self.model.get_weights()

    def GetModel(self):
        return (
            self.model,
            self.loss_fn,
            self.optimizer,
            self.metric,
            self.SavedWeights1,
            self.SavedWeights2,
            self.SavedWeights3,
            self.SavedWeights4,
        )

    def CustomTrainingLoop(
        self,
        NumEpochs,
        dataset,
        model,
        loss_fn,
        optimizer,
        metric,
        Conds,
        val_dataset=None,
        val_metric=None,
    ):
        Conds1 = Conds[0]
        Conds2 = Conds[1]
        Conds3 = Conds[2]
        Conds4 = Conds[3]
        Conds5 = Conds[4]

        Network1loss = 0
        Network2loss = 0
        Network3loss = 0
        Network4loss = 0

        epochs = NumEpochs
        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))
            start_time = time.time()

            # Iterate over the batches of the dataset.
            for step, (x, y) in enumerate(dataset):
                if Conds1 <= x < Conds2:
                    model.set_weights(self.SavedWeights1)
                elif Conds2 <= x < Conds3:
                    model.set_weights(self.SavedWeights2)
                elif Conds3 <= x < Conds4:
                    model.set_weights(self.SavedWeights3)
                elif Conds4 <= x < Conds5:
                    model.set_weights(self.SavedWeights4)

                x = tf.reshape(x, (1))
                y = tf.reshape(y, (1))
                with tf.GradientTape() as tape:
                    output = model(x)
                    loss_value = loss_fn(y, output)
                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                if Conds1 <= x < Conds2:
                    self.SavedWeights1 = model.get_weights()
                    Network1loss = loss_value
                elif Conds2 <= x < Conds3:
                    self.SavedWeights2 = model.get_weights()
                    Network2loss = loss_value
                elif Conds3 <= x < Conds4:
                    self.SavedWeights3 = model.get_weights()
                    Network3loss = loss_value
                elif Conds4 <= x < Conds5:
                    self.SavedWeights4 = model.get_weights()
                    Network4loss = loss_value

                # Update training metric.
                metric.update_state(y, output)

                # Log every 200 batches.

                print(f"Training loss for at step {step}")
                print(
                    f"Network 1 is {Network1loss} Network 2 is {Network2loss} Network 3 is {Network3loss} Network 4 is {Network4loss}"
                )

                # print("Seen so far: %d samples" % ((step)))
                print(f"Epoch: {epoch}")

            # Display metrics at the end of each epoch.
            train_acc = metric.result()
            # print("Training acc over epoch: %.4f" % (float(train_acc),))

            # Reset training metrics at the end of each epoch
            metric.reset_states()

            # # Run a validation loop at the end of each epoch.
            # for x_val, y_val in val_dataset:
            #     val_output = model(x_val, training=False)
            #     # Update val metrics
            #     val_metric.update_state(y_val, val_output)
            # val_acc = val_metric.result()
            # val_metric.reset_states()
            # print("Validation acc: %.4f" % (float(val_acc),))
            # print("Time taken: %.2fs" % (time.time() - start_time))

    def PlotOutput(self, Model, Dataset):
        Output = []
        TruePlot = []
        for step, (x, y) in enumerate(Dataset):
            x = tf.reshape(x, (1))
            y = tf.reshape(y, (1))
            Output.append(Model(x))
            TruePlot.append(y)
        return Output, TruePlot


# %%
