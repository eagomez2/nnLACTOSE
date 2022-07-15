#%%
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

######################################
# GENERATE DATA AND PROCESS 1D TO 1D #
######################################

Fs = 1000
N = 1000
t = np.linspace(0, 10, Fs)
x_train = np.sin(t)
y_train = np.tanh(40 * x_train)

############################
# INPUT AND OUTPUT SIGNALS #
############################
plt.figure()
plt.plot(t, x_train)
plt.plot(t, y_train)
plt.title("Input and Output Signal")
plt.legend(["Input", "Output"])
plt.show()

#####################
# TRANSFER FUNCTION #
#####################
plt.figure()
plt.plot(x_train, y_train)
plt.title("Transfer Function")
plt.axhline(y=0, color="k")
plt.axvline(x=0, color="k")
plt.show()
###########################################
# CREATE FUNCTIONAL MODEL WITH TENSORFLOW #
###########################################

input = tf.keras.Input(shape=(1,), name="Input")
dense1 = tf.keras.layers.Dense(64, activation="relu", name="Dense1")  # first layer
x = dense1(input)
x = tf.keras.layers.Dense(64, activation="relu", name="Dense2")(x)  # second layer
output = tf.keras.layers.Dense(1, name="Output")(x)  # output layer

model = tf.keras.Model(inputs=input, outputs=output)
model.summary()
tf.keras.utils.plot_model(model, "./images/default_dense.png")

model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

history = model.fit(x_train, y_train, batch_size=1, epochs=10)
model_out = model.predict(x_train)
plt.plot(t, x_train)
plt.plot(t, y_train)
plt.plot(t, model_out)


# %%
