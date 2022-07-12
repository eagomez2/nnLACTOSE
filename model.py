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
x = np.sin(t)
y = np.tanh(4 * x)

############################
# INPUT AND OUTPUT SIGNALS #
############################
plt.figure()
plt.plot(t, x)
plt.plot(t, y)
plt.title("Input and Output Signal")
plt.legend(["Input", "Output"])

#####################
# TRANSFER FUNCTION #
#####################
plt.figure()
plt.plot(x, y)
plt.title("Transfer Function")
plt.axhline(y=0, color="k")
plt.axvline(x=0, color="k")
#%%
###########################################
# CREATE FUNCTIONAL MODEL WITH TENSORFLOW #
###########################################

input = tf.keras.Input(shape=(1,))
dense1 = tf.keras.layers.Dense(64, activation="relu")  # first layer
x = dense1(input)
x = tf.keras.layers.Dense(64, activation="relu")(x)  # second layer
output = tf.keras.layers.Dense(1)(x)  # output layer

model = tf.keras.Model(inputs=input, outputs=output)
model.summary()
tf.keras.utils.plot_model(model, "my_first_model.png")


# %%
