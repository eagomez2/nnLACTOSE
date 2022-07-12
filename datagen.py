#%%
import numpy as np
import matplotlib.pyplot as plt


class DataGenerator:
    def __init__(self, Fs, Seconds):
        self.Fs = Fs
        self.t = np.arange(0, Seconds, 1 / Fs)

    def PiecewiseFunction(self, x, CoeffsArray, CondsArray):  # max of 8 coeffs
        A = CoeffsArray[0]
        B = CoeffsArray[1]
        C = CoeffsArray[2]
        D = CoeffsArray[3]
        E = CoeffsArray[4]
        F = CoeffsArray[5]
        G = CoeffsArray[6]
        H = CoeffsArray[7]
        #########################
        # PREPROCESS CONDSARRAY #
        #########################
        ConditionsArray = np.sort(CondsArray)
        self.Conditions = [
            (ConditionsArray[0] <= x) & (x < ConditionsArray[1]),
            (ConditionsArray[1] <= x) & (x < ConditionsArray[2]),
            (ConditionsArray[2] <= x) & (x < ConditionsArray[3]),
            (ConditionsArray[3] <= x) & (x <= ConditionsArray[4]),
        ]
        self.FunctionsList = [
            lambda x: A * x + B,
            lambda x: C * x + D,
            lambda x: E * x + F,
            lambda x: G * x + H,
        ]
        output = np.piecewise(x, self.Conditions, self.FunctionsList)
        return output

    def GenerateData(self):
        self.Input = np.sin(2 * 100 * np.pi * self.t) + np.sin(2 * np.pi * 130 * self.t)
        self.Input = self.Input / np.max(np.abs(self.Input))
        plt.plot(self.t, self.Input)

        Conditions = np.random.rand(3) * 2 - 1
        Conditions = np.append(Conditions, [1, -1])
        Coefficients = np.random.randint(1, 9, size=(8)) - 5
        print(Coefficients)
        Output = self.PiecewiseFunction(
            self.Input,
            Coefficients,
            Conditions,
        )
        self.Output = Output / np.max(np.abs(Output))
        plt.plot(self.t, self.Output)
        plt.title("Input and Output Signal")
        return self.Input, self.Output, self.t

    def PlotTransferFunction(self):
        plt.figure()
        plt.plot(self.Input, self.Output)
        plt.title("Transfer Function")
        plt.axhline(y=0, color="k")
        plt.axvline(x=0, color="k")


MyDataGenerator = DataGenerator(48000, 0.25)
Input, Output, t = MyDataGenerator.GenerateData()
MyDataGenerator.PlotTransferFunction()


#%%
#############
# PLOT DATA #
#############

plt.figure()
plt.plot(t, Input)
plt.plot(t, Output)
plt.figure()
plt.plot(Input, Output)

# %%
