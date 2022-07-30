#%%
import nnLACTOSE as nnLACTOSE

LayerInfoDict = {
    "0": {"size": (1, 2), "type": "input"},
    "1": {"size": 8, "type": "dense"},
    "2": {"size": 8, "type": "dense"},
    "3": {"size": 8, "type": "dense"},
    "4": {"size": 1, "type": "dense"},
}
ConditionArray = [-1, -0.6, -0.2, 0.2, 0.5, 0.8, 1.0]
Model = nnLACTOSE.LactoseModel(LayerInfoDict, ConditionArray, DisplayPlot=False)
# %%
