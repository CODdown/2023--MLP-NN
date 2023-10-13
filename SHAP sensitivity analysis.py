# name ZHS
# class Pass
# time 2022/9/16/0016 11:03
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras import Sequential
import os
import shap
import pandas as pd

feature_number = 15

h5_load_path = "./model.h5" #  *** Important! Please change the file name of the model saved for use before.
model1 = Sequential([
    Flatten(input_shape=(feature_number,)),
    Dense(350, activation='relu'), #  *** The parameters should be modified to be same as those used to construct the model.
    Dense(350, activation='relu'), #  *** The parameters should be modified to be same as those used to construct the model.
    Dense(350, activation='relu'), #  *** The parameters should be modified to be same as those used to construct the model.
    Dense(1, activation='linear')
])

if os.path.exists(h5_load_path):
    print('-------------load the model-----------------')
    model1.load_weights(h5_load_path)
    model1.summary()

# the load path of train set
train_path = "./input.txt"

# the load of train set data.
# *** important! the input file should contain the header (the names of features) of data to facilitate the plot,
# but should exclude the relative GFP abundance.
X = pd.read_csv(train_path, header=0, index_col=False, sep="\t")
X = X.dropna(axis=1, how="all")
X1 = np.array(X)
background = X1

# calculate the shap values of features for each sample.
explainer = shap.DeepExplainer(model1, data=background)
shap_values = explainer.shap_values(background)

# plot
shap.summary_plot(shap_values[0], background, feature_names=X.columns.values)
shap.summary_plot(shap_values[0], background, feature_names=X.columns.values, plot_type="bar")
shap.dependence_plot(0, shap_values[0], X, interaction_index=None)
shap.dependence_plot(1, shap_values[0], X, interaction_index=None)
shap.dependence_plot(2, shap_values[0], X, interaction_index=None)
shap.dependence_plot(3, shap_values[0], X, interaction_index=None)
shap.dependence_plot(4, shap_values[0], X, interaction_index=None)
shap.dependence_plot(5, shap_values[0], X, interaction_index=None)
shap.dependence_plot(6, shap_values[0], X, interaction_index=None)
shap.dependence_plot(7, shap_values[0], X, interaction_index=None)
shap.dependence_plot(8, shap_values[0], X, interaction_index=None)
shap.dependence_plot(9, shap_values[0], X, interaction_index=None)
shap.dependence_plot(10, shap_values[0], X, interaction_index=None)
shap.dependence_plot(11, shap_values[0], X, interaction_index=None)
shap.dependence_plot(12, shap_values[0], X, interaction_index=None)
shap.dependence_plot(13, shap_values[0], X, interaction_index=None)
shap.dependence_plot(14, shap_values[0], X, interaction_index=None)
shap.dependence_plot(15, shap_values[0], X, interaction_index=None)









