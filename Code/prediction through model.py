import tensorflow as tf
import numpy as np
import os

number_of_features = 15

h5_load_path = "./model.h5" #  *** Important! Please change the file name of the model saved for use before.
model1 = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(number_of_features, )),
    tf.keras.layers.Dense(350, activation='relu'), #  *** The parameters should be modified to be same as those used to construct the model.
    tf.keras.layers.Dense(350, activation='relu'), #  *** The parameters should be modified to be same as those used to construct the model.
    tf.keras.layers.Dense(350, activation='relu'), #  *** The parameters should be modified to be same as those used to construct the model.
    tf.keras.layers.Dense(1, activation='linear')
])

if os.path.exists(h5_load_path):
    print('-------------load the model-----------------')
    model1.load_weights(h5_load_path)
    model1.summary()

# load the feature values of the new 5' UTR sequence to predict the relative GFP abundance.
# The rank of the features should be the same as that of train set.
input_path = "./input.txt"
features = []
with open(input_path, "r") as input:
    for line in input:
        line = line.strip().split("\t")
        for i in range(len(line)):
            line[i] = np.float64(line[i])
        features.append(line)
input.close()
features = np.array(features)

# predict
y = model1.predict(features)

# output the prediction
with open("./prediction.txt", "w") as output:
    for i in y:
        for j in i:
            output.write(str(j))
            output.write("\n")
