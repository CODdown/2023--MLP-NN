import tensorflow as tf
import numpy as np
import os
from scipy.stats import pearsonr

n_features = 15
all_path = ".\\train.txt"

#  to perform "n"-fold cross-validation
#  This could be changed as needed.
n_fold = 5

# the function of list split
def split(lst, grp):
    cnt, re = divmod(len(lst), grp)
    tem_lst = [cnt] * grp
    for i in range(re):
         tem_lst[i] += 1
    result = []
    ori = 0
    for j in tem_lst:
        result.append(lst[ori: ori+j])
        ori += j
    return result

# training will be stopped early if the loss of training set is lower than 50.
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get("loss") < 50):
            print("\n-------------Loss is low so cancelling training!-------------")
            self.model.stop_training = True

#  the function of split and transformation of features and corresponding labels
def generateds(path):
    f = open(path, 'r')
    contents = f.readlines()
    f.close()
    x, y_ = [], []
    for content in contents[1:]:
        all = content.strip().split("\t")
        length = len(all)
        featurei = all[1:length-2]
        for i in range(0, len(featurei)):
            featurei[i] = float(featurei[i])
        labeli = float(all[-1])
        x.append(featurei)
        y_.append(labeli)
    x = np.array(x)
    y_ = np.array(y_)
    y_ = y_.astype(np.float64)
    return x, y_

#  load the train set and make a split.
all = []
f = open(all_path, "r")
content = f.readlines()
f.close()
for line in content:
    line = line.strip().split("\t")
    all.append(line)
train_split = split(all, n_fold)

with open(".\\result of cross-validation.txt", "w") as output:
    output.write("setup\tR2_train\tR2_validation\n")
    #  n_layer and n_units are the hyper-parameters which are being optimized.
    #  These two could be changed as needed.
    for n_layer in [2,3,4]:
        for n_units in [100,200,300,400,500]:
            for i in range(0, n_fold):
                # the save/load path of train subsets and validation sets
                train_path = ".\\split\\" + str(i+1) + "_train.txt"
                validation_path = ".\\split\\" + str(i+1) + "_validation.txt"
                x_train_savepath = ".\\split\\" + str(i+1) + "_x_train.npy"
                y_train_savepath = ".\\split\\" + str(i+1) + "_y_train.npy"
                x_validation_savepath = ".\\split\\" + str(i+1) + "_x_validation.npy"
                y_validation_savepath = ".\\split\\" + str(i+1) + "_y_validation.npy"
                if not os.path.exists(".\\split\\"):
                    os.mkdir(".\\split\\")
                if os.path.exists(x_train_savepath) and os.path.exists(y_train_savepath) and os.path.exists(
                        x_validation_savepath) and os.path.exists(y_validation_savepath):
                    print('-------------Load Datasets-----------------')
                    x_train_save = np.load(x_train_savepath)
                    y_train = np.load(y_train_savepath)
                    x_validation_save = np.load(x_validation_savepath)
                    y_validation = np.load(y_validation_savepath)
                    x_train = np.reshape(x_train_save, (len(x_train_save), n_features))
                    x_validation = np.reshape(x_validation_save, (len(x_validation_save), n_features))
                else:
                    print('-------------Generate Datasets-----------------')
                    validation = train_split[i]
                    train = []
                    temp = train_split.copy()
                    del temp[i]
                    for c in temp:
                        train.extend(c)
                    print(train)
                    with open(train_path, "w") as train_ouput:
                        for x in train:
                            for y in x:
                                train_ouput.write(str(y) + "\t")
                            train_ouput.write("\n")
                    train_ouput.close()
                    with open(validation_path, "w") as validation_output:
                        for x in validation:
                            for y in x:
                                validation_output.write(str(y) + "\t")
                            validation_output.write("\n")
                    validation_output.close()
                    x_train, y_train = generateds(train_path)
                    x_validation, y_validation = generateds(validation_path)
                    print('-------------Save Datasets-----------------')
                    x_train_save = np.reshape(x_train, (len(x_train), -1))
                    x_validation_save = np.reshape(x_validation, (len(x_validation), -1))
                    np.save(x_train_savepath, x_train_save)
                    np.save(y_train_savepath, y_train)
                    np.save(x_validation_savepath, x_validation_save)
                    np.save(y_validation_savepath, y_validation)


                # training
                if n_layer == 1:
                    model1 = tf.keras.models.Sequential([
                        tf.keras.layers.Flatten(input_shape=(n_features, )),
                        tf.keras.layers.Dense(n_units, activation='relu'),
                        tf.keras.layers.Dense(1, activation='linear')
                    ])
                if n_layer == 2:
                    model1 = tf.keras.models.Sequential([
                        tf.keras.layers.Flatten(input_shape=(n_features,)),
                        tf.keras.layers.Dense(n_units, activation='relu'),
                        tf.keras.layers.Dense(n_units, activation='relu'),
                        tf.keras.layers.Dense(1, activation='linear')
                    ])
                if n_layer == 3:
                    model1 = tf.keras.models.Sequential([
                        tf.keras.layers.Flatten(input_shape=(n_features,)),
                        tf.keras.layers.Dense(n_units, activation='relu'),
                        tf.keras.layers.Dense(n_units, activation='relu'),
                        tf.keras.layers.Dense(n_units, activation='relu'),
                        tf.keras.layers.Dense(1, activation='linear')
                    ])
                if n_layer == 4:
                    model1 = tf.keras.models.Sequential([
                        tf.keras.layers.Flatten(input_shape=(n_features,)),
                        tf.keras.layers.Dense(n_units, activation='relu'),
                        tf.keras.layers.Dense(n_units, activation='relu'),
                        tf.keras.layers.Dense(n_units, activation='relu'),
                        tf.keras.layers.Dense(n_units, activation='relu'),
                        tf.keras.layers.Dense(1, activation='linear')
                    ])
                if n_layer == 5:
                    model1 = tf.keras.models.Sequential([
                        tf.keras.layers.Flatten(input_shape=(n_features,)),
                        tf.keras.layers.Dense(n_units, activation='relu'),
                        tf.keras.layers.Dense(n_units, activation='relu'),
                        tf.keras.layers.Dense(n_units, activation='relu'),
                        tf.keras.layers.Dense(n_units, activation='relu'),
                        tf.keras.layers.Dense(n_units, activation='relu'),
                        tf.keras.layers.Dense(1, activation='linear')
                    ])
                if n_layer == 6:
                    model1 = tf.keras.models.Sequential([
                        tf.keras.layers.Flatten(input_shape=(n_features,)),
                        tf.keras.layers.Dense(n_units, activation='relu'),
                        tf.keras.layers.Dense(n_units, activation='relu'),
                        tf.keras.layers.Dense(n_units, activation='relu'),
                        tf.keras.layers.Dense(n_units, activation='relu'),
                        tf.keras.layers.Dense(n_units, activation='relu'),
                        tf.keras.layers.Dense(n_units, activation='relu'),
                        tf.keras.layers.Dense(1, activation='linear')
                    ])
                model1.summary()
                model1.compile(optimizer='Adam',
                               loss='mse')
                DNN_earlystop = tf.keras.callbacks.EarlyStopping(patience=150,
                                                                 monitor="val_loss",
                                                                 mode="min")
                stopbytrainset = myCallback()
                history = model1.fit(x_train, y_train,
                                     batch_size=256, epochs=1000, shuffle=True,
                                     validation_data=(x_validation, y_validation), validation_freq=1,
                                     callbacks=[stopbytrainset])

                # prediction of validation set and train subset
                y_predict_validation = model1.predict(x_validation)
                y_predict_train = model1.predict(x_train)
                # R2 were calculated.
                R2_validation = np.square(pearsonr(y_validation, y_predict_validation)[0][0])
                R2_train = np.square(pearsonr(y_train, y_predict_train)[0][0])
                print(pearsonr(y_train, y_predict_train)[0][0])
                print(R2_train, R2_validation)
                output.write(str(n_layer) + " layer(s)_" + str(n_units) + " units_" + str(i+1) + " validation\t" + str(R2_train) + "\t" + str(R2_validation) + "\n")




