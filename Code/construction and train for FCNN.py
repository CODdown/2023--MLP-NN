import tensorflow as tf
import random
import numpy as np
import os


number_of_features = 15
time_of_repeating_training = 3


# the load/save paths of raw files containing test set or train set
all_path = './all.txt'
train_path = './train.txt'
test_path = './test.txt'

# the save paths of files containing test set or train set in which the data were transformed.
x_train_savepath = './x_train.npy'
y_train_savepath = './y_train.npy'

x_test_savepath = './x_test.npy'
y_test_savepath = './y_test.npy'


# the function to transform the data
def generateds(path):
    f = open(path, 'r')
    content = f.readlines()
    f.close()
    x, y_ = [], []
    for line in content:
        all = line.strip().split("\t")
        length = len(all)
        featurei = all[0:length-1]
        for i in range(0, len(featurei)):
            featurei[i] = float(featurei[i])
        labeli = float(all[-1])
        x.append(featurei)
        y_.append(labeli)

    x = np.array(x)
    y_ = np.array(y_)
    y_ = y_.astype(np.float64)
    return x, y_


# data transform and data save
if os.path.exists(x_train_savepath) and os.path.exists(y_train_savepath) and os.path.exists(
        x_test_savepath) and os.path.exists(y_test_savepath):
    print('-------------Load Datasets-----------------')
    x_train_save = np.load(x_train_savepath)
    y_train = np.load(y_train_savepath)
    x_test_save = np.load(x_test_savepath)
    y_test = np.load(y_test_savepath)

    x_train = np.reshape(x_train_save, (len(x_train_save), number_of_features))
    x_test = np.reshape(x_test_save, (len(x_test_save), number_of_features))
else:
    # ***Important！ The data in all.txt should contain no header.
    print('-------------Generate Datasets-----------------')
    # shuffle samples and divide them into test set(20%) and train set(20%)
    f = open(all_path, 'r')
    content = f.readlines()
    f.close()
    shuffled = []
    for line in content:
        line = line.strip().split("\t")
        shuffled.append(line)
    random.shuffle(shuffled)
    size_of_all = len(shuffled) #  the number of all samples
    size_of_test = round(size_of_all/5) #  the size of test set
    with open(test_path, "w") as test:
        for i in shuffled[0:size_of_test]:
            for j in i:
                test.write(str(j) + "\t")
            test.write("\n")
    test.close()
    with open(train_path, "w") as train:
        for i in shuffled[size_of_test: size_of_all]:
            for j in i:
                train.write(str(j) + "\t")
            train.write("\n")
    train.close()

    x_train, y_train = generateds(train_path)
    x_test, y_test = generateds(test_path)

    print('-------------Save Datasets-----------------')
    x_train_save = np.reshape(x_train, (len(x_train), -1))
    x_test_save = np.reshape(x_test, (len(x_test), -1))
    np.save(x_train_savepath, x_train_save)
    np.save(y_train_savepath, y_train)
    np.save(x_test_savepath, x_test_save)
    np.save(y_test_savepath, y_test)

    print('-------------Load Datasets-----------------')
    x_train_save = np.load(x_train_savepath)
    y_train = np.load(y_train_savepath)
    x_test_save = np.load(x_test_savepath)
    y_test = np.load(y_test_savepath)

    x_train = np.reshape(x_train_save, (len(x_train_save), number_of_features))
    x_test = np.reshape(x_test_save, (len(x_test_save), number_of_features))


# create the file to save models
model_save_dict = "./model"
if not os.path.exists(model_save_dict):
    os.mkdir(model_save_dict)

# training of model
for i in range(0, time_of_repeating_training):
    # construction of model
    model1 = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(number_of_features,)),
        tf.keras.layers.Dense(350, activation='relu'),
        tf.keras.layers.Dense(350, activation='relu'),
        tf.keras.layers.Dense(350, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    # show the structure of the model
    model1.summary()
    model1.compile(optimizer='Adam',
                   loss='mse')

    # save the model
    rep_save_dict = "./model/rep" + str(i + 1)
    if not os.path.exists(rep_save_dict):
        os.mkdir(rep_save_dict)
    h5_save_path = rep_save_dict + "/model_{epoch:03d}-loss_{loss:.4f}-val_loss_{val_loss:.4f}.h5"
    DNN_callback = tf.keras.callbacks.ModelCheckpoint(filepath=h5_save_path,
                                                save_weights_only=True,
                                                verbose=0,
                                                save_best_only=True)
    DNN_earlystop = tf.keras.callbacks.EarlyStopping(patience=250,
                                                     monitor="val_loss",
                                                     mode="min")
    history = model1.fit(x_train, y_train,
                         batch_size=256, epochs=1500, shuffle=True,
                         validation_data=(x_test, y_test), validation_freq=1,
                         callbacks=[DNN_callback, DNN_earlystop])



