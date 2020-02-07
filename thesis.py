import glob
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import tensorflow as tf
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,
                                     Dropout, Flatten, GlobalAveragePooling2D,
                                     Input, MaxPool2D)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import (array_to_img, img_to_array,
                                                  load_img)
from tensorflow.keras.utils import to_categorical

#####################################################################################
# PARAMETERS
#####################################################################################

## [ DATASET ] ##
IMG_DIRECTORY = '/data/'
IMG_DIRECTORY_TEST = '/test/'
NUMBER_OF_CLASSES = 93

## [ IMAGES ]
IMG_HEIGHT = 750
IMG_WIDTH = 250
COLOR = 3 

# [ K.FOLD ] #
K_FOLDS = 5
SEED = 42

# [MODEL ] # 
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 0.01 

#####################################################################################
# DATASET
#####################################################################################

#Get List of Files as Filepaths
def get_file_list(img_dir):
    if tf.io.gfile.exists(img_dir) == False:
        return None
    img_files = tf.io.gfile.listdir(img_dir)
    file_list = []
    for file in img_files:
        file = img_dir + file
        file_list.append(file)
    return file_list

#Get Labels
def get_labels(img_dir):
    images = tf.io.gfile.listdir(img_dir)
    num_of_images = len(images)
    label_list = []
    for i in range(0, num_of_images):
        suffix = int(images[i][:4])-1000
        label_list.append(suffix)
    return label_list

#Get Image Array 
def get_images(img_dir):
    img_files = get_file_list(img_dir)
    num_of_images = len(img_files)
    images = np.zeros((num_of_images, IMG_HEIGHT, IMG_WIDTH, COLOR), dtype=np.float32)
    for i in range(0, num_of_images):
        image = load_img(img_files[i], target_size=(IMG_HEIGHT, IMG_WIDTH ))
        image = img_to_array(image)
        image /= 255
        images[i, :, : , : ] = image
    return images

#####################################################################################
# MODEL
#####################################################################################
#Return compiled Model
def get_model():
    input_tensor = Input(shape=(IMG_HEIGHT, IMG_WIDTH, COLOR))
    base_model = keras.applications.xception.Xception(include_top=False, weights='imagenet', input_tensor=input_tensor)
    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    predictions = Dense(NUMBER_OF_CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    sgd = SGD(lr=LEARNING_RATE, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


#Print History Object
def plot_history(history):
 # summarize accuracy for loss
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper left')
    plt.show()
 # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper left')
    plt.show()


#####################################################################################
# EVALUATE MODEL
#####################################################################################
#k-fold crossvalidation
def get_k_fold_score():    
    X = get_images(IMG_DIRECTORY)
    Y = np.array(get_labels(IMG_DIRECTORY))
    scores = []
    folds = list(StratifiedKFold(n_splits=K_FOLDS, random_state=42, shuffle=True).split(X,Y))
    for j, (train_index, val_index) in enumerate(folds):
        tf.keras.backend.clear_session()
        X_train = X[train_index]
        y_train = Y[train_index]
        X_valid = X[val_index]
        y_valid= Y[val_index]
        y_train = to_categorical(y_train, num_classes=NUMBER_OF_CLASSES)
        y_valid = to_categorical(y_valid, num_classes=NUMBER_OF_CLASSES)
        model = get_model()
        print('\n Train and Evaluate Fold ',j+1)
        model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_valid, y_valid))
        score = model.evaluate(X_valid, y_valid)
        print(score)
        scores.append(score[1])
        del model 
    scores = np.mean(scores)
    return scores

#predict test images
def train_model_and_predict():
    X = get_images(IMG_DIRECTORY)
    Y = get_labels(IMG_DIRECTORY)
    model = get_model()
    X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    print(y_train)
    print(y_valid)
    y_train = to_categorical(y_train, num_classes=NUMBER_OF_CLASSES)
    y_test = to_categorical(y_valid, num_classes=NUMBER_OF_CLASSES)
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_valid,y_test), shuffle=True)
    eval_score = model.evaluate(X_valid,y_test)
    print(eval_score)
    test_images = get_images(IMG_DIRECTORY_TEST)
    test_labels = get_labels(IMG_DIRECTORY_TEST)
    labels_length = len(test_labels)
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=-1)
    correctly_classified = 0 
    for i in range(0, labels_length):
        if(test_labels[i]==predicted_labels[i]):
            print(test_labels[i])
            correctly_classified += 1
    print('Predicted ' + str(correctly_classified) + ' Images out of ' + str(labels_length) + ' correctly.' ) 
    return history

################################################################
# METHODS
################################################################

#Define Model in get_model()
model = get_model()
model.summary()

#k-fold cross validation
score = get_k_fold_score()
print(score)

#prediction on testset
history = train_model_and_predict()
plot_history(history)
