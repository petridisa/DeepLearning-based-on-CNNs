import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from numpy.lib.npyio import save
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.metrics import Precision, Recall
from tensorflow.python.ops.gen_batch_ops import batch
from architectures import createAlternativeModel, createDoubleConvolutionArc, createInitialModel, createFixedArch
from sklearn.model_selection import train_test_split
from IPython.display import clear_output
from sklearn.metrics import classification_report
import cv2 as cv
from tensorflow.keras import callbacks, models
import tensorflow.keras.metrics as metr
from tensorflow.keras import backend as K
from sklearn.metrics import classification_report
import time


# Display curves on both accuracy and loss function
# Plot Loss
def display_loss(history):
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['loss'], 'cyan', linewidth=3.0)
    plt.plot(history.history['val_loss'], 'black', ls='--', linewidth=3.0)
    plt.legend(['Training Loss', 'Validation Loss'], fontsize=18)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)
    plt.savefig(experiments_path + "/Experiment" + number + "loss.png")
    plt.show()


# Plot Accuracy:
def display_acc(history):
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['accuracy'], 'coral', linewidth=3.0)
    plt.plot(history.history['val_accuracy'], 'magenta', ls='--', linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)
    plt.savefig(experiments_path + "/Experiment" + str(number) + "acc.png")
    plt.show()


number = input("Give number of experiment/index of architecture : \n")

# change directory to load fer dataset from pandas
cwd = os.getcwd()
experiments_path = (cwd + '\\Experiments')
checkpoint_path = (cwd + '\\model_checkpoint')

cwd = (cwd + '/fer2013/fer2013')
# checkpoint_path = os.getcwd()

# filename = os.path.join(os.path.curdir, 'fer2013', 'fer2013', 'fer2013.csv')
basic_emotions = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]

df = pd.read_csv(cwd + "/fer2013.csv")
images = []
labels = df.emotion.values

for i in range(df.shape[0]):
    image = df['pixels'][i].split()
    image = np.reshape(image, (48, 48, 1))
    image = image.astype("float32")
    image = image / 255
    images.append(image)
images = np.stack(images, axis=0)
img_array = np.array(images)

# Convert to float 32 and  Normalizing dataset(Works-->Tested) :
# img_array = img_array.astype("float32")
# img_array = img_array / 255


# Split train and test dataset :
x_train, x_test, y_train, y_test = train_test_split(img_array, labels, test_size=0.1)

# Initialize model from architectures file
model = createFixedArch()
# model = createInitialModel()
# model.summary()
with open(experiments_path + '/Experiment' + number + '.txt', 'w') as file:
    model.summary(print_fn=lambda x: file.write(x + "\n"))

# Callback checkpoint
file_name = 'best_model.h5'
callback_check = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy', verbose=1,
                                                    save_freq='epoch', save_best_only=True, save_weights_only=False,
                                                    mode='max')

# opt = tf.keras.optimizers.RMSprop(lr=0.0001)

# Model compilation : a) loss function, b)optimizer, c) metrics
model.compile(loss="sparse_categorical_crossentropy", optimizer='Adam', metrics=['accuracy'])

# loss="mean_squared_error",optimizer='Adam'

# load obtained weights before overfitting & save specific model
# classifier.load_weights('checkpoint_path')
# classifier.save('shapes_cnn.h5')

model_history = model.fit(x_train, y_train, batch_size=32, epochs=12, callbacks=callback_check,
                          validation_split=0.2)  # overfitting on the 10th epoch meaning that it reaches the optimal weights

# finalModel.compile(loss="mean_squared_error",optimizer='Adam',metrics=['accuracy'])
# finalModel.summary()

display_acc(model_history)
display_loss(model_history)

# #Final Evaluation
loss, accuracy = model.evaluate(x_test, y_test, batch_size=32)
print("Model Loss : ", (loss))
print("Model Accuracy : %.2f%%" % (accuracy * 100))

# Prints classification report including recall/precision/f1_score
y_pred = model.predict(x_test, batch_size=32, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_bool))

# Plot the filter representation
filters, biases = model.layers[1].get_weights()
# normalize filter values to 0-1 so we can visualize them
print(filters.shape)
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

print(filters.shape)
# plot first few filters, just in a 5x5 so we don't have to plot the all
fig_1 = plt.figure(figsize=(8, 12))
c = 5
r = 5
n_filters = c * r

for i in range(1, n_filters + 1):
    f = filters[:, :, :, i - 1]
    fig_1 = plt.subplot(r, c, i)
    fig_1.set_xticks([])
    fig_1.set_yticks([])
    plt.imshow(f[:, :, 0], cmap='gray')
plt.show()

# Here we test our model and in the range we specify how many images we want to predict

for k in range(0, 10):
    labeled = y_test[k]
    print(f' Actual label :', basic_emotions[labeled])
    predicted = model.predict(tf.expand_dims(x_test[k], 0)).argmax()
    print(f' Predicted label :', basic_emotions[predicted])
    plt.imshow(x_test[k])
    plt.show()
    time.sleep(5)
    clear_output(wait=True)


