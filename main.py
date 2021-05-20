import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.ops.gen_batch_ops import batch
from architectures import createInitialModel
from sklearn.model_selection import train_test_split
from IPython.display import clear_output
import time

# Display curves on both accuracy and loss function
#Plot Loss
def display_loss(history):
    plt.figure(figsize=[8,6])
    plt.plot(history.history['loss'],'red',linewidth = 3.0)
    plt.plot(history.history['val_loss'],'black',ls='--',linewidth = 3.0)
    plt.legend(['Training Loss' , 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs' ,fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)
    plt.show()

#Plot Accuracy:
def display_acc(history):
    plt.figure(figsize=[8,6])
    plt.plot(history.history['accuracy'],'coral',linewidth = 3.0)
    plt.plot(history.history['val_accuracy'],'magenta',ls='--',linewidth = 3.0)
    plt.legend(['Training Accuracy' , 'Validation Accuracy'],fontsize=18)
    plt.xlabel('Epochs' ,fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)
    plt.show()



# change directory to load fer dataset from pandas
cwd = os.getcwd()
checkpoint_path = os.chdir('model_checkpoint')
cwd = (cwd+'/fer2013/fer2013')
checkpoint_path = os.getcwd()

#filename = os.path.join(os.path.curdir, 'fer2013', 'fer2013', 'fer2013.csv')
basic_emotions = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]

df = pd.read_csv(cwd+"/fer2013.csv")

images = []
#emotions = df['emotion']
#emotions = [basic_emotions[i] for i in df.emotion.values]
#emotions = np.array(emotions)

labels = df.emotion.values
#labels = emotions
#print(labels.shape)
#print(emotions.shape)

for i in range(df.shape[0]):
    image = df['pixels'][i].split()
    #image = [float(i)/255 for i in image]
    image = np.reshape(image, (48, 48, 1))
    images.append(image)
images = np.stack(images, axis=0)
img_array = np.array(images)


# Split training and testing data
x_train, x_test, y_train, y_test = train_test_split(img_array, labels, random_state=0 ,test_size=0.1)
#print("X_train shape : \n",x_train.shape),#print("X_test shape : \n",x_test.shape),#print("Y_train shape : \n",y_train.shape),#print("Y_test shape : \n",y_test.shape)

# Convert to float 32
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

# Normalizing train/test data(Works-->Tested) :
x_train = x_train / 255
x_test = x_test / 255

# Initialize model from architectures file
model = createInitialModel()

#Modifying Optimizer's Learning Rate:
opt = tf.optimizers.RMSprop(learning_rate=0.001)
#opt = tensorflow.keras.optimizers.Adam(lr=0.01)
model.compile(loss="sparse_categorical_crossentropy",optimizer=opt,metrics=['accuracy'])


# Checkpoint:

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor = 'val_acc',verbose=1, save_best_only=True)
# fit model


# load obtained weights before overfitting & save specific model
# classifier.load_weights('checkpoint_path')
# classifier.save('shapes_cnn.h5')


model_history = model.fit(x_train,y_train,batch_size=32,epochs=10,validation_split=0.2,callbacks=[checkpoint]) # overfitting on the 10th epoch meaning that it reaches the optimal weights

#finalModel.compile(loss="mean_squared_error",optimizer='Adam',metrics=['accuracy'])
#finalModel.summary()

# #Final Accuracy
scores = model.evaluate(x_train,y_train,batch_size=32)
print("Model Accuracy : %.2f%%"%(scores[1]*100))

display_acc(model_history)
display_loss(model_history)

# Here we test our model and in the range we specify how many images we want to predict
for k in range(0,10):
    labeled = y_test[k]
    print(f' Actual label :',basic_emotions[labeled])
    predicted = model.predict(tf.expand_dims(x_test[k],0)).argmax()
    print(f' Predicted label :',basic_emotions[predicted])
    plt.imshow(x_test[k])
    plt.show()
    time.sleep(5)
    clear_output(wait=True)


