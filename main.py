import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.ops.gen_batch_ops import batch
from architectures import createInitialModel
from sklearn.model_selection import train_test_split

# change directory to load fer dataset from pandas
cwd = os.getcwd()
os.chdir('DeepLearning-based-on-CNNs/fer2013/fer2013')
filename = os.getcwd()

#filename = os.path.join(os.path.curdir, 'fer2013', 'fer2013', 'fer2013.csv')
basic_emotions = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]

df = pd.read_csv(filename+"/fer2013.csv")

images = []
emotions = df['emotion']
emotions = [basic_emotions[i] for i in emotions]
emotions = np.array(emotions)

labels = df.emotion.values
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
#print("X_train shape : \n",x_train.shape)
#print("X_test shape : \n",x_test.shape)
#print("Y_train shape : \n",y_train.shape)
#print("Y_test shape : \n",y_test.shape)

# Convert to float 32
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

# Normalizing train/test data(Works-->Tested) :
x_train = x_train / 255
x_test = x_test / 255

# Initialize model from architectures file
model = createInitialModel()

#Modifying Optimizer's Learning Rate:
opt = tf.optimizers.RMSprop(learning_rate=0.0001)
#opt = tensorflow.keras.optimizers.Adam(lr=0.01)
model.compile(loss="sparse_categorical_crossentropy",optimizer=opt,metrics=['accuracy'])
model_history = model.fit(x_train,y_train,batch_size=32,epochs=20,validation_split=0.1)

#finalModel.compile(loss="mean_squared_error",optimizer='Adam',metrics=['accuracy'])
#finalModel.summary()


# #Final Accuracy
scores = model.evaluate(x_train,y_train,batch_size=32)
print("Model Accuracy : %.2f%%"%(scores[1]*100))



#print(img_array.shape)

