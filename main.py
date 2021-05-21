import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.ops.gen_batch_ops import batch
from architectures import createAlternativeModel, createInitialModel
from sklearn.model_selection import train_test_split
from IPython.display import clear_output
import time

# Display curves on both accuracy and loss function
#Plot Loss
def display_loss(history):
    plt.figure(figsize=[8,6])
    plt.plot(history.history['loss'],'cyan',linewidth = 3.0)
    plt.plot(history.history['val_loss'],'black',ls='--',linewidth = 3.0)
    plt.legend(['Training Loss' , 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs' ,fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)
    plt.savefig(experiments_path+"/Experiment"+number+"loss.png")
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
    plt.savefig(experiments_path+"/Experiment"+str(number)+"acc.png")
    plt.show()

number = input("Give number of experiment/index of architecture : \n")

# change directory to load fer dataset from pandas
cwd = os.getcwd()
experiments_path = (cwd+'\\Experiments')
cwd = (cwd+'/fer2013/fer2013')
#checkpoint_path = os.getcwd()

#filename = os.path.join(os.path.curdir, 'fer2013', 'fer2013', 'fer2013.csv')
basic_emotions = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]

df = pd.read_csv(cwd+"/fer2013.csv")
images = []
labels = df.emotion.values


for i in range(df.shape[0]):
    image = df['pixels'][i].split()
    image = np.reshape(image, (48, 48, 1))
    images.append(image)
images = np.stack(images, axis=0)
img_array = np.array(images)


# Split training and testing data
# print("X_train shape : \n",x_train.shape),
# print("X_test shape : \n",x_test.shape),
# print("Y_train shape : \n",y_train.shape),
# print("Y_test shape : \n",y_test.shape)


# Convert to float 32 and  Normalizing dataset(Works-->Tested) :
img_array = img_array.astype("float32")
img_array = img_array / 255

# Split train and test dataset :
x_train, x_test, y_train, y_test = train_test_split(img_array, labels, random_state=0 ,test_size=0.1)



#x_train = img_array[:28708]
#x_test = img_array[28709:32299]
#y_train = labels[:28708]
#y_test = labels[28709:32299]


# Initialize validation set :
#x_val = img_array[32300:]
#y_val = labels[32300:]

# Initialize model from architectures file
# model = createAlternativeModel()
model = createInitialModel()
#model.summary()
with open(experiments_path+'/Experiment'+number+'.txt','w') as file:
    model.summary(print_fn=lambda x: file.write(x+"\n"))



#Modifying Optimizer's Learning Rate:
opt = tf.optimizers.RMSprop(learning_rate=0.0001)
#opt = tensorflow.keras.optimizers.Adam(lr=0.01)
model.compile(loss="sparse_categorical_crossentropy",optimizer=opt,metrics=['accuracy'])

# loss="mean_squared_error",optimizer='Adam'

# load obtained weights before overfitting & save specific model
# classifier.load_weights('checkpoint_path')
# classifier.save('shapes_cnn.h5')


model_history = model.fit(x_train,y_train,batch_size=32,epochs=10,validation_split=0.11)  # overfitting on the 10th epoch meaning that it reaches the optimal weights

#finalModel.compile(loss="mean_squared_error",optimizer='Adam',metrics=['accuracy'])
#finalModel.summary()
# #Final Accuracy
scores = model.evaluate(x_test,y_test,batch_size=32)
print("Model Accuracy : %.2f%%"%(scores[1]*100))

display_acc(model_history)
display_loss(model_history)


# Here we test our model and in the range we specify how many images we want to predict
"""
for k in range(0,10):
    labeled = y_test[k]
    print(f' Actual label :',basic_emotions[labeled])
    predicted = model.predict(tf.expand_dims(x_test[k],0)).argmax()
    print(f' Predicted label :',basic_emotions[predicted])
    plt.imshow(x_test[k])
    plt.show()
    time.sleep(5)
    clear_output(wait=True)

"""
