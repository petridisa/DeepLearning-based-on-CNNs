import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from torch.nn import conv2d

filename = os.path.join(os.path.curdir, 'fer2013', 'fer2013', 'fer2013.csv')
basic_emotions = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]

df = pd.read_csv(filename)
images = []
emotions = df['emotion']
emotions = [basic_emotions[i] for i in emotions]

for i in range(df.shape[0]):
    image = df['pixels'][i].split()
    image = [float(i)/255 for i in image]
    image = np.reshape(image, (48, 48))
    images.append(image)
images = np.stack(images, axis=0)
