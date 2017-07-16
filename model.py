import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from random import shuffle
import sklearn
from keras.optimizers import Adam


#read the csv and get all the samples
samples = []
with open('./training/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


#Split the training and the validation sample
train_samples, validation_samples = train_test_split(samples, test_size=0.2, random_state=42)


"""
This function is to read all the images with steering angles and send them 
to the generator. The images are read from left right and center cameras of
the car. The images are then flipped and for images left and right there is
a correction factor added to it.

"""
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images=[]
            steering=[]
            correction = 0.25

            for row in batch_samples:
            	for index in range(0,3):
            		newpath='./training/IMG/'+row[index].split('/')[-1]
            		getImg=cv2.imread(newpath)
            		readImg=cv2.cvtColor(getImg, cv2.COLOR_BGR2RGB)
            		readSteering = float(row[3])
            		images.append(readImg)
            		images.append(cv2.flip(readImg,1))
            		if index==1:
            			readSteering+=correction
            		elif index ==2:
            			readSteering-=correction
            		steering.append(readSteering)
            		steering.append(readSteering*-1.0)


            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(steering)
            yield sklearn.utils.shuffle(X_train, y_train)



#training samples for the model
train_generator = generator(train_samples, batch_size=32)

#validation sample for the model
validation_generator = generator(validation_samples, batch_size=32)

#Adam optimizer with values initialized
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)


"""
Below is the Nvidia model using to train the car
"""
model = Sequential()
model.add(Lambda(lambda x: (x / 127.5)-1.0, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24,3,125,subsample=(2, 2), border_mode='valid', activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2, 2), border_mode='valid', activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2, 2), border_mode='valid', activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.summary()

#
model.compile(loss='mse',optimizer ='adam')
# model.fit(X_train, y_train, nb_epoch=3, validation_split=0.2, shuffle=True)
# train = rain_gen(X_train, y_train)

#fit generator uses 5 epochs and not more than 5 to avoid overfitting of the moodel.
model.fit_generator(train_generator,samples_per_epoch=20000,nb_val_samples=len(validation_samples),nb_epoch=5,validation_data=validation_generator)
model.save('model.h5')
