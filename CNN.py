import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
T_data = np.load('datasets/Training_data_NSV.npy')
T_label = np.load('datasets/Training_label_NSV.npy')
V_data = np.load('datasets/Testing_data_NSV.npy')
V_label = np.load('datasets/Testing_label_NSV.npy')

'''Set GPU'''
gpus = [0] # Here I set CUDA to only see one GPU
os.environ['CUDA_VISIBLE_DEVICES']=','.join([str(i) for i in gpus])

# Model
# config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 48} ) 
# sess = tf.Session(config=config) 
# keras.backend.tensorflow_backend._get_available_gpus()

model = Sequential()
model.add(keras.layers.Conv2D(128,[8,8],strides=(2,2),padding='same',input_shape=(56,256,256)))
model.add(keras.layers.Conv2D(128,[3,3],strides=(1,1),padding='same'))
# 
model.add(keras.layers.Conv2D(256,[8,8],strides=(2,2),padding='same'))
model.add(keras.layers.Conv2D(256,[3,3],strides=(1,1),padding='same'))
# 
model.add(keras.layers.Conv2D(512,[8,8],strides=(4,4),padding='same'))
model.add(keras.layers.Conv2D(512,[3,3],strides=(3,3),padding='same'))
# 
model.add(keras.layers.Conv2D(512,[8,8],strides=(4,4),padding='same'))
model.add(keras.layers.Conv2D(512,[3,3],strides=(3,3),padding='same'))
# 
model.add(keras.layers.Conv2D(512,[8,8],strides=(4,4),padding='same'))
model.add(keras.layers.Conv2D(512,[3,3],strides=(3,3),padding='same'))
# 
model.add(Flatten())
model.add(keras.layers.Dense(512,activation='relu'))
model.add(keras.layers.Dense(512,activation='relu'))
model.add(keras.layers.Dense(512,activation='relu'))
model.add(keras.layers.Dense(3,activation='sigmoid'))

adam = keras.optimizers.Adam(0.00001)
model.compile(loss='mean_squared_error',optimizer=adam,metrics=['mean_squared_error'])
# Tensorboard
logdir = './logs'
tensorboard_callback = keras.callbacks.TensorBoard(logdir, histogram_freq=1)
Earlystop = keras.callbacks.EarlyStopping(monitor='val_loss',patience=20)
model_save = keras.callbacks.ModelCheckpoint('CNN3_shuffle.h5',monitor='val_loss',save_best_only=True)
# Fit the model
history = model.fit(T_data, T_label, 
                    epochs=300,
                    batch_size=140,
                    verbose=1,
                    validation_data=(V_data,V_label),
                    callbacks=[tensorboard_callback,Earlystop,model_save],
                    shuffle=True)
# model.fit(T_data,T_label,epochs=1000,batch_size=140)
mse=model.evaluate(T_data,T_label)
print(mse)

# evaluation
# Fit the model
# list all data in history
print(history.history.keys())
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('CNN3_validation.png')

# model.save('CNN_ep300.h5')
# # Testing code
# print("*texture:")
# print(Train_data[0][0][0])
# print("*depth:")
# print(Train_data[0][1][0])
# print("*position:")
# print(Train_data[0][2][0])
# print("*orientaion:")
# print(Train_data[0][3][0])
# print("output Training_data.csv")
# 
# # Output .csv
# with open('Training_data.csv', 'w') as myfile:
#     wr = csv.writer(myfile)
#     for i in range(len(Train_data)):
#         wr.writerow(Train_data[i])

# with open('Training_label.csv', 'w') as myfile:
#     wr = csv.writer(myfile)
#     for i in range(len(Train_label)):
#         wr.writerow(Train_label[i])



