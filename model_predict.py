import tensorflow as tf
import keras
import numpy as np
model = keras.models.load_model('CNN3.h5')
T_data = np.load('datasets/T_data_ram_no.npy')
# T_label = np.load('datasets/T_label.npy')

# temp=[]
# temp.append(T_data[5])
# temp.append(T_data[43])
# temp.append(T_data[126])
# Test_data = np.asarray(temp)
model_output = model.predict(T_data)
for i in range(len(model_output)):
    for j in range(3):
        model_output[i][j]=round(model_output[i][j]*7,0)

# print(model.evaluate(T_data,T_label))
# print(model_output.tolist())
np.savetxt("CNN_train3_predict_ran_no.csv", model_output, delimiter=",")
