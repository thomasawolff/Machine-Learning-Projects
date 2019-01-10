import os
import sys
import numpy as np
import pandas as pd
import pprint as p
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

os.chdir('C:\\\\Users\\\\moose\\\\OneDrive\\\\Code\\\\Tensorflow Course\\\\Tensorflow-Bootcamp-master\\\\02-TensorFlow-Basics')
homePrice = pd.read_csv('cal_housing_clean.csv')
##print (homePrice.describe())

labels = homePrice['medianHouseValue']
x_data = homePrice.drop('medianHouseValue',axis=1)

x_train,x_test,y_train,y_test = train_test_split(x_data,labels,test_size=0.35,random_state=101)

##scaler = MinMaxScaler()
##scaler.fit(x_train)

x_train = pd.DataFrame(x_train,columns=x_train.columns,index=x_train.index)
x_test = pd.DataFrame(x_test,columns=x_test.columns,index=x_test.index)

medianAge = tf.feature_column.numeric_column('housingMedianAge')
rooms = tf.feature_column.numeric_column('totalRooms')
totalBeds = tf.feature_column.numeric_column('totalBedrooms')
population = tf.feature_column.numeric_column('population')
households = tf.feature_column.numeric_column('households')
medianIncome = tf.feature_column.numeric_column('medianIncome')

feat_cols = [medianAge,rooms,totalBeds,population,households,medianIncome]

# Training the model ####
input_func = tf.estimator.inputs.pandas_input_fn(x=x_train,y=y_train,\
                                                 batch_size=100,num_epochs=10000,\
                                                 shuffle=True)

model = tf.estimator.DNNRegressor(hidden_units=[6,6,6,6],feature_columns=feat_cols)
#model = tf.estimator.LinearRegressor(feature_columns=feat_cols)
model.train(input_fn=input_func,steps=100000)

# Generate predictions #### 
predict_input_func = tf.estimator.inputs.pandas_input_fn(x=x_test,\
                                                         batch_size=100,\
                                                         num_epochs=1,
                                                         shuffle=False)

pred_gen = model.predict(predict_input_func)
#for line in pred_gen:
#    print (line.values())

final_preds = []
for pred in pred_gen:
    #print(pred)
    final_preds.append(pred['predictions'])

newX = []
for i in range(0,len(final_preds)):
    newX.append(i)

plt.scatter(newX,final_preds)
plt.show()

p.pprint(np.mean(final_preds))
##    
##error = mean_squared_error(y_test,final_preds)**0.5
##print (error)
