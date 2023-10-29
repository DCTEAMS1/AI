import pandas as pd
import random
import datetime
import numpy as np
#from keras.models import Sequential
#from keras.layers import Dense
import tensorflow as tf
def FromNameToNumber(array):
    tmp = {}
    i = 1
    for item in array:
        tmp[item] = i
        i += 1
    return tmp
def extract_month(date_time):
    return int(date_time[5:7])
def AllColumnsToNumber(data):
    cols = data.columns.tolist()
    for col in cols:
        if col == "Дата и время ":
            data['Дата и время '] = data['Дата и время '].apply(extract_month)
        else:
            data[col] = data[col].map(FromNameToNumber(data[col].unique()))

data = pd.read_csv("ДТП.csv", delimiter=',')
AllColumnsToNumber(data)
print(data.head())
data = data.sample(frac=1)
print(data.head())
#dataset = data.apply(lambda row: np.array(row), axis=1).values
#dataset_tensor = tf.constant(dataset, dtype=tf.dtypes.int32)
#dataset = tf.data.Dataset.from_tensor_slices(dataset)
#data = data.shuffle(buffer_size=16,reshuffle_each_iteration=True)
#data = data.batch(16)
#data = data.prefetch(8)
#print(data)
#train = data.take(int((len(data)*.7)//1))
#test = data.skip(int((len(data)*.7)//1)).take(int((len(data)*.3)//1))
#samples, labels = train.as_numpy_iterator().next()