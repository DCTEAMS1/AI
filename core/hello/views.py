from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import datetime
import numpy as np
import keras.models
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import tensorflow as tf


def extract_month(date_time):
    return int(date_time[5:7])
def AllColumnsToNumber(data):
    cols = data.columns.tolist()
    for col in cols:
        if col == "Местное время":
            data['Местное время'] = data['Местное время'].apply(extract_month)
        else:
            #print(FromNameToNumber(data[col].unique()))
            #if col == 'meteostation':

            data[col] = data[col].map(FromNameToNumber(data[col].unique()))

def FromNameToNumber(array):
    tmp = {}
    i = 0
    for item in array:
        tmp[item] = i
        i += 1
    return tmp
def load_neuro(name, data):
    model = keras.models.load_model(name)
    data = data.reshape(-1, 6, 1)
    print(data)
    return model.predict(data)
    #return random.randint(1,100) / 100

def Input_data():
    case = input('Выберите кейс прогнозирования: 1. Пожар\n')
    date = input('Введите время прогнозирования в формате гггг-мм-дд: ')
    data = pd.read_csv("reswithnonull.csv", delimiter=',')
    slov = FromNameToNumber(data['meteostation'].unique())
    met = input('Введите город: ')
    T = float(input('Введите T: '))
    Po = float(input('Введите Po: '))
    FF = float(input('Введите FF: '))
    features = [float(date[5:7]), float(slov[met]), T, Po, FF]
    return (case,tf.data.Dataset.from_tensor_slices(features ))
def stand():
    data = pd.read_csv("reswithnonull.csv", delimiter=',')
    AllColumnsToNumber(data)

    X = data.drop(columns=['Тип выезда'])
    X = X.values

    # Разделите данные на обучающий и тестовый наборы
    split_ratio = 0.8  # Процент данных для обучения
    split_index = int(len(X) * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]

    # Масштабируйте признаки (стандартизация)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    return (mean, std)

def Input_data2(st):
    
    features = st.split(',')
    features[0] = float(features[0][5:7])
    data = pd.read_csv("C:\\Users\yfvfy\Downloads\django\core\hello\LessFireResWithNoNull.csv", delimiter=',')
    slov = FromNameToNumber(data['meteostation'].unique())
    slov2 = FromNameToNumber(data['W1'].unique())
    print(slov2)
    features[1] = float(slov[features[1]])
    features[2] = float(features[2])
    features[3] = float(features[3])
    features[4] = float(features[4])
    features[5] = float(slov2[features[5]])
    #features = pd.DataFrame([features]*5, columns = ['Местное время','meteostation','T','Po','Ff'])
    features[0] =(features[0]-1)/11
    features[1] =(features[1]-0)/23
    features[2] =(features[2]+39.42)/69.32
    features[3] =(features[3]-694.65)/87.1875
    features[4] =(features[4]-0)/10
    features[5] =(features[5]-0)/10
    #print(st)
    input_data = np.array(features)
    #print(input_data)
    return ('1',input_data)
def Input_data3():
    data = pd.read_csv("test.csv", delimiter=',')

def Output_data(st):
    input = Input_data2(st)
    if input[0] == '1':
        return(1-load_neuro('F:\modelLSTMFireTest',input[1])[0][0])
    else:
        print('неверный кейс!')

 


def index(request):
    # получаем из данных запроса POST отправленные через форму данные
   
    age = {"probabilityFire": " "}
    return render(request,"index.html",context=age)

def ret(request,d):
    # получаем из данных запроса POST отправленные через форму данные
   
    age = {"probabilityFire": Output_data(d)}
    return render(request,"index.html",context=age)


def  postuser(request):
    # получаем из данных запроса POST отправленные через форму данные
    TypeIncident = selected_option = request.POST.get('incident')
    
    data =request.POST.get("data")
    windSpeed = request.POST.get("windSpeed")
    airTemperature = request.POST.get("airTemperature")
    Humidity= request.POST.get("Humidity")
    weatherStation = selected_option = request.POST.get('weatherStation') 
    typePrecipitation = selected_option = request.POST.get('typePrecipitation')
    if (typePrecipitation == "Облака покрывали более половины неба в течение одной части соответствующего периода"):
            typePrecipitation = "Облака покрывали более половины неба в течение одной части соответствующего периода и половину или менее в течение другой части периода."
    d = data+"," + weatherStation+"," + airTemperature+"," + Humidity+"," + windSpeed+"," + typePrecipitation
    age = {"probabilityFire": Output_data(d)}
    return render(request,"index.html",context=age)
   

