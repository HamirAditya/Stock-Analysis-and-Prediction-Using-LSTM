import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
import math
from sklearn.preprocessing import MinMaxScaler
from datetime import date
import streamlit as st

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

colT1,colT2 = st.columns([4,12])
with colT2:st.title('Stock Forecast App')
stocks = ('ONGC.NS','NTPC.NS','HDFCLIFE.NS','BAJAJ-AUTO.NS','RELIANCE.NS','COALINDIA.NS','LT.NS','BRITANNIA.NS','BHARTIARTL.NS'
          ,'KOTAKBANK.NS','ONGC.NS','ICICIBANK.NS','CIPLA.NS','TATACONSUM.NS','MARUTI.NS','ITC.NS','TECHM.NS','INDUSINDBK.NS','TCS.NS',
          'TITAN.NS','ULTRACEMCO.NS','HEROMOTOCO.NS','APOLLOHOSP.NS','WIPRO.NS','NESTLEIND.NS','BAJAJFINSV.NS','WIPRO.NS','NESTLEIND.NS',
          'APOLLOHOSP.NS','HINDALCO.NS','SHREECEM.NS','BAJFINANCE.NS','MM.NS')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

@st.cache
def load_data(ticker):
    df = yf.download(ticker, START, TODAY)
    df.reset_index(inplace=True)
    return df

data_load_state = st.text('Loading data...')
df = load_data(selected_stock)
data_load_state.text('Loading data... done!')

colT1, colT2 = st.columns([6, 12])
with colT2:st.subheader('Last 5 Year Data')
st.write(df.describe())

colT1, colT2 = st.columns([4, 12])
with colT2:st.subheader('Closing Price vs Time Chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.ylabel('Price')
plt.xlabel('Time')
st.pyplot(fig)

colT1, colT2 = st.columns([2, 12])
with colT2:st.subheader('Closing Price vs 100 Days Moving Average')
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100, 'b',label= '100 Daily Moving Average')
plt.plot(df.Close, 'r',label= 'Close Price')
plt.plot(ma100)
plt.plot(df.Close)
plt.ylabel('Price')
plt.xlabel('Time')
st.pyplot(fig)
plt.legend()

colT1, colT2 = st.columns([1, 12])
with colT2:st.subheader('Closing Price vs 100 & 200 Days Moving Average')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100, 'b',label= '100 Daily Moving Average')
plt.plot(ma100, 'g',label= '200 Daily Moving Average')
plt.plot(df.Close, 'r',label= 'Close Price')
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
plt.ylabel('Price')
plt.xlabel('Time')
st.pyplot(fig)
plt.legend()

clsd=df[['Close']]

ds = clsd.values
normalizer=MinMaxScaler(feature_range=(0,1))
ds_scaled=normalizer.fit_transform(np.array(ds).reshape(-1,1))
train_size = int(len(ds_scaled)*0.70)
test_size = len(ds_scaled) - train_size
ds_train,ds_test=ds_scaled[0:train_size,:],ds_scaled[train_size:len(ds_scaled),:1]

data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.7): int(len(df))])
print(data_training.shape)
print(data_testing.shape)

scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

#data in time series
def create_ds(dataset,step):
    Xtrain, Ytrain = [],[]
    for i in range(len(dataset)-step-1):
        a = dataset[i:(i+step),0]
        Xtrain.append(a)
        Ytrain.append(dataset[i+step,0])
    return np.array(Xtrain), np.array(Ytrain)

time_stamp=100
X_train, Y_train=create_ds(ds_train,time_stamp)
X_test, Y_test = create_ds(ds_test,time_stamp)

#RESHAPE TO FIT LSTM
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)

from keras.models import Sequential
from keras.layers import Dense, LSTM , Dropout

#LSTM Layer 1
model = Sequential()
model.add(LSTM(units=128,return_sequences=True, input_shape=(X_train.shape[1],1)))
#LSTM Layer 2
model.add(LSTM(units=64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=100,batch_size=64,verbose=1)
model.summary()
loss=model.history.history['loss']

colT1, colT2 = st.columns([8, 12])
with colT2:st.subheader('Loss')
fig=plt.figure(figsize=(12,6))
plt.plot(loss)
st.pyplot(fig)

past_100_days=data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted= y_predicted * scale_factor
y_test = y_test * scale_factor

#final graph
colT1, colT2 = st.columns([4, 12])
with colT2:st.subheader('Prediction vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test, 'b',label= 'Original Price')
plt.plot(y_predicted, 'r',label= 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

#Metrics
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
mse1=mse(y_test,y_predicted)
print(mse1)
rmse1=math.sqrt(mse1)
print(rmse1)
mae1=mae(y_test,y_predicted)
print(mae1)

train_predict=model.predict(X_train)
test_predict=model.predict(X_test)
#Inv transform
train_predict=normalizer.inverse_transform(train_predict)
test_predict=normalizer.inverse_transform(test_predict)

test=np.vstack((train_predict,test_predict))

#last 100 days records
fut_inp = ds_test[(len(ds_test)-100):]
fut_inp = fut_inp.reshape(1,-1)
tmp_inp = list(fut_inp)

#list of last 100 data
tmp_inp = tmp_inp[0].tolist()

#predicting next 60 days
lst_output=[]
n_steps=100
i=0
while(i<60):
    if(len(tmp_inp)>100):
        fut_inp=np.array(tmp_inp[1:])
        fut_inp=fut_inp.reshape(1,-1)
        fut_inp=fut_inp.reshape(1,n_steps,1)
        yhat=model.predict(fut_inp,verbose=0)
        tmp_inp.extend(yhat[0].tolist())
        tmp_inp=tmp_inp[1:]
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        fut_inp=fut_inp.reshape((1,n_steps,1))
        yhat=model.predict(fut_inp,verbose=0)
        tmp_inp.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i=i+1
print(lst_output)

#dummy plane
plot_new=np.arange(1,101)
plot_pred=np.arange(101,161)
ds_new=ds_scaled.tolist()

#missing value place
ds_new.extend(lst_output)
plt.plot(ds_new[1200:])

#creating final data for plotting
final_graph=normalizer.inverse_transform(ds_new).tolist()

#final data 60days plot
colT1,colT2 = st.columns([5, 12])
with colT2:st.subheader('60 Days Prediction')
fig=plt.figure(figsize=(12,6))
plt.plot(final_graph,)
plt.plot(ma100)
plt.plot(ma200)
plt.ylabel('Price')
plt.xlabel('Time')
plt.title("{0} Prediction of next 60 days".format(selected_stock))
plt.axhline(y=final_graph[len(final_graph)-1], color = 'red', linestyle = ':', label = 'NEXT 60 D : {0} '.format(round(float(*final_graph[len(final_graph)-1]),2)))
plt.legend()
st.pyplot(fig)