from read_data import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

fred = import_data()
y = tf.convert_to_tensor(fred[['CPIAUCSL']])
X = tf.convert_to_tensor(fred[['CPIAPPSL', 'CPITRNSL', 'CPIMEDSL', 'CUSR0000SAC', 'CUSR0000SAD', 'CUSR0000SAS']])
model = Sequential(input_shape = (6, 1))
model.add(LSTM(units = 50, return_sequences = True, input_shape=(1, 1)))
model.add(LSTM(units = 50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X, y, epochs=20, batch_size=32)