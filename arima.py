from read_data import *
from statsmodels.tsa.arima.model import ARIMA

fred = import_data()
order = (1 ,1 ,1)
model = ARIMA(fred['CPIAUCSL'], order = order)
model_fit = model.fit()
print(model_fit.get_forecast(steps = 1).predicted_mean)