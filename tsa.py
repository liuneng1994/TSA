from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
import numpy as np
import os


def decompose(ts, freqs=(3, 6), verbose=False, name='time_series', export_dir='./'):
    result = []
    for freq in freqs:
        decomposition = seasonal_decompose(ts, freq=freq)
        if verbose:
            plt.subplot(411)
            plt.plot(ts, label='Original')
            plt.legend(loc='best')
            plt.subplot(412)
            plt.plot(decomposition.trend, label='Trend')
            plt.legend(loc='best')
            plt.subplot(413)
            plt.plot(decomposition.seasonal, label='Seasonality')
            plt.legend(loc='best')
            plt.subplot(414)
            plt.plot(decomposition.resid, label='Residuals')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig(os.path.join(export_dir, "%s_freq_%d" % (name, freq)))
        result.append(
            (freq, {'trend': decomposition.trend, 'seasonal': decomposition.seasonal, 'resid': decomposition.resid}))
    return result


def combine_predict(predict_trend, predict_resid, seasonal, time_index):
    seasonal_map = dict()
    for idx in seasonal.index:
        if idx not in seasonal:
            seasonal_map[idx.month] = [seasonal[idx]]
        else:
            seasonal_map[idx.month].append(seasonal[idx])
    for month in seasonal_map:
        seasonal_map[month] = np.mean(seasonal_map[month])
    result = predict_trend + predict_resid
    for idx, time in enumerate(time_index):
        result[idx] += seasonal_map[idx]


def predict(ts,params,num):
    model = ARIMA(ts.dropna(),order=params)
    result = model.fit(disp=-1,method='css')
    result.forecast(num)
