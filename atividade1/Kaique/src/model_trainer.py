import warnings
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

def difference_series(series, d):
    for _ in range(d):
        series = series.diff().dropna()
    return series

def find_best_d(series, max_d=2):
    d = 0
    for _ in range(max_d):
        result = adfuller(series)
        if result[1] > 0.05:
            series = difference_series(series, 1)
            d += 1
        else:
            break
    return d

def train_arima_model(series, p_range=(0, 2), d_range=(0, 1), q_range=(0, 2)):
    warnings.filterwarnings("ignore")
    
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    
    d = find_best_d(series.copy())
    
    best_aic = np.inf
    best_model = None
    
    for p in range(p_range[0], p_range[1]+1):
        for q in range(q_range[0], q_range[1]+1):
            try:
                model = ARIMA(series, order=(p, d, q))
                model_fit = model.fit()
                if model_fit.aic < best_aic:
                    best_aic = model_fit.aic
                    best_model = model_fit
            except:
                continue
                
    if best_model is None:
        model = ARIMA(series, order=(1, d, 1))
        best_model = model.fit()
    
    return best_model

def forecast(model, steps):
    return model.forecast(steps=steps)