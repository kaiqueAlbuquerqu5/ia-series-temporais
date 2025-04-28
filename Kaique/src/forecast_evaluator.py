import numpy as np

def calculate_rmse(actual, predicted):
    """Calcula a Raiz do Erro Quadrático Médio (RMSE)"""
    return np.sqrt(np.mean((np.array(actual) - np.array(predicted))**2))

def calculate_wrmse(rmses, weights=None):
    """Calcula o RMSE Ponderado (WRMSE) para todas as séries"""
    if weights is None:
        weights = [1/len(rmses)] * len(rmses)  # Peso igual se não fornecido
    return np.sum([w * rmse for w, rmse in zip(weights, rmses)])