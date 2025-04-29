import os
import sys
import pandas as pd
import numpy as np
from data_loader import load_data, prepare_series
from model_trainer import train_arima_model, forecast
from forecast_evaluator import calculate_rmse, calculate_wrmse
from utils import print_verification_results, print_final_results, save_forecasts

# Constantes
DATA_PATH = "../data/DadosCompeticao.xlsx"
OUTPUT_PATH = "../output/previsoes.txt"
HORIZON = 12  # Número de meses para prever
MIN_TRAIN_SIZE = 5  # Mínimo de pontos para treino

def load_and_validate_data():
    """Carrega e valida os dados"""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Arquivo de dados não encontrado em {DATA_PATH}")
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    print("Carregando dados...")
    df = load_data(DATA_PATH)
    series = prepare_series(df)
    return series, list(series.keys())

def process_series_verification(series_data):
    """Processa as séries no modo de verificação (com validação)"""
    forecasts = []
    rmses = []
    
    for name, data in series_data.items():
        print(f"\nProcessando série {name}...")
        
        try:
            if len(data) < HORIZON + MIN_TRAIN_SIZE:
                raise ValueError(f"Dados insuficientes para a série {name}")
            
            train = data[:-HORIZON]
            test = data[-HORIZON:]
            
            model = train_arima_model(train)
            pred = forecast(model, HORIZON)
            
            forecasts.append(pred)
            rmses.append(calculate_rmse(test, pred))
            
            print(f"Previsão concluída para série {name} | RMSE: {rmses[-1]:.4f}")
            
        except Exception as e:
            print(f"Erro ao processar série {name}: {str(e)}")
            forecasts.append([np.nan] * HORIZON)
            rmses.append(np.nan)
    
    return forecasts, rmses

def process_series_prediction(series_data):
    """Processa as séries no modo de previsão final"""
    forecasts = []
    
    for name, data in series_data.items():
        print(f"\nProcessando série {name}...")
        
        try:
            model = train_arima_model(data)
            pred = forecast(model, HORIZON)
            forecasts.append(pred)
            print(f"Previsão concluída para série {name}")
            
        except Exception as e:
            print(f"Erro ao processar série {name}: {str(e)}")
            forecasts.append([np.nan] * HORIZON)
    
    return forecasts

def verify_model():
    """Fluxo completo para verificação do modelo"""
    series_data, series_names = load_and_validate_data()
    forecasts, rmses = process_series_verification(series_data)
    wrmse = calculate_wrmse([rmse for rmse in rmses if not np.isnan(rmse)])
    print_verification_results(series_names, forecasts, rmses, wrmse)
    save_forecasts(forecasts, OUTPUT_PATH)

def obtain_predictions():
    """Fluxo completo para obtenção de previsões"""
    series_data, series_names = load_and_validate_data()
    forecasts = process_series_prediction(series_data)
    print_final_results(series_names, forecasts)
    save_forecasts(forecasts, OUTPUT_PATH)

def main():
    try:
        check_or_result = int(input("Deseja checar o modelo [0] ou ver os resultados [1]? "))
        
        if check_or_result == 0:
            print("\nModo de verificação do modelo (com validação)")
            verify_model()
        else:
            print("\nModo de previsão final")
            obtain_predictions()
            
        print(f"\nPrevisões salvas em {os.path.abspath(OUTPUT_PATH)}")
        
    except Exception as e:
        print(f"\nErro durante a execução: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()