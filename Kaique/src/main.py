import os
import sys
import pandas as pd
from data_loader import load_data, prepare_series
from model_trainer import train_arima_model, forecast
from forecast_evaluator import calculate_rmse, calculate_wrmse
from utils import print_results, save_forecasts

def main():
    try:
        DATA_PATH = "../data/DadosCompeticao.xlsx"
        OUTPUT_PATH = "../output/previsoes.txt"
        HORIZON = 12  # Prever os próximos 12 meses
        
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Arquivo de dados não encontrado em {DATA_PATH}")
        
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        
        print("Carregando dados...")
        df = load_data(DATA_PATH)
        series = prepare_series(df)
        series_names = list(series.keys())
        
        print("\nTreinando modelos ARIMA...")
        forecasts = []
        rmses = []
        
        for name, data in series.items():
            print(f"\nProcessando série {name}...")
            
            try:
                if len(data) < HORIZON + 5:
                    raise ValueError(f"Dados insuficientes para a série {name}")
                
                train = data[:-HORIZON]
                test = data[-HORIZON:]
                
                model = train_arima_model(train)
                
                pred = forecast(model, HORIZON)
                forecasts.append(pred)
                
                rmse = calculate_rmse(test, pred)
                rmses.append(rmse)
                
                print(f"Previsão concluída para série {name} | RMSE: {rmse:.4f}")
                
            except Exception as e:
                print(f"Erro ao processar série {name}: {str(e)}")
                forecasts.append([np.nan] * HORIZON)
                rmses.append(np.nan)
                continue
        
        valid_rmses = [rmse for rmse in rmses if not np.isnan(rmse)]
        wrmse = calculate_wrmse(valid_rmses) if valid_rmses else np.nan
        
        print_results(series_names, forecasts, rmses, wrmse)
        save_forecasts(forecasts, OUTPUT_PATH)
        print(f"\nPrevisões salvas em {os.path.abspath(OUTPUT_PATH)}")
        
    except Exception as e:
        print(f"\nErro durante a execução: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()