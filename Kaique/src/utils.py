import numpy as np

def print_verification_results(series_names, forecasts, rmses, wrmse):
    print("\nResultados das Previsões:")
    print("-" * 50)
    for name, forecast, rmse in zip(series_names, forecasts, rmses):
        print(f"Série {name}:")
        print(f"Previsões: {np.round(forecast, 4)}")
        print(f"RMSE: {rmse:.6f}")
        print("-" * 50)
    
    print(f"\nWRMSE Total: {wrmse:.6f}")
    
    
def print_final_results(series_names, forecasts):
    print("\nResultados das Previsões:")
    print("-" * 50)
    for name, forecast in zip(series_names, forecasts):
        print(f"Série {name}:")
        print(f"Previsões: {np.round(forecast, 4)}")
        print("-" * 50)
    

def save_forecasts(forecasts, file_path):
    with open(file_path, 'w') as f:
        for i, forecast in enumerate(forecasts, 1):
            f.write(f"Série #{i}: {list(np.round(forecast, 6))}\n")