import matplotlib.pyplot as plt
import os

def plot_and_save_series(historical_data, forecast_data, series_name):
    """
    Cria e salva um gráfico da série temporal com previsões
    
    Args:
        historical_data: Array com os dados históricos
        forecast_data: Array com as previsões
        series_name: Nome da série (para título e nome do arquivo)
    """
    plt.figure(figsize=(12, 6))
    
    # Cria índices para os dados
    hist_idx = range(len(historical_data))
    forecast_idx = range(len(historical_data), len(historical_data) + len(forecast_data))
    
    # Plot dos dados
    plt.plot(hist_idx, historical_data, 'b-', label='Dados Históricos', linewidth=2)
    plt.plot(forecast_idx, forecast_data, 'r--', label='Previsão', linewidth=2)
    
    # Linha divisória
    plt.axvline(x=len(historical_data)-0.5, color='k', linestyle=':', alpha=0.5)
    
    # Configurações do gráfico
    plt.title(f'Série {series_name} - Previsão', pad=20)
    plt.xlabel('Períodos')
    plt.ylabel('Valores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Ajustes finais
    plt.tight_layout()
    
    # Salva o gráfico
    output_dir = "../output"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/{series_name.replace(' ', '_')}_forecast.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gráfico salvo como: {filename}")