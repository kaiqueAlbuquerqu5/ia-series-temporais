# abordagem MLP
# prediz os próximos 12 meses




##### BIBLIOTECAS

import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import math





##### DADOS E VARS GLOBAIS

# dados
data = pd.read_excel("DadosCompeticao.xlsx")
# razao treino/teste
r = .8
# indice final dos dados de treino
split = round(data.shape[0]*r) 




##### FUNCOES DO MODELO

# preparacao:
# separa entre dados de treino e dados de teste
def tt_split(s, n_months = 12):
    # inputs de treino (ultimos n_months)
    single_col_X_train = data.iloc[:split,s]
    X_train = pd.DataFrame(np.zeros((single_col_X_train.shape[0] - n_months, n_months)))
    for col in range(n_months):
        X_train.iloc[:,col] = single_col_X_train.shift(-col).head(single_col_X_train.shape[0] - n_months)
    # outputs de treino (proximos 12 meses)
    single_col_Y_train = data.iloc[n_months:split+12,s]
    Y_train = pd.DataFrame(np.zeros((single_col_X_train.shape[0] - n_months, 12)))
    for col in range(12):
        Y_train.iloc[:,col] = single_col_Y_train.shift(-col).head(single_col_Y_train.shape[0] - 12)
    # bagunçando aleatoriamente os dados de treino
    concatd = pd.concat([X_train, Y_train], axis=1).sample(frac=1).reset_index(drop=True)
    X_train = concatd.iloc[:,:12]
    Y_train = concatd.iloc[:,12:]
    # inputs de teste
    single_col_X_test = data.iloc[split-n_months:-11,s]
    X_test = pd.DataFrame(np.zeros((data.shape[0]-split-11, n_months)))
    for col in range(n_months):
        X_test.iloc[:,col] = single_col_X_test.shift(-col).head(single_col_X_test.shape[0] - n_months)
    single_col_Y_test = data.iloc[split:,s]
    # outputs de teste
    Y_test = pd.DataFrame(np.zeros((single_col_X_test.shape[0] - n_months, 12)))
    for col in range(12):
        Y_test.iloc[:,col] = single_col_Y_test.shift(-col).head(single_col_Y_test.shape[0] - 11).values
    # ultimo periodo dos dados, para previsao dos proximos 12 meses
    X_final = pd.DataFrame(np.zeros((1, n_months)))
    for col in range(n_months):
        X_final.iloc[0,-(1+col)] = data.iloc[-(1+col),s]
    return (X_train, Y_train, X_test, Y_test, X_final)


# roda o modelo:
def run_model(s, tt_data):
    X_train, Y_train, X_test, Y_test, X_final = tt_data
    # camada 1: normalizacao dos dados de teste
    norm_layer = tf.keras.layers.Normalization()
    norm_layer.adapt(X_train.values.reshape(-1,1))
    # demais camadas:
    # (funcao relu nas camadas ocultas por eficiencia,
    # funcao sigmoide ou padrao no output, conforme testes
    # feitos para cada serie)
    if s in [3,10]:
        model = tf.keras.Sequential([
            norm_layer,
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dense(12)
        ])
    else:
        model = tf.keras.Sequential([
            norm_layer,
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dense(12, activation="sigmoid")
        ])
    # definindo o otimizador do gradient descent
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    # configurando o modelo
    model.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"])
    # rodando
    history = model.fit(X_train, Y_train, epochs=50)
    # avaliando
    mse_test, rmse_test = model.evaluate(X_test, Y_test)
    # prevendo para os dados de teste
    Y_pred = model.predict(X_test)
    # agora prevendo os 12 meses posteriores aos dados dispinibilizados
    Y_final = model.predict(X_final)
    return (Y_pred, rmse_test, Y_final)

# funcao que roda o modelo para todas as series
# e retorna um dicionario com subdicionarios com resultados de cada serie
def run_all_series():
    result = {}
    for s in range(data.shape[1]):
        key = f"series #{s}"
        tt_data = tt_split(s)
        Y_pred, rmse_test, Y_final = run_model(s, tt_data)
        result[key] = {
            "test": tt_data[3],
            "predicted": Y_pred,
            "rmse": rmse_test,
            "final_prediction": Y_final
        }
    return result

# funcao que plota tudo
def plot_all_series(all_series_dict):
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    for i, ax in enumerate(axes.flat):  # axes.flat transforma a grade em uma lista
        if i < 11:
            key = f"series #{i}"
            rmse_test = all_series_dict[key]["rmse"]
            # plot das series de teste
            Y_test = all_series_dict[key]["test"]
            for k in range(len(Y_test)):
                test = pd.Series(Y_test.iloc[k,:].values, index=list(range(k,k+12)))
                ax.plot(test, label="real")
            # plot das series de previsao
            Y_pred = all_series_dict[key]["predicted"]
            for j in range(len(Y_pred)):
                pred = pd.Series(Y_pred[j], index=list(range(j,j+12)))
                ax.plot(pred, label="simulado", linestyle="--", alpha=0.4)
            ax.set_title(f"Série {i} (RSME: {round(rmse_test, 3)})")
            #ax.legend(fontsize=8)
        else:
            ax.axis("off")  # desativar slot extra
    plt.tight_layout()
    plt.show()




##### RODA E VISUALIZA

# rodando o modelo
all_series_dict = run_all_series()

# visualizando
plot_all_series(all_series_dict)





##### SALVA

# salvando os dados
Y_final_df = pd.DataFrame([value["final_prediction"][0] for value in all_series_dict.values()]).T
Y_final_df.columns = [f"#{i+1}" for i in range(11)]
Y_final_df.to_excel("output_henrique.xlsx", index=False)




##### RMSE


# concatena em uma so coluna todas os dados de teste (reais),
# e em outra coluna as respectivas previsoes 
def consolidate(test, predicted):
    test_consolidated = pd.melt(test).iloc[:,1]
    predicted_consolidated = pd.melt(pd.DataFrame([prediction for prediction in predicted])).iloc[:,1]
    result = pd.DataFrame({"test": test_consolidated, "predicted": predicted_consolidated})
    return result

# pega df de duas colunas e calcula a rsme entre elas
def rmse(df):
    quad_diffs = (df.iloc[:,0] - df.iloc[:,1])**2
    result = math.sqrt(quad_diffs.sum()/len(quad_diffs))
    return result

# calculando os rmses de cada serie
rmses = []
for i in range(data.shape[1]):
    rmses.append(rmse(consolidate(all_series_dict[f"series #{i}"]["test"], all_series_dict[f"series #{i}"]["predicted"])))
pd.DataFrame([rmses]).to_csv("rmses_henrique.csv", index=False)