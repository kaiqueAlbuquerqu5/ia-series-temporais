import pandas as pd
import numpy as np

def load_data(file_path):
    df = pd.read_excel(file_path, header=0)
    return df

def prepare_series(df):
    series = {}
    for col in df.columns:
        series[col] = df[col].dropna().values
    return series