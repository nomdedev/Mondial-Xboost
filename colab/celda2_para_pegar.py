# CELDA 2: CARGAR DATOS
from google.colab import drive
import pandas as pd
import numpy as np
import os

drive.mount('/content/drive')

DATA_PATH = '/content/drive/MyDrive/Mondial-Xboost/data/all_matches.parquet'
print(f'Leyendo: {DATA_PATH}')

raw = pd.read_parquet(DATA_PATH)
raw['date'] = pd.to_datetime(raw['date'], errors='coerce')
raw = raw.dropna(subset=['date','home_team','away_team','home_goals','away_goals'])
raw = raw.sort_values('date').reset_index(drop=True)
raw['home_goals'] = raw['home_goals'].astype(int)
raw['away_goals'] = raw['away_goals'].astype(int)
raw['outcome'] = np.where(raw['home_goals']>raw['away_goals'],0,
                 np.where(raw['home_goals']==raw['away_goals'],1,2))
DATA_INFO = {
    'total': len(raw), 'leagues': raw['league_code'].nunique(),
    'from': str(raw['date'].min().date()), 'to': str(raw['date'].max().date()),
    'home_wins': int((raw.outcome==0).sum()),
    'draws': int((raw.outcome==1).sum()),
    'away_wins': int((raw.outcome==2).sum())
}
print(f'Dataset: {DATA_INFO["total"]} partidos, {DATA_INFO["leagues"]} ligas')
print(f'Rango: {DATA_INFO["from"]} -> {DATA_INFO["to"]}')