import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

scaler = joblib.load('../Models/Classification/scaler.pkl')
def predict_clasificare(model, data_input: dict):
    coloane_ordine = [
        'HomeTeam', 'AwayTeam',
        'HTH Goals', 'HTA Goals', 'HT Result',
        'H Shots', 'A Shots',
        'H SOT', 'A SOT',
        'H Corners', 'A Corners',
        'H Fouls', 'A Fouls',
        'H Yellow', 'A Yellow',
        'H Red', 'A Red',
    ]

    df = pd.DataFrame([data_input])[coloane_ordine]

    X = df.values
    X = scaler.transform(X)

    predictie = int(model.predict(X)[0])

    proba = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)[0]

    return predictie, proba