import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler


scaler = joblib.load('../Models/scaler.pkl')
def predict_regresie(model, input):
    input_df = pd.DataFrame([input])
    input_scaled = scaler.transform(input_df)
    output = model.predict(input_scaled)

    return output[0]
