import streamlit as sl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve

import joblib
from Classes.RegressionUtils import predict_regresie

df_date = pd.read_csv('../Datasets/regresie_date_finale.csv')

sl.set_page_config(layout="wide")
sl.header(" Pagină pentru predicția temperaturii")
sl.write(
    "Pe această pagină veți putea vedea o analiză comparativă între mai multe modele de Machine Learning, cu scopul de a prezice o valoare continuă."
    " Domeniul pe care l-am ales este cel despre vreme, deoarece este foarte asemănător cu domeniul de statistică sau inteligență artificială, folosindu-se de "
    "probabilități.")
sl.write(
    "Astfel, am ales un set de date de pe Kaggle (https://www.kaggle.com/datasets/budincsevity/szeged-weather/data) care conține măsurători din perioada 2006-2016 ale temperaturii și ale mai multor factori termici care pot influența temperatura direct sau indirect.")
sl.write("Aceste măsurători care se regăsesc în dataset ca și coloane sunt următoarele:")
sl.markdown("""
    - Data (sau timpul când a avut loc măsurătoarea).
    - Tipul de precipitație care a avut loc.
    - Temperatura (atributul din dataset pe care dorim să îl prezicem).
    - Temperatura resimțită.
    - Umiditatea din aer.
    - Viteza vântului.
    - Presiunea atmosferică.
""")
sl.write(
    "Practic, dorim să descoperim niște relații matematice între cele prezentate anterior, astfel încât predicția să fie foarte bună.")
sl.write("Mai jos puteți vedea un exemplu de câteva rânduri din datele noastre:")
sl.dataframe(df_date.head(5))
sl.write(
    "**Notă**: '1' în dreptul coloanelor pentru tipul de precipitație înseamnă faptul că are loc acel tip, respectiv '0' înseamnă că nu are loc.")
sl.divider()

sl.header("Grafice relevante din etapa de descoperire a relațiilor dintre coloane:")

fig1, ax1 = plt.subplots(figsize=(5, 3))
sns.scatterplot(data=df_date, x='Humidity', y='Temperature (C)', ax=ax1)
ax1.set_title("Grafic între temperatură și umiditatea din aer.")
sl.pyplot(fig1, use_container_width=False)

sl.write(
    "În graficul de mai sus se poate observa faptul că relația dintre temperatură și umiditate tinde să aibă un sens comun, spre o anumită direcție, care tinde spre o liniaritate, invers proporțională.")

fig2, ax2 = plt.subplots(figsize=(5, 3))
sns.scatterplot(data=df_date, x='month', y='Temperature (C)', ax=ax2)
ax2.set_title("Grafic între luna din an și temperatură.")
sl.pyplot(fig2, use_container_width=False)

sl.write(
    "De data aceasta se poate observa o creștere a temperaturii până la jumătatea anului, iar apoi o scădere, asemănător cu clopotul lui Gauss, oarecum logic, deoarece temperatura maximă atinsă într-un an este în timpul verii, deci la mijlocul anului.")
sl.write(
    "Restul graficelor fie reprezintă o mulțime care se întinde în tot spațiul de căutări și nu se poate găsi nimic specific, fie datele sunt grupate pe linii sau coloane, de exemplu anul, deoarece nu se poate altfel.")

sl.header("Prezentarea celor mai bune modele de Machine Learning:")
sl.markdown("""
Dataset-ul anterior a fost supus unor teste pe mai multe tipuri de algoritmi de regresie, cum ar fi:
- Linear Regression 
- Decision Tree Regressor 
- Random Forest Regressor 
- Support Vector Regressor 
- K-Nearest Neighbor Regressor 
- Gaussian Process Regressor 
- XGBoost Regressor 
- CatBoost Regressor
- Explainable Boosting Regressor
""")
sl.write("Pentru fiecare s-au folosit parametrii de bază oferiți de clase și s-au calculat următoarele rezultate:")
sl.markdown("""
- MSE
- MAE
- RMSE
- R2_score (coeficient de determinare)
""")
sl.write(
    "Primele 3 reprezintă măsurări ale erorii sub diferite forme, iar R2_score ne spune cât de bine prezice modelul rezultatul continuu.")
sl.write("Astfel, primele 5 modele ca și performanță sunt:")
sl.markdown("""
- Random Forest Regressor
- XGBoost Regressor
- CatBoost Regressor
- K-Nearest Neighbor Regressor
- Explainable Boosting Regressor
""")
sl.write(
    "***Notă*** : pentru primele 3 modele, puteți observa o analiză mai amănunțită din punct de vedere al valorii SHAP.")

modele_5 = {
    "Random Forest Regressor": "rf.pkl",
    "XGBoost Regressor": "xgb.pkl",
    "CatBoost Regressor": "cat.pkl",
    "K-Nearest Neighbor Regressor": "knn.pkl",
    "Explainable Boosting Regressor": "ebr.pkl"
}

optiune = sl.selectbox(
    "Ce model dorești să testezi?",
    list(modele_5.keys())
)

humidity = sl.number_input("Introduceți umiditatea din aer", value=0.0, min_value=0.0, max_value=1.0)
wind_speed = sl.number_input("Introduceți viteza vântului (km/h)", value=0.0, min_value=0.0)
pressure = sl.number_input("Introduceți presiunea atmosferică (millibars)", value=0.0, min_value=0.0)
wind_bearing = sl.number_input("Introduceți direcția vântului (grade)", value=0, min_value=0, max_value=360)
visibility = sl.number_input("Introduceți vizibilitatea (în km)", value=0.0, min_value=0.0)
month = sl.number_input("Introduceți luna anului", value=1, min_value=1, max_value=12)
year = sl.number_input("Introduceți anul", value=2006, min_value=2006, max_value=2026)

precip_type = sl.selectbox(
    "Tipul de precipitații:",
    ["Niciunul", "Ploaie", "Zăpadă"]
)

precip_none = 0
precip_rain = 0
precip_snow = 0

if precip_type == "Niciunul":
    precip_none = 1
elif precip_type == "Ploaie":
    precip_rain = 1
elif precip_type == "Zăpadă":
    precip_snow = 1

data_input = {
    'Humidity': humidity,
    'Wind Speed (km/h)': wind_speed,
    'Wind Bearing (degrees)': wind_bearing,
    'Visibility (km)': visibility,
    'Pressure (millibars)': pressure,
    'year': year,
    'month': month,
    'Precip Type_none': precip_none,
    'Precip Type_rain': precip_rain,
    'Precip Type_snow': precip_snow
}

if sl.button("Calculează Predicția"):
    model_selected_base = "base_" + modele_5[optiune]
    cale_model = f"../Models/{model_selected_base}"
    try:
        model = joblib.load(cale_model)

        predictie = predict_regresie(model, data_input)

        sl.write("### Rezultat:")
        sl.success(f"Predicția pentru input-ul dumneavoastră este: **{predictie:.2f}**")

    except FileNotFoundError:
        sl.error(f"Eroare: Nu am găsit fișierul modelului la calea {cale_model}")
    except Exception as e:
        sl.error(f"A apărut o eroare neașteptată: {e}")

df_scores = pd.read_csv('../Datasets/Scores.csv')

if sl.button('Arată statisticile pentru modelul selectat:'):
    model_selected_opt = optiune
    path_model = f'../Models/opt_{modele_5[optiune]}'
    model_opt = joblib.load(path_model)
    try:
        df_scores = df_scores.set_index('Model')
        mse = df_scores.loc[model_selected_opt, 'MSE']
        mae = df_scores.loc[model_selected_opt, 'MAE']
        rmse = df_scores.loc[model_selected_opt, 'RMSE']
        r2_score = df_scores.loc[model_selected_opt, 'R2-score']
        sl.write(f"Modelul selectat are următoarele erori:")
        sl.markdown(f"""
        - RMSE : {rmse:.2f} - cea mai bună metrică pentru eroare, eroarea este în grade Celsius.
        - MAE : {mae:.2f}
        - MSE : {mse:.2f}
        - R2-score : {r2_score:.2f} : coeficientul de determinare
        """)
        learning_curve_png = f'./Plots/{model_selected_opt}.png'
        sl.image(learning_curve_png, caption=f"Curba de învățare pentru modelul {model_selected_opt}")

        parametrii = model_opt.get_params()
        sl.write("Parametrii folosiți pentru modelul optimizat, la hiperparametrizare:")
        sl.json(parametrii)
    except Exception as e:
        sl.error(f"A apărut o eroare neașteptată: {e}")

sl.header("Analiza SHAP")
sl.write("Analiza valorii SHAP s-a realizat pentru primele 3 modele de regresie ca și performanță.")
shaps = ['Random Forest Regressor', 'CatBoost Regressor', 'XGBoost Regressor']
optiune_shap = sl.selectbox(
    "Ce model dorești să analizezi?",
    shaps
)

if sl.button("Arată analiza:"):

    try:
        sl.write("Analiza globală:")
        sl.write(
            "În această parte, putem observa în graficul de mai jos cât de mult influențează fiecare parametru rezultatul final, pe scurt, cât de importanți sunt aceștia."
            " În plus, se poate evidenția faptul că cel mai mult contează luna anului, umiditatea din aer și dacă ninge sau nu, indiferent de model.")
        image_global = f'./Shap/{optiune_shap}/global.png'
        sl.image(image_global, caption=f"Analiza globală SHAP pentru {optiune_shap}")

        sl.write(
            "În summary plot-ul de mai jos este exemplificată tot influența atributelor, dar de data aceasta ne spune dacă este una bună sau rea, dacă împinge rezultatul regresiei în sus (cu roșu) sau în jos (cu albastru), prin valoarea sa.")
        sl.write(
            "Umiditatea are o valoare ridicată în partea negativă, trăgând în jos rezultatul, iar zăpada trage în sus rezultatul, dar are un impact mic.")
        image_summary = f'./Shap/{optiune_shap}/summary.png'
        sl.image(image_summary, caption=f"Summary plot pentru {optiune_shap}")

        sl.write(
            "În diagrama de tip waterfall de mai jos ne este pus în evidență impactul atributelor dataset-ului. Ce este cu roșu împinge în sus rezultatul regresiei, iar ce este cu albastru împinge în jos."
            " La toate modelele, umiditatea este principalul factor care împinge în jos rezultatul semnificativ."
            " În plus, pe grafic se pot observa 2 valori: base_value (media predicțiilor modelului pe întreg setul de date de antrenament, aproximativ 12) și f(x). Suma tuturor influențelor (valorile SHAP) adunată la base_value ne dă predicția finală.")
        image_waterfall = f'./Shap/{optiune_shap}/waterfall.png'
        sl.image(image_waterfall, caption=f"Waterfall pentru {optiune_shap}")


    except Exception as e:
        sl.error(f"A apărut o eroare neașteptată: {e}")