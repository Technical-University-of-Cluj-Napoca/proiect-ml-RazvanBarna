import streamlit as sl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib
from Classes.ClassificationUtils import predict_clasificare

sl.set_page_config(layout="wide", page_title="Predicție Meciuri Premier League")

sl.header("Pagină pentru predicția rezultatelor din Premier League")
sl.write(
    "Pe această pagină veți putea vedea o analiză comparativă între mai multe modele de Machine Learning, "
    "cu scopul de a prezice rezultatul final al unui meci de fotbal din Premier League. "
    "Domeniul ales este fotbalul, unul dintre cele mai imprevizibile sporturi din lume — "
    "o acuratețe de 60% este considerată foarte bună în acest context."
)
sl.write(
    "Dataset-ul folosit provine de pe Kaggle "
    "(https://www.kaggle.com/datasets/panaaaaa/english-premier-league-and-championship-full-dataset/data?select=England+CSV.csv) "
    "și conține meciuri din Premier League din mai mulți ani, inclusiv anii '90."
)
sl.write(
    "**Perspectiva aleasă**: ne aflăm la pauza meciului, cu acces la statisticile de până la pauză, "
    "și dorim să prezicem rezultatul final."
)
sl.write("Coloanele folosite ca input sunt:")
sl.markdown("""
- Echipa gazdă și echipa oaspete (codificate numeric).
- Rezultatul la pauză (H = gazde, D = egal, A = oaspeți).
- Goluri înscrise la pauză de fiecare echipă.
- Număr de șuturi și șuturi pe spațiul porții (la pauză).
- Cornere, faulturi, cartonașe galbene și roșii (la pauză).
""")
sl.write("**Coloana țintă (y)**: FT Result — rezultatul final al meciului (H / D / A).")

df_date = pd.read_csv('../Datasets/Football.csv')
sl.write("Mai jos puteți vedea câteva rânduri din datele folosite:")
sl.dataframe(df_date.head(5))

sl.divider()

sl.header("Grafice relevante din etapa de explorare a datelor:")

try:
    col1, col2 = sl.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        ft_counts = df_date['FT Result'].value_counts().rename({0: 'Gazde (H)', 1: 'Egal (D)', 2: 'Oaspeți (A)'})
        ft_counts.plot(kind='bar', ax=ax1, color=['#4c72b0', '#dd8452', '#55a868'])
        ax1.set_title("Distribuția rezultatelor finale")
        ax1.set_xlabel("Rezultat")
        ax1.set_ylabel("Număr meciuri")
        ax1.tick_params(axis='x', rotation=0)
        sl.pyplot(fig1, use_container_width=False)
        sl.write(
            "Cea mai frecventă situație este victoria echipei gazde, urmată de egalitate. "
            "Avantajul terenului propriu este semnificativ în Premier League."
        )

    with col2:
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        if 'HT Result' in df_date.columns and 'FT Result' in df_date.columns:
            ct = pd.crosstab(df_date['HT Result'], df_date['FT Result'])
            ct.plot(kind='bar', ax=ax2, colormap='Set2')
            ax2.set_title("Rezultat la pauză vs. Rezultat final")
            ax2.set_xlabel("Rezultat la pauză")
            ax2.set_ylabel("Număr meciuri")
            ax2.tick_params(axis='x', rotation=0)
            ax2.legend(title="Rezultat final", labels=['Gazde', 'Egal', 'Oaspeți'])
        sl.pyplot(fig2, use_container_width=False)
        sl.write(
            "Dacă echipa gazdă conduce la pauză, are cele mai mari șanse să câștige meciul. "
            "Cel mai comun scenariu la pauză este egalul."
        )

except Exception:
    sl.info("Graficele EDA nu pot fi afișate fără datele originale.")

sl.divider()


sl.header("Prezentarea celor mai bune modele de Machine Learning:")
sl.markdown("""
Dataset-ul a fost testat pe mai mulți algoritmi de clasificare:
- Naïve Bayes
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbor
- XGBoost
- CatBoost
- Explainable Boosting Machine (EBM)
""")
sl.write("Primele 5 modele ca și performanță (după acuratețe și AUC-ROC) sunt:")
sl.markdown("""
- **Explainable Boosting Machine (EBM)**
- **Logistic Regression**
- **SVM**
- **Random Forest**
- **CatBoost**
""")
sl.write(
    "***Notă***: pentru primele 3 modele puteți observa o analiză SHAP detaliată mai jos."
)

modele_5 = {
    "Explainable Boosting Machine": "ebm.pkl",
    "Logistic Regression":  "log.pkl",
    "SVM": "svm.pkl",
    "Random Forest": "rf.pkl",
    "CatBoost": "cat.pkl",
}

label_map = {0: " Victorie Gazde (H)", 1: " Egal (D)", 2: " Victorie Oaspeți (A)"}

optiune = sl.selectbox(
    "Ce model dorești să testezi?",
    list(modele_5.keys())
)

sl.subheader("Introduceți statisticile de la pauză:")

col_a, col_b = sl.columns(2)

echipe = [
    "Arsenal", "Aston Villa", "Birmingham", "Blackburn", "Blackpool",
    "Bolton", "Bournemouth", "Brentford", "Brighton", "Burnley",
    "Cardiff", "Chelsea", "Crystal Palace", "Everton", "Fulham",
    "Huddersfield", "Hull", "Ipswich", "Leeds", "Leicester",
    "Liverpool", "Luton", "Man City", "Man United", "Middlesbrough",
    "Newcastle", "Norwich", "Nottm Forest", "Portsmouth", "QPR",
    "Reading", "Sheffield United", "Southampton", "Stoke", "Sunderland",
    "Swansea", "Tottenham", "Watford", "West Brom", "West Ham",
    "Wigan", "Wolves"
]

with col_a:
    home_team_name = sl.selectbox("Echipa gazdă", echipe, index=0)
    hth_goals = sl.number_input("Goluri gazde la pauză (HTH Goals)", value=0, min_value=0, max_value=10)
    h_shots = sl.number_input("Șuturi gazde (H Shots)", value=0, min_value=0, max_value=30)
    h_sot = sl.number_input("Șuturi pe poartă gazde (H SOT)", value=0, min_value=0, max_value=20)
    h_corners = sl.number_input("Cornere gazde (H Corners)", value=0, min_value=0, max_value=20)
    h_fouls = sl.number_input("Faulturi gazde (H Fouls)", value=0, min_value=0, max_value=20)
    h_yellow = sl.number_input("Cartonașe galbene gazde (H Yellow)", value=0, min_value=0, max_value=5)
    h_red = sl.number_input("Cartonașe roșii gazde (H Red)", value=0, min_value=0, max_value=3)

with col_b:
    away_team_name = sl.selectbox("Echipa oaspete", echipe, index=1)
    hta_goals = sl.number_input("Goluri oaspeți la pauză (HTA Goals)", value=0, min_value=0, max_value=10)
    a_shots = sl.number_input("Șuturi oaspeți (A Shots)", value=0, min_value=0, max_value=30)
    a_sot = sl.number_input("Șuturi pe poartă oaspeți (A SOT)", value=0, min_value=0, max_value=20)
    a_corners = sl.number_input("Cornere oaspeți (A Corners)", value=0, min_value=0, max_value=20)
    a_fouls = sl.number_input("Faulturi oaspeți (A Fouls)", value=0, min_value=0, max_value=20)
    a_yellow = sl.number_input("Cartonașe galbene oaspeți (A Yellow)", value=0, min_value=0, max_value=5)
    a_red = sl.number_input("Cartonașe roșii oaspeți (A Red)", value=0, min_value=0, max_value=3)

ht_result_input = sl.selectbox(
    "Rezultatul la pauză (HT Result)",
    options=[("Gazde conduc (H)", 0), ("Egal (D)", 1), ("Oaspeți conduc (A)", 2)],
    format_func=lambda x: x[0]
)
ht_result_val = ht_result_input[1]

sorted_teams = sorted(echipe)
team_grid = {team: i for i, team in enumerate(sorted_teams)}
home_team_code = team_grid.get(home_team_name, 0)
away_team_code = team_grid.get(away_team_name, 1)

data_input = {
    'HomeTeam': home_team_code,
    'AwayTeam': away_team_code,
    'HTH Goals': hth_goals,
    'HTA Goals': hta_goals,
    'HT Result': ht_result_val,
    'H Shots': h_shots,
    'A Shots': a_shots,
    'H SOT': h_sot,
    'A SOT': a_sot,
    'H Corners': h_corners,
    'A Corners': a_corners,
    'H Fouls': h_fouls,
    'A Fouls': a_fouls,
    'H Yellow': h_yellow,
    'A Yellow': a_yellow,
    'H Red': h_red,
    'A Red': a_red,
}

if sl.button("Calculează Predicția"):
    model_filename = "base_" + modele_5[optiune]
    cale_model = f"../Models/Classification/{model_filename}"
    try:
        model = joblib.load(cale_model)
        predictie, proba = predict_clasificare(model, data_input)
        rezultat_text = label_map.get(predictie, str(predictie))

        sl.write("### Rezultat:")
        sl.success(f"Predicția pentru meciul **{home_team_name} vs {away_team_name}** este: **{rezultat_text}**")

        if proba is not None:
            sl.write("**Probabilități per clasă:**")
            prob_df = pd.DataFrame({
                "Rezultat": [" Gazde (H)", " Egal (D)", "Oaspeți (A)"],
                "Probabilitate (%)": [f"{p*100:.1f}%" for p in proba]
            })
            sl.table(prob_df)

    except FileNotFoundError:
        sl.error(f"Eroare: Nu am găsit modelul la calea '{cale_model}'.")
    except Exception as e:
        sl.error(f"A apărut o eroare neașteptată: {e}")

sl.write("***Nota: Am tot incercat sa dau run la modele si sa le salvez, cateodata merge sa faca predictia, cateodata da eroare de dat fit chiar daca le dau si le las sa ruleze in notebook si sa se salveze.***")
sl.divider()


try:
    df_scores = pd.read_csv('../Datasets/Scores_clasificare.csv')
except FileNotFoundError:
    df_scores = None

if sl.button("Arată statisticile pentru modelul selectat"):
    path_model_opt = f'../Models/Classification/best_{modele_5[optiune]}'
    try:
        model_opt = joblib.load(path_model_opt)

        if df_scores is not None:
            df_sc = df_scores.set_index('Model')
            acc   = df_sc.loc[optiune, 'Accuracy']
            prec  = df_sc.loc[optiune, 'Precision']
            rec   = df_sc.loc[optiune, 'Recall']
            f1    = df_sc.loc[optiune, 'F1-Score']
            auc   = df_sc.loc[optiune, 'AUC-ROC']
            sl.write(f"Modelul **{optiune}** optimizat are următoarele performanțe:")
            sl.markdown(f"""
            - **Accuracy**: {acc:.2f} — proporția predicțiilor corecte
            - **Precision**: {prec:.2f} — precizia medie ponderată
            - **Recall**: {rec:.2f} — acoperirea medie ponderată
            - **F1-score**: {f1:.2f} — media armonică dintre precision și recall
            - **AUC-ROC**: {auc:.2f} — capacitatea de discriminare între clase (one-vs-rest)
            """)
        else:
            sl.info("Fișierul cu scoruri nu a fost găsit.")

        learning_curve_png = f'./Plots/{optiune}.png'
        sl.image(learning_curve_png, caption=f"Curba de învățare pentru {optiune}")

        parametrii = model_opt.get_params()
        sl.write("Parametrii modelului optimizat (hiperparametrizare):")
        sl.json(parametrii)

    except FileNotFoundError:
        sl.error(f"Modelul optimizat nu a fost găsit la calea '{path_model_opt}'.")
    except Exception as e:
        sl.error(f"A apărut o eroare neașteptată: {e}")

sl.divider()

sl.header("Analiza SHAP")
sl.write(
    "Analiza valorilor SHAP s-a realizat pentru primele 3 modele de clasificare ca și performanță: "
    "EBM, CatBoost și SVM. "
    "SHAP ne ajută să înțelegem **de ce** modelul a luat o anumită decizie, "
    "identificând care caracteristici au influențat cel mai mult predicția."
)

shaps = [
    "Explainable Boosting Machine",
    "Logistic Regression",
    "Support Vector Machine",
]
optiune_shap = sl.selectbox(
    "Ce model dorești să analizezi (SHAP)?",
    shaps
)

clasa_shap = sl.selectbox(
    "Pentru ce rezultat dorești analiza?",
    options=[("Victorie Gazde (H)", 0), ("Egal (D)", 1), ("Victorie Oaspeți (A)", 2)],
    format_func=lambda x: x[0]
)
clasa_idx = clasa_shap[1]
clasa_nume = clasa_shap[0]

if sl.button("Arată analiza SHAP"):
    try:
        sl.subheader("Importanța globală a caracteristicilor")
        sl.write(
            "Graficul de mai jos arată cât de mult influențează fiecare caracteristică predicția modelului, "
            "în medie, pe toate exemplele și toate clasele. "
            "Cel mai important atribut este **HT Result** (rezultatul la pauză), "
            "urmat de șuturile pe spațiul porții și golurile de la pauță."
        )
        image_global = f'./Shap/{optiune_shap}/global.png'
        sl.image(image_global, caption=f"Importanța globală SHAP — {optiune_shap}")

        sl.subheader(f"Summary Plot — Clasa: {clasa_nume}")
        sl.write(
            "Summary plot-ul de mai jos arată direcția influenței fiecărei caracteristici pentru clasa selectată: "
            "valorile roșii împing predicția **în sus** (spre această clasă), "
            "iar cele albastre o împing **în jos**."
        )
        image_summary = f'./Shap/{optiune_shap}/summary_clasa_{clasa_idx}.png'
        sl.image(image_summary, caption=f"Summary Plot SHAP — {optiune_shap} — {clasa_nume}")

        sl.subheader("Explicație locală — Waterfall Plot")
        sl.write(
            "Diagrama waterfall explică predicția pentru un singur exemplu concret. "
            "Valoarea de bază (**base value**) reprezintă media predicțiilor modelului pe setul de antrenament. "
            "Fiecare caracteristică adaugă sau scade din această valoare, rezultând predicția finală **f(x)**."
        )
        image_waterfall = f'./Shap/{optiune_shap}/waterfall.png'
        sl.image(image_waterfall, caption=f"Waterfall SHAP — {optiune_shap} — {clasa_nume}")

        sl.subheader("Top caracteristici influente")
        sl.markdown("""
        Cele mai importante atribute care influențează predicția modelului sunt:
        - **HT Result** (rezultatul la pauză) — cel mai puternic predictor. Dacă gazdele conduc la pauză, șansele de victorie finală cresc semnificativ.
        - **H SOT / A SOT** (șuturi pe poartă) — cu cât o echipă șutează mai mult pe poartă, cu atât modelul crește probabilitatea victoriei ei.
        - **HomeTeam** (echipa gazdă) — unele echipe au un avantaj acasă mai pronunțat decât altele.
        """)


    except FileNotFoundError as e:
        sl.error(f"Imaginea SHAP nu a fost găsită: {e}. Asigurați-vă că imaginile sunt salvate în './Shap/{optiune_shap}/'.")
    except Exception as e:
        sl.error(f"A apărut o eroare neașteptată: {e}")