from Classes import FootballFeature
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def create_columns(df, feat:FootballFeature, n=5):
    new_rows = []

    for idx,row in df.iterrows():
        current_date = row['Date']
        h_team = row["HomeTeam"]
        a_team = row['AwayTeam']

        h_history = feat.get_team_history(h_team, current_date, n=n)
        a_history = feat.get_team_history(a_team, current_date, n=n)
        h2h = feat.get_H2H(h_team, a_team, current_date, n=n)

        if h_history is not None and a_history is not None:
            stats = {
                'original_index': idx,
                'H_Scored_Avg': feat.get_scored_goals(h_team, h_history),
                'A_Scored_Avg': feat.get_scored_goals(a_team, a_history),
                'H_Conceded_Avg': feat.get_conceded_goals(h_team, h_history),
                'A_Conceded_Avg': feat.get_conceded_goals(a_team, a_history),
                'H_Points_Avg': feat.get_last_results(h_team, h_history),
                'A_Points_Avg': feat.get_last_results(a_team, a_history),
                'H_Red_Avg': feat.get_red_cards(h_team, h_history),
                'A_Red_Avg': feat.get_red_cards(a_team, a_history),
                'H2H_Score': h2h if h2h is not None else 0
            }
            new_rows.append(stats)

    stats_df = pd.DataFrame(new_rows).set_index('original_index')

    final_df = df.join(stats_df, how='inner')

    final_df['Diff_Form'] = final_df['H_Points_Avg'] - final_df['A_Points_Avg']
    final_df['Diff_Scored'] = final_df['H_Scored_Avg'] - final_df['A_Scored_Avg']
    final_df['Diff_Conceded'] = final_df['H_Conceded_Avg'] - final_df['A_Conceded_Avg']

    return final_df

def plot_learning_curve(estimator, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y,
        cv=5,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring='f1_weighted',
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Scor Antrenare")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Scor Validare (CV)")


    plt.title(title)
    plt.xlabel("Nr. meciuri")
    plt.ylabel("Scor F1-Weighted")
    plt.legend(loc="best")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()