from Classes import FootballFeature
import pandas as pd

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