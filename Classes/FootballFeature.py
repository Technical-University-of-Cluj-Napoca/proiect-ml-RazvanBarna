import pandas as pd
from datetime import datetime

class FootballFeature:

    def __init__(self, df):
        self.df = df.copy()
        self.df['Date'] = pd.to_datetime(self.df['Date'], format='%d/%m/%Y')
        self.df = self.df.sort_values('Date')


    def get_team_history(self, team:int, current_time : datetime, n = 5):
        my_filter = (
                ((self.df['HomeTeam'] == team) | (self.df['AwayTeam'] == team)) &
                (self.df['Date'] < current_time)
        )
        last_matches = self.df[my_filter]
        if len(last_matches) < n:
            return None
        return last_matches.tail(n)

    def get_scored_goals(self, team:int, history):
        goals = []
        for _, row in history.iterrows():
            if row['HomeTeam'] == team:
                goals.append(row['FTH Goals'])
            else:
                goals.append(row['FTA Goals'])
        return  sum(goals) / len(goals)

    def get_conceded_goals(self, team:int, history):
        goals = []
        for _, row in history.iterrows():
            if row['HomeTeam'] == team:
                goals.append(row['FTA Goals'])
            else:
                goals.append(row['FTH Goals'])
        return sum(goals) / len(goals)

    def get_red_cards(self, team:int, history):
        red_cards = []
        for _, row in history.iterrows():
            if row['HomeTeam'] == team:
                red_cards.append(row['H Red'])
            else:
                red_cards.append(row['A Red'])
        return sum(red_cards) / len(red_cards)

    def get_H2H(self, team_1H: int, team_2A:int,current_time, n=5):
        my_filter = (
                ((self.df['HomeTeam'] == team_1H) & (self.df['AwayTeam'] == team_2A) |
                 (self.df['HomeTeam'] == team_2A) & (self.df['AwayTeam'] == team_1H)) &
                (self.df['Date'] < current_time)
        )
        last_matches = self.df[my_filter].tail(n)
        if len(last_matches) < n:
            return None

        h2h_score = 0
        # H = 0, D = 1, A =2
        for _, row in last_matches.iterrows():
            if row['HomeTeam'] == team_1H and row['FT Result'] == 0:
                h2h_score += 1
            elif row['AwayTeam'] == team_1H and row['FT Result'] == 2:
                h2h_score += 1
        return h2h_score

    def get_last_results(self, team:int, history):
        points = []
        for _,row in history.iterrows():
            if row['HomeTeam'] == team:
                if row['FT Result'] == 0:
                    points.append(3)
                elif row['FT Result'] == 1:
                    points.append(1)
                else:
                    points.append(0)
            else:
                if row['FT Result'] == 2:
                    points.append(3)
                elif row['FT Result'] == 1:
                    points.append(1)
                else:
                    points.append(0)
        return sum(points) / len(points)
