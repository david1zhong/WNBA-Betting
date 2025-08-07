import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import json
import warnings
import psycopg2
from sqlalchemy import create_engine, text

warnings.filterwarnings('ignore')


class WNBACyclicalPatternDetector:
    def __init__(self, focus_player=None, custom_game_dates=None):
        self.focus_player = focus_player
        self.custom_game_dates = custom_game_dates or []
        self.years = {
            2025: "playerboxes/player_box_2025.csv",
            2024: "playerboxes/player_box_2024.csv",
            2023: "playerboxes/player_box_2023.csv",
            2022: "playerboxes/player_box_2022.csv",
            2021: "playerboxes/player_box_2021.csv",
            2020: "playerboxes/player_box_2020.csv",
            2019: "playerboxes/player_box_2019.csv",
            2018: "playerboxes/player_box_2018.csv",
            2017: "playerboxes/player_box_2017.csv",
            2016: "playerboxes/player_box_2016.csv",
            2015: "playerboxes/player_box_2015.csv",
            2014: "playerboxes/player_box_2014.csv",
            2013: "playerboxes/player_box_2013.csv",
            2012: "playerboxes/player_box_2012.csv",
            2011: "playerboxes/player_box_2011.csv",
            2010: "playerboxes/player_box_2010.csv",
            2009: "playerboxes/player_box_2009.csv"
        }

    def get_current_players(self):
        try:
            df_2024 = pd.read_csv(self.years[2025])
            current_players = set(df_2024['athlete_display_name'].unique())
            return current_players
        except Exception:
            return set()

    def load_all_player_data(self, current_players_only=True):
        all_data = []
        current_players = self.get_current_players() if current_players_only else None

        for year, file_path in self.years.items():
            try:
                df = pd.read_csv(file_path)
                if current_players_only and current_players:
                    df = df[df['athlete_display_name'].isin(current_players)]
                if not df.empty:
                    df['Year'] = year
                    all_data.append(df)
            except:
                pass

        if not all_data:
            raise ValueError("No data found")

        combined_data = pd.concat(all_data, ignore_index=True)
        return self.prepare_data(combined_data)

    def prepare_data(self, df):
        """Clean and prepare the data for analysis"""
        df = df.copy()
        df['game_date'] = pd.to_datetime(df['game_date'])
        df['shooting_pct'] = df['field_goals_made'] / np.maximum(df['field_goals_attempted'], 1)
        df['points_per_minute'] = df['points'] / np.maximum(df['minutes'], 1)
        df['day_of_month'] = df['game_date'].dt.day
        df['month'] = df['game_date'].dt.month
        df['days_since_epoch'] = (df['game_date'] - pd.Timestamp('2009-01-01')).dt.days
        df['cycle_28_day'] = df['days_since_epoch'] % 28
        df = df.sort_values(['athlete_display_name', 'game_date']).reset_index(drop=True)
        return df

    def calculate_weighted_baselines(self, df, recent_years_weight=3):
        """Calculate performance baselines with more weight on recent years"""
        player_baselines = {}

        for player in df['athlete_display_name'].unique():
            player_data = df[df['athlete_display_name'] == player].copy()
            max_year = player_data['Year'].max()
            player_data['year_weight'] = player_data['Year'].apply(
                lambda x: recent_years_weight if x >= max_year - 2 else 1
            )
            total_weight = player_data['year_weight'].sum()

            baseline_points = (player_data['points'] * player_data['year_weight']).sum() / total_weight
            baseline_shooting = (player_data['shooting_pct'] * player_data['year_weight']).sum() / total_weight
            baseline_minutes = (player_data['minutes'] * player_data['year_weight']).sum() / total_weight

            player_baselines[player] = {
                'points': baseline_points,
                'shooting_pct': baseline_shooting,
                'minutes': baseline_minutes,
                'games': len(player_data),
                'seasons': player_data['Year'].nunique()
            }

        return player_baselines

    def detect_recurring_dips(self, df, player_baselines, dip_threshold=0.15):
        """Detect recurring performance dips using pattern recognition"""
        results = {}

        for player in df['athlete_display_name'].unique():
            player_data = df[df['athlete_display_name'] == player].copy()
            baseline = player_baselines[player]

            points_threshold = baseline['points'] * (1 - dip_threshold)
            shooting_threshold = baseline['shooting_pct'] * (1 - dip_threshold)

            player_data['is_dip'] = (
                    (player_data['points'] < points_threshold) &
                    (player_data['shooting_pct'] < shooting_threshold)
            )

            dip_games = player_data[player_data['is_dip']]

            if len(dip_games) > 5:
                day_dip_counts = dip_games['day_of_month'].value_counts().sort_index()
                cycle_dip_counts = dip_games['cycle_28_day'].value_counts().sort_index()
                month_dip_counts = dip_games['month'].value_counts().sort_index()

                total_games = len(player_data)
                dip_rate = len(dip_games) / total_games

                day_clusters = self.find_clusters(day_dip_counts, window=7)
                cycle_clusters = self.find_clusters(cycle_dip_counts, window=5)

                results[player] = {
                    'total_games': total_games,
                    'dip_games': len(dip_games),
                    'dip_rate': dip_rate,
                    'seasons_analyzed': baseline['seasons'],
                    'day_pattern': day_dip_counts,
                    'cycle_pattern': cycle_dip_counts,
                    'month_pattern': month_dip_counts,
                    'day_clusters': day_clusters,
                    'cycle_clusters': cycle_clusters,
                    'confidence': 'High' if baseline['seasons'] >= 5 else 'Medium' if baseline[
                                                                                          'seasons'] >= 3 else 'Low'
                }

        return results

    def find_clusters(self, pattern_counts, window=5):
        """Find clusters of high activity in patterns"""
        if len(pattern_counts) < window:
            return []

        clusters = []
        threshold = pattern_counts.mean() + pattern_counts.std()

        for i in range(len(pattern_counts) - window + 1):
            window_sum = pattern_counts.iloc[i:i + window].sum()
            if window_sum > threshold:
                start_day = pattern_counts.index[i]
                end_day = pattern_counts.index[i + window - 1]
                clusters.append({
                    'start': start_day,
                    'end': end_day,
                    'intensity': window_sum
                })

        return clusters

    def generate_2025_predictions(self, player_data, baseline, results, custom_dates=None):
        """Generate predictions for May-September 2025 season"""
        if custom_dates:
            prediction_dates = [pd.Timestamp(date) for date in custom_dates]
        else:
            start_date = pd.Timestamp('2025-05-15')
            end_date = pd.Timestamp('2025-09-15')
            prediction_dates = pd.date_range(start=start_date, end=end_date, freq='3D')

        predictions = []
        for date in prediction_dates:
            day_of_month = date.day
            month = date.month
            days_since_epoch = (date - pd.Timestamp('2009-01-01')).days
            cycle_28_day = days_since_epoch % 28

            day_dips = results['day_pattern']
            cycle_dips = results['cycle_pattern']
            month_dips = results['month_pattern']

            day_total = player_data[player_data['day_of_month'] == day_of_month].shape[0]
            cycle_total = player_data[player_data['cycle_28_day'] == cycle_28_day].shape[0]
            month_total = player_data[player_data['month'] == month].shape[0]

            day_dip_rate = day_dips.get(day_of_month, 0) / max(day_total, 1)
            cycle_dip_rate = cycle_dips.get(cycle_28_day, 0) / max(cycle_total, 1)
            month_dip_rate = month_dips.get(month, 0) / max(month_total, 1)

            combined_dip_prob = (day_dip_rate * 0.4 + cycle_dip_rate * 0.4 + month_dip_rate * 0.2)

            # More granular performance categorization
            if combined_dip_prob > 0.35:
                predicted_points = baseline['points'] * 0.65
                performance_category = 'Very Bad Game'
            elif combined_dip_prob > 0.25:
                predicted_points = baseline['points'] * 0.75
                performance_category = 'Bad Game'
            elif combined_dip_prob > 0.15:
                predicted_points = baseline['points'] * 0.85
                performance_category = 'Below Average'
            elif combined_dip_prob > 0.05:
                predicted_points = baseline['points'] * 0.95
                performance_category = 'Average Game'
            else:
                predicted_points = baseline['points'] * 1.05
                performance_category = 'Good Game'

            predictions.append({
                'date': date,
                'predicted_points': predicted_points,
                'dip_probability': combined_dip_prob,
                'category': performance_category,
                'day_of_month': day_of_month,
                'cycle_day': cycle_28_day
            })

        return pd.DataFrame(predictions)

    def get_predictions_only(self, df, player_name, player_baselines, dip_results):
        """Get predictions for custom dates - no visualization"""
        player_data = df[df['athlete_display_name'] == player_name].copy()

        if player_name not in dip_results:
            return None

        baseline = player_baselines[player_name]
        results = dip_results[player_name]

        predictions_2025 = self.generate_2025_predictions(player_data, baseline, results, self.custom_game_dates)

        # Store results for printing instead of printing directly
        self.prediction_results = {
            'player_name': player_name,
            'predictions': predictions_2025
        }

        return predictions_2025

    def analyze_player(self, player_name, custom_dates=None):
        """Analyze a specific player with optional custom dates - minimal output"""
        self.focus_player = player_name
        if custom_dates:
            self.custom_game_dates = custom_dates

        df = self.load_all_player_data(current_players_only=True)

        # Check if player exists - skip if not found
        available_players = df['athlete_display_name'].unique()
        if player_name not in available_players:
            return None

        player_baselines = self.calculate_weighted_baselines(df)
        dip_results = self.detect_recurring_dips(df, player_baselines)

        if player_name in dip_results:
            predictions = self.get_predictions_only(df, player_name, player_baselines, dip_results)
            return predictions
        else:
            return None


def predict(player):
    analyzer = WNBACyclicalPatternDetector()
    results = analyzer.analyze_player(player["name"], [player["date"]])

    if results is not None and hasattr(analyzer, 'prediction_results'):
        pred_df = analyzer.prediction_results.get('predictions')
        if pred_df is not None and not pred_df.empty:
            predicted_points = int(round(float(pred_df['predicted_points'].iloc[0])))
            over_line = float(player['over_line'])
            under_line = float(player['under_line'])
            bet = "OVER" if predicted_points > over_line else "UNDER"

            return {
                "predicted_points": predicted_points,
                "bet": bet,
                "over_line": over_line,
                "under_line": under_line
            }

    return None


"""
if __name__ == "__main__":
    with open("props.json", "r") as f:
        data = json.load(f)

    for player in data["players"]:
        pred = predict(player)
        if pred:
            print(f"{player['name']} predicted points: {pred['predicted_points']:.1f}")
            print(f"Bet: {pred['bet']}, Over line: {pred['over_line']}, Under line: {pred['under_line']}")
            print()
"""
