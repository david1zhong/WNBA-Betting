import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import json
import warnings
import psycopg2
from sqlalchemy import create_engine, text
from collections import defaultdict
from dotenv import load_dotenv
import os

warnings.filterwarnings('ignore')

# Database connection setup
USER = os.getenv("DB_USER"),
PASSWORD = os.getenv("DB_PASSWORD"),
HOST = os.getenv("DB_HOST"),
PORT = 6543,
DBNAME = os.getenv("DB_NAME")

DATABASE_URL = f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}?sslmode=require"
engine = create_engine(DATABASE_URL)


class WNBALearningPatternDetector:
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
        self.learning_insights = {}

    def get_historical_betting_data(self, player_name):
        """Fetch historical betting results for a specific player"""
        try:
            with engine.connect() as connection:
                # Query to get all bets for this player from previous models
                query = text("""
                    SELECT 
                        model_name,
                        bet as predicted_bet,
                        result as actual_result,
                        predicted_pts as predicted_points,
                        actual_pts as actual_points,
                        over_line,
                        under_line,
                        date
                    FROM predictions 
                    WHERE LOWER(TRIM(player_name)) = LOWER(TRIM(:player_name))
                    AND model_name IN ('model_CL1', 'model_CL2')
                    ORDER BY date DESC
                """)

                result = connection.execute(query, {"player_name": player_name})
                df = pd.DataFrame(result.fetchall(), columns=result.keys())

                print(f"DEBUG: Found {len(df)} historical bets for {player_name}")
                if not df.empty:
                    print(
                        f"DEBUG: Bet distribution - OVER: {sum(df['predicted_bet'] == 'OVER')}, UNDER: {sum(df['predicted_bet'] == 'UNDER')}")
                    print(
                        f"DEBUG: Result distribution - WON: {sum(df['actual_result'] == 'WON')}, LOST: {sum(df['actual_result'] == 'LOST')}")

                return df if not df.empty else None

        except Exception as e:
            print(f"Error fetching historical data for {player_name}: {e}")
            return None

    def analyze_model_performance(self, historical_data):
        """Analyze the performance patterns of previous models"""
        if historical_data is None or historical_data.empty:
            return None

        # Count correct bets properly by counting WON results
        won_bets = historical_data[historical_data['actual_result'] == 'WON']
        lost_bets = historical_data[historical_data['actual_result'] == 'LOST']

        # Only count games that have results (WON or LOST)
        games_with_results = len(won_bets) + len(lost_bets)
        total_bets = len(historical_data)
        correct_bets = len(won_bets)

        analysis = {
            'total_bets': total_bets,
            'games_with_results': games_with_results,
            'correct_bets': correct_bets,
            'accuracy': 0,
            'over_bets': 0,
            'under_bets': 0,
            'over_success_rate': 0,
            'under_success_rate': 0,
            'over_wins': 0,
            'under_wins': 0,
            'prediction_bias': None,
            'reality_bias': None,
            'learning_recommendation': None,
            'best_model': None,
            'model_analysis': {}
        }

        if games_with_results > 0:
            analysis['accuracy'] = correct_bets / games_with_results

            # Analyze each model separately - FIX #2
            model_performance = {}
            for model in ['model_CL1', 'model_CL2']:
                model_data = historical_data[historical_data['model_name'] == model]
                if len(model_data) > 0:
                    model_won = model_data[model_data['actual_result'] == 'WON']
                    model_lost = model_data[model_data['actual_result'] == 'LOST']
                    model_games_with_results = len(model_won) + len(model_lost)

                    if model_games_with_results > 0:
                        model_accuracy = len(model_won) / model_games_with_results
                        model_performance[model] = {
                            'total_bets': len(model_data),
                            'games_with_results': model_games_with_results,
                            'wins': len(model_won),
                            'accuracy': model_accuracy,
                            'over_bets': len(model_data[model_data['predicted_bet'] == 'OVER']),
                            'under_bets': len(model_data[model_data['predicted_bet'] == 'UNDER']),
                            'over_wins': len(model_won[model_won['predicted_bet'] == 'OVER']),
                            'under_wins': len(model_won[model_won['predicted_bet'] == 'UNDER'])
                        }

            analysis['model_analysis'] = model_performance

            # Determine best performing model
            if len(model_performance) >= 2:
                best_model = max(model_performance.keys(),
                                 key=lambda x: model_performance[x]['accuracy'] if model_performance[x][
                                                                                       'games_with_results'] >= 3 else 0)
                analysis['best_model'] = best_model
                print(
                    f"DEBUG: Best performing model: {best_model} with accuracy {model_performance[best_model]['accuracy']:.2f}")

            # Analyze bet distribution
            over_bets = historical_data[historical_data['predicted_bet'] == 'OVER']
            under_bets = historical_data[historical_data['predicted_bet'] == 'UNDER']

            analysis['over_bets'] = len(over_bets)
            analysis['under_bets'] = len(under_bets)

            # Calculate win rates for each bet type (only count games with results)
            over_results = over_bets[over_bets['actual_result'].isin(['WON', 'LOST'])]
            under_results = under_bets[under_bets['actual_result'].isin(['WON', 'LOST'])]

            if len(over_results) > 0:
                analysis['over_wins'] = len(over_bets[over_bets['actual_result'] == 'WON'])
                analysis['over_success_rate'] = analysis['over_wins'] / len(over_results)

            if len(under_results) > 0:
                analysis['under_wins'] = len(under_bets[under_bets['actual_result'] == 'WON'])
                analysis['under_success_rate'] = analysis['under_wins'] / len(under_results)

            # FIX #1: Only apply bias if there's a clear majority (70%+)
            total_games_with_results = analysis['games_with_results']
            over_games_with_results = len(over_results)
            under_games_with_results = len(under_results)

            # Check for betting bias - only if one bet type is 70%+ of total
            if under_games_with_results >= total_games_with_results * 0.7:
                analysis['prediction_bias'] = 'UNDER'
                print(
                    f"DEBUG: Strong UNDER betting bias detected: {under_games_with_results}/{total_games_with_results} = {under_games_with_results / total_games_with_results:.2f}")
            elif over_games_with_results >= total_games_with_results * 0.7:
                analysis['prediction_bias'] = 'OVER'
                print(
                    f"DEBUG: Strong OVER betting bias detected: {over_games_with_results}/{total_games_with_results} = {over_games_with_results / total_games_with_results:.2f}")
            else:
                analysis['prediction_bias'] = 'BALANCED'
                print(
                    f"DEBUG: Balanced betting pattern: OVER {over_games_with_results}, UNDER {under_games_with_results}")

            # Analyze what actually wins more often
            won_over_bets = won_bets[won_bets['predicted_bet'] == 'OVER']
            won_under_bets = won_bets[won_bets['predicted_bet'] == 'UNDER']

            if len(won_over_bets) > len(won_under_bets) * 1.5:
                analysis['reality_bias'] = 'OVER'
            elif len(won_under_bets) > len(won_over_bets) * 1.5:
                analysis['reality_bias'] = 'UNDER'
            else:
                analysis['reality_bias'] = 'BALANCED'

            print(
                f"DEBUG: Analysis - Under games with results: {len(under_results)}, Under wins: {analysis['under_wins']}, Under success: {analysis['under_success_rate']:.2f}")
            print(
                f"DEBUG: Analysis - Over games with results: {len(over_results)}, Over wins: {analysis['over_wins']}, Over success: {analysis['over_success_rate']:.2f}")
            print(f"DEBUG: Total correct bets: {analysis['correct_bets']}/{analysis['games_with_results']}")

            # FIX #3: Apply learning only when there's a strong bias AND poor performance
            if analysis['games_with_results'] >= 5:
                # Strong learning with 5+ games
                if (analysis['prediction_bias'] == 'UNDER' and
                        analysis['under_success_rate'] <= 0.30):  # 30% or worse with majority UNDER bets
                    analysis['learning_recommendation'] = 'FORCE_OVER'
                    print(
                        f"DEBUG: FORCING OVER due to majority UNDER bets with poor performance ({analysis['under_success_rate']:.2f})")

                elif (analysis['prediction_bias'] == 'OVER' and
                      analysis['over_success_rate'] <= 0.30):  # 30% or worse with majority OVER bets
                    analysis['learning_recommendation'] = 'FORCE_UNDER'
                    print(
                        f"DEBUG: FORCING UNDER due to majority OVER bets with poor performance ({analysis['over_success_rate']:.2f})")

                elif (analysis['prediction_bias'] == 'UNDER' and
                      analysis['under_success_rate'] <= 0.40):  # 40% or worse with majority UNDER bets
                    analysis['learning_recommendation'] = 'FAVOR_OVER'
                    print(
                        f"DEBUG: FAVORING OVER due to majority UNDER bets with mediocre performance ({analysis['under_success_rate']:.2f})")

                elif (analysis['prediction_bias'] == 'OVER' and
                      analysis['over_success_rate'] <= 0.40):  # 40% or worse with majority OVER bets
                    analysis['learning_recommendation'] = 'FAVOR_UNDER'
                    print(
                        f"DEBUG: FAVORING UNDER due to majority OVER bets with mediocre performance ({analysis['over_success_rate']:.2f})")

                else:
                    analysis['learning_recommendation'] = 'NEUTRAL'
                    print(f"DEBUG: No strong learning recommendation - performance not poor enough or no bias")

            elif analysis['games_with_results'] >= 3:
                # Weaker learning with 3-4 games, only for very clear patterns
                if (analysis['prediction_bias'] == 'UNDER' and
                        analysis['under_success_rate'] == 0):  # 0% success with majority UNDER bets
                    analysis['learning_recommendation'] = 'FAVOR_OVER'
                    print(f"DEBUG: FAVORING OVER due to 0% UNDER success with majority UNDER bets")
                elif (analysis['prediction_bias'] == 'OVER' and
                      analysis['over_success_rate'] == 0):  # 0% success with majority OVER bets
                    analysis['learning_recommendation'] = 'FAVOR_UNDER'
                    print(f"DEBUG: FAVORING UNDER due to 0% OVER success with majority OVER bets")
                else:
                    analysis['learning_recommendation'] = 'NEUTRAL'
            else:
                # Not enough data for meaningful learning
                analysis['learning_recommendation'] = 'NEUTRAL'
                print(
                    f"DEBUG: Not enough data ({analysis['games_with_results']} games with results) for learning recommendations")

            print(f"DEBUG: Final learning recommendation: {analysis['learning_recommendation']}")

        return analysis

    def calculate_learning_adjustment(self, player_name):
        """Calculate adjustment factors based on historical performance"""
        print(f"DEBUG: Calculating learning adjustment for {player_name}")
        historical_data = self.get_historical_betting_data(player_name)

        if historical_data is None:
            print(f"DEBUG: No historical data found for {player_name}")
            return {
                'confidence_multiplier': 1.0,
                'bias_adjustment': 0.0,
                'recommendation': 'NEUTRAL',
                'learning_strength': 0.0,
                'best_model': None
            }

        performance_analysis = self.analyze_model_performance(historical_data)

        if performance_analysis is None:
            print(f"DEBUG: No performance analysis for {player_name}")
            return {
                'confidence_multiplier': 1.0,
                'bias_adjustment': 0.0,
                'recommendation': 'NEUTRAL',
                'learning_strength': 0.0,
                'best_model': None
            }

        # Calculate learning strength based on sample size (games with results)
        games_with_results = performance_analysis['games_with_results']
        if games_with_results >= 5:
            learning_strength = 1.0  # Full strength
        elif games_with_results >= 3:
            learning_strength = 0.6  # Moderate strength
        else:
            learning_strength = 0.2  # Very weak strength

        print(f"DEBUG: Learning strength: {learning_strength} (based on {games_with_results} games with results)")

        # Adjust bias based on recommendation and learning strength
        bias_adjustment = 0.0
        recommendation = performance_analysis['learning_recommendation']

        if recommendation == 'FORCE_OVER':
            bias_adjustment = 8.0 * learning_strength  # Strong OVER bias
            print(f"DEBUG: FORCING OVER with bias: {bias_adjustment}")

        elif recommendation == 'FORCE_UNDER':
            bias_adjustment = -8.0 * learning_strength  # Strong UNDER bias
            print(f"DEBUG: FORCING UNDER with bias: {bias_adjustment}")

        elif recommendation == 'FAVOR_OVER':
            bias_adjustment = 3.0 * learning_strength
            print(f"DEBUG: FAVOR OVER bias: {bias_adjustment}")

        elif recommendation == 'FAVOR_UNDER':
            bias_adjustment = -3.0 * learning_strength
            print(f"DEBUG: FAVOR UNDER bias: {bias_adjustment}")

        # Confidence multiplier based on data quality
        if games_with_results >= 5:
            if abs(performance_analysis['over_success_rate'] - performance_analysis['under_success_rate']) > 0.3:
                confidence_multiplier = 2.0
            else:
                confidence_multiplier = 1.5
        else:
            confidence_multiplier = 1.0

        adjustment = {
            'confidence_multiplier': confidence_multiplier,
            'bias_adjustment': bias_adjustment,
            'recommendation': recommendation,
            'learning_strength': learning_strength,
            'best_model': performance_analysis['best_model']
        }

        print(f"DEBUG: Final adjustment for {player_name}: {adjustment}")

        self.learning_insights[player_name] = {
            'historical_data': performance_analysis,
            'learning_adjustment': adjustment
        }

        return adjustment

    def get_current_players(self):
        """Get list of players who played in 2025"""
        try:
            df_2025 = pd.read_csv(self.years[2025])
            current_players = set(df_2025['athlete_display_name'].unique())
            return current_players
        except Exception:
            return set()

    def load_all_player_data(self, current_players_only=True):
        """Load data for all years, optionally filtering for current players only"""
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
        df = df.dropna(subset=[
            'game_date',
            'field_goals_made',
            'field_goals_attempted',
            'points',
            'minutes',
            'athlete_display_name'
        ])

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

    def get_performance_buckets(self, player_data_2025):
        """Get actual point ranges for different performance categories from 2025 season only"""
        if len(player_data_2025) == 0:
            return None

        points = player_data_2025['points'].values

        # Remove outliers (bottom and top 5%)
        p5, p95 = np.percentile(points, [5, 95])
        filtered_points = points[(points >= p5) & (points <= p95)]

        if len(filtered_points) < 5:  # Need at least 5 games
            filtered_points = points

        # Create performance buckets based on percentiles
        buckets = {
            'very_bad': np.percentile(filtered_points, 15),  # Bottom 15%
            'bad': np.percentile(filtered_points, 30),  # 15th-30th percentile
            'below_avg': np.percentile(filtered_points, 45),  # 30th-45th percentile
            'average': np.percentile(filtered_points, 55),  # 45th-55th percentile (median area)
            'good': np.percentile(filtered_points, 75),  # 55th-75th percentile
            'very_good': np.percentile(filtered_points, 90)  # Top 25%
        }

        return buckets

    def generate_2025_predictions(self, player_data, baseline, results, custom_dates=None, learning_adj=None):
        """Generate predictions for May-September 2025 season using actual 2025 performance data and learning"""
        if custom_dates:
            prediction_dates = [pd.Timestamp(date) for date in custom_dates]
        else:
            start_date = pd.Timestamp('2025-05-15')
            end_date = pd.Timestamp('2025-09-15')
            prediction_dates = pd.date_range(start=start_date, end=end_date, freq='3D')

        # Get 2025 season data only for realistic point predictions
        player_data_2025 = player_data[player_data['Year'] == 2025].copy()
        performance_buckets = self.get_performance_buckets(player_data_2025)

        # Fallback to baseline if no 2025 data
        if performance_buckets is None:
            performance_buckets = {
                'very_bad': baseline['points'] * 0.4,
                'bad': baseline['points'] * 0.6,
                'below_avg': baseline['points'] * 0.8,
                'average': baseline['points'] * 1.0,
                'good': baseline['points'] * 1.2,
                'very_good': baseline['points'] * 1.4
            }

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

            # Apply learning adjustment to dip probability
            if learning_adj:
                # Adjust the dip probability based on learned bias
                if learning_adj['bias_adjustment'] > 0:  # Favor OVER
                    combined_dip_prob *= (1 - learning_adj['bias_adjustment'] * 0.1)
                elif learning_adj['bias_adjustment'] < 0:  # Favor UNDER
                    combined_dip_prob *= (1 + abs(learning_adj['bias_adjustment']) * 0.1)

            # LEARNING-ENHANCED: Adjusted thresholds based on learning
            base_thresholds = [0.4, 0.3, 0.2, 0.12, 0.05]
            if learning_adj and learning_adj['learning_strength'] > 0.5:
                # Adjust thresholds based on learning
                threshold_adjustment = learning_adj['bias_adjustment'] * 0.05
                adjusted_thresholds = [max(0.01, t + threshold_adjustment) for t in base_thresholds]
            else:
                adjusted_thresholds = base_thresholds

            if combined_dip_prob > adjusted_thresholds[0]:
                predicted_points = performance_buckets['very_bad']
                performance_category = 'Very Bad Game'
            elif combined_dip_prob > adjusted_thresholds[1]:
                predicted_points = performance_buckets['bad']
                performance_category = 'Bad Game'
            elif combined_dip_prob > adjusted_thresholds[2]:
                predicted_points = performance_buckets['below_avg']
                performance_category = 'Below Average'
            elif combined_dip_prob > adjusted_thresholds[3]:
                predicted_points = performance_buckets['average']
                performance_category = 'Average Game'
            elif combined_dip_prob > adjusted_thresholds[4]:
                predicted_points = performance_buckets['good']
                performance_category = 'Good Game'
            else:
                predicted_points = performance_buckets['very_good']
                performance_category = 'Very Good Game'

            predictions.append({
                'date': date,
                'predicted_points': predicted_points,
                'dip_probability': combined_dip_prob,
                'category': performance_category,
                'day_of_month': day_of_month,
                'cycle_day': cycle_28_day
            })

        return pd.DataFrame(predictions)

    def get_predictions_only(self, df, player_name, player_baselines, dip_results, learning_adj):
        """Get predictions for custom dates with learning adjustment"""
        player_data = df[df['athlete_display_name'] == player_name].copy()

        if player_name not in dip_results:
            return None

        baseline = player_baselines[player_name]
        results = dip_results[player_name]

        predictions_2025 = self.generate_2025_predictions(
            player_data, baseline, results, self.custom_game_dates, learning_adj
        )

        # Store results for printing instead of printing directly
        self.prediction_results = {
            'player_name': player_name,
            'predictions': predictions_2025,
            'learning_insights': learning_adj
        }

        return predictions_2025

    def analyze_player(self, player_name, custom_dates=None):
        """Analyze a specific player with learning from historical bets"""
        self.focus_player = player_name
        if custom_dates:
            self.custom_game_dates = custom_dates

        # Calculate learning adjustment FIRST
        learning_adj = self.calculate_learning_adjustment(player_name)

        df = self.load_all_player_data(current_players_only=True)

        # Check if player exists - skip if not found
        available_players = df['athlete_display_name'].unique()
        if player_name not in available_players:
            return None

        player_baselines = self.calculate_weighted_baselines(df)
        dip_results = self.detect_recurring_dips(df, player_baselines)

        if player_name in dip_results:
            predictions = self.get_predictions_only(df, player_name, player_baselines, dip_results, learning_adj)
            return predictions
        else:
            return None

    def calculate_confidence_score(self, learning_adj, dip_probability, performance_category):
        """Calculate confidence score from 1-5 or None - More selective and balanced"""

        # Start with no bet - force the system to prove confidence
        confidence_points = 0

        # STRONG learning signals (only the best get high confidence)
        recommendation = learning_adj.get('recommendation', 'NEUTRAL')
        learning_strength = learning_adj.get('learning_strength', 0)

        if recommendation == 'FORCE_OVER' or recommendation == 'FORCE_UNDER':
            if learning_strength >= 0.8:  # Very strong historical pattern
                confidence_points += 3  # Strong foundation
            elif learning_strength >= 0.6:
                confidence_points += 2
            else:
                confidence_points += 1
        elif recommendation == 'FAVOR_OVER' or recommendation == 'FAVOR_UNDER':
            if learning_strength >= 0.8:
                confidence_points += 2
            elif learning_strength >= 0.6:
                confidence_points += 1
            # If learning_strength < 0.6, no points added

        # Pattern certainty bonus - only for very clear patterns
        if dip_probability > 0.4:  # Very high chance of bad game
            confidence_points += 2
        elif dip_probability > 0.3:  # High chance
            confidence_points += 1
        elif dip_probability < 0.05:  # Very low chance (good game expected)
            confidence_points += 2
        elif dip_probability < 0.1:  # Low chance
            confidence_points += 1

        # Performance category bonus - only for extremes
        if performance_category in ['Very Bad Game', 'Very Good Game']:
            confidence_points += 1

        # Best model bonus - if we have a clearly superior historical model
        if learning_adj.get('best_model') and learning_strength >= 0.6:
            confidence_points += 1

        # Convert confidence points to bet amount with strict thresholds
        if confidence_points >= 6:  # Very confident - rare
            return 5
        elif confidence_points >= 5:  # Confident
            return 4
        elif confidence_points >= 4:  # Moderately confident
            return 3
        elif confidence_points >= 3:  # Slightly confident
            return 2
        elif confidence_points >= 2:  # Minimal confidence
            return 1
        else:
            # Not confident enough to bet
            return None

        # Additional safety check - require minimum learning strength for any bet
        if learning_strength < 0.2:
            return None

        # Debug output
        print(f"DEBUG: Confidence calculation - Points: {confidence_points}, Learning: {learning_strength:.2f}, "
              f"Recommendation: {recommendation}, Dip Prob: {dip_probability:.3f}")

        # Map to final bet amount
        if confidence_points >= 6:  # Very confident - rare (only best patterns)
            final_amount = 5
        elif confidence_points >= 5:  # Confident
            final_amount = 4
        elif confidence_points >= 4:  # Moderately confident
            final_amount = 3
        elif confidence_points >= 3:  # Slightly confident
            final_amount = 2
        elif confidence_points >= 2:  # Minimal confidence
            final_amount = 1
        else:
            final_amount = None

        # Final safety checks
        if learning_strength < 0.3 and confidence_points < 4:
            # Don't bet small amounts on weak learning
            final_amount = None

        if final_amount and confidence_points < 3:
            # Only bet $1-2 if we have some learning strength
            if learning_strength < 0.5:
                final_amount = None

        print(f"DEBUG: Final bet amount: ${final_amount if final_amount else 'None'}")
        return final_amount


def predict(player):
    """Main prediction function that returns the required format"""
    player_name = player["name"]
    game_date = player.get("game_date", player.get("date"))

    print(f"--- Running prediction for {player_name} on {game_date} ---")

    analyzer = WNBALearningPatternDetector()
    results = analyzer.analyze_player(player_name, [game_date])

    if results is not None and hasattr(analyzer, 'prediction_results'):
        pred_df = analyzer.prediction_results.get('predictions')
        learning_insights = analyzer.prediction_results.get('learning_insights', {})

        if pred_df is not None and not pred_df.empty:
            # Get the raw predicted points
            raw_predicted_points = pred_df['predicted_points'].iloc[0]
            dip_probability = pred_df['dip_probability'].iloc[0]
            performance_category = pred_df['category'].iloc[0]

            # Check for NaN values and discard if found
            if pd.isna(raw_predicted_points) or pd.isna(dip_probability):
                print(f"NaN values detected for {player_name}, discarding prediction")
                return None

            # Round to nearest whole number
            predicted_points = round(float(raw_predicted_points))

            over_line = float(player['over_line'])
            under_line = float(player['under_line'])

            print(f"DEBUG: {player_name} - Predicted: {predicted_points} pts, Over: {over_line}, Under: {under_line}")
            print(f"DEBUG: Learning recommendation: {learning_insights.get('recommendation', 'NEUTRAL')}")
            print(f"DEBUG: Best performing model: {learning_insights.get('best_model', 'None')}")
            print(f"DEBUG: Performance category: {performance_category}")

            # Apply learning adjustments to prediction if needed
            adjusted_points = predicted_points
            recommendation = learning_insights.get('recommendation', 'NEUTRAL')

            if recommendation in ['FORCE_OVER', 'FAVOR_OVER']:
                if predicted_points < over_line:
                    adjustment_factor = 1.2 if recommendation == 'FORCE_OVER' else 1.1
                    adjusted_points = max(over_line + 1, round(predicted_points * adjustment_factor))
                    print(
                        f"DEBUG: Adjusted points upward from {predicted_points} to {adjusted_points} for {recommendation}")

            elif recommendation in ['FORCE_UNDER', 'FAVOR_UNDER']:
                if predicted_points > under_line:
                    adjustment_factor = 0.8 if recommendation == 'FORCE_UNDER' else 0.9
                    adjusted_points = min(under_line - 1, round(predicted_points * adjustment_factor))
                    print(
                        f"DEBUG: Adjusted points downward from {predicted_points} to {adjusted_points} for {recommendation}")

            final_predicted_points = adjusted_points

            if final_predicted_points is None:
                return None

            # Simple bet logic: OVER if predicted > over_line, else UNDER
            bet = "OVER" if final_predicted_points > over_line else "UNDER"

            # Apply learning overrides if needed
            if recommendation == 'FORCE_OVER':
                bet = "OVER"
                print(f"DEBUG: FORCING OVER bet due to learning")
            elif recommendation == 'FORCE_UNDER':
                bet = "UNDER"
                print(f"DEBUG: FORCING UNDER bet due to learning")
            elif recommendation == 'FAVOR_OVER' and final_predicted_points >= over_line - 1:
                bet = "OVER"
                print(f"DEBUG: FAVORING OVER bet due to learning")
            elif recommendation == 'FAVOR_UNDER' and final_predicted_points <= under_line + 1:
                bet = "UNDER"
                print(f"DEBUG: FAVORING UNDER bet due to learning")

            # Calculate bet amount (1-5 or None)
            amount = analyzer.calculate_confidence_score(learning_insights, dip_probability, performance_category)

            # Ensure performance note is max 3 words
            performance_note = performance_category

            print(f"Prediction successful for {player_name}: {final_predicted_points} pts")
            if amount:
                print(f"DEBUG: Bet amount: {amount}/5")
            else:
                print(f"DEBUG: Bet amount: None (insufficient data)")

            print()


            return {
                "predicted_points": round(final_predicted_points),
                "bet": bet,
                "over_line": over_line,
                "under_line": under_line,
                "note": performance_note,
                "amount": amount
            }
        else:
            print(f"No prediction data available for {player_name}")
            return None
    else:
        if results is None:
            print(f"{player_name} not in dip results")
        print(f"Prediction not generated for {player_name}")
        return None

"""
# Testing block
if __name__ == "__main__":
    import json


    try:
        with open('props.json', 'r') as f:
            props_data = json.load(f)
        for player in props_data['players']:
            player_info = {
                'name': player['name'],
                'game_date': player['date'],
                'over_line': player['over_line'],
                'under_line': player['under_line']
            }
            result = predict(player_info)


            # Only print if we have a valid prediction (clear menstrual pattern found)
            if result is not None:
                print(f"{player['name']} predicted points: {result['predicted_points']}")
                print(f"Bet: {result['bet']}, Over line: {result['over_line']}, Under line: {result['under_line']}")
                print(f"Bet Amount: ${result['amount']}" if result['amount'] else "No bet recommended")
                print()
    except FileNotFoundError:
        print("props.json file not found")
    except Exception as e:
        print(f"Error: {e}")
"""
