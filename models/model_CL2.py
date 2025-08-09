import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


def load_player_data(player_name):
    """Load and combine all historical data for a specific player"""
    data_files = {
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

    all_data = []

    try:
        for year, file_path in data_files.items():
            try:
                df = pd.read_csv(file_path)

                # Try multiple matching approaches
                # 1. Exact match
                player_df = df[df['athlete_display_name'] == player_name]

                # 2. If no exact match, try case-insensitive contains
                if player_df.empty:
                    player_df = df[df['athlete_display_name'].str.contains(player_name, case=False, na=False)]

                # 3. If still no match, try partial matching (last name)
                if player_df.empty and ' ' in player_name:
                    last_name = player_name.split()[-1]
                    player_df = df[df['athlete_display_name'].str.contains(last_name, case=False, na=False)]

                    # If multiple players match last name, try to get exact match
                    if len(player_df) > 1:
                        exact_match = df[df['athlete_display_name'].str.contains(player_name, case=False, na=False)]
                        if not exact_match.empty:
                            player_df = exact_match

                if not player_df.empty:
                    all_data.append(player_df)

            except FileNotFoundError:
                continue
            except Exception as e:
                continue

        if all_data:
            player_data = pd.concat(all_data, ignore_index=True)
            player_data['game_date'] = pd.to_datetime(player_data['game_date'])
            player_data = player_data.sort_values('game_date').reset_index(drop=True)

            # Calculate field goal percentage
            player_data['fg_pct'] = np.where(
                player_data['field_goals_attempted'] > 0,
                player_data['field_goals_made'] / player_data['field_goals_attempted'],
                0
            )

            # Remove rows with NaN values in critical columns
            critical_columns = ['points', 'field_goals_made', 'field_goals_attempted', 'game_date']
            player_data = player_data.dropna(subset=critical_columns)

            # If after cleaning we have no data, return None
            if len(player_data) == 0:
                return None

            return player_data
        else:
            return None

    except Exception as e:
        return None


def calculate_rolling_averages(player_data, window=5):
    """Calculate rolling averages for performance metrics"""
    if player_data is None:
        return player_data

    player_data['points_rolling'] = player_data['points'].rolling(window=window, min_periods=1).mean()
    player_data['fg_pct_rolling'] = player_data['fg_pct'].rolling(window=window, min_periods=1).mean()
    player_data['minutes_rolling'] = player_data['minutes'].rolling(window=window, min_periods=1).mean()

    return player_data


def identify_performance_dips(player_data):
    """Identify games where player significantly underperformed"""
    if player_data is None:
        return player_data

    # Calculate z-scores for key metrics
    player_data['points_zscore'] = (
            (player_data['points'] - player_data['points'].mean()) /
            player_data['points'].std()
    )

    player_data['fg_pct_zscore'] = (
            (player_data['fg_pct'] - player_data['fg_pct'].mean()) /
            player_data['fg_pct'].std()
    )

    # Combined performance score (lower is worse)
    player_data['performance_score'] = (
                                               player_data['points_zscore'] + player_data[
                                           'fg_pct_zscore']
                                       ) / 2

    # Identify significant dips (bottom 25%)
    dip_threshold = player_data['performance_score'].quantile(0.25)
    player_data['is_dip'] = player_data['performance_score'] < dip_threshold

    return player_data


def find_cyclical_patterns(player_data):
    """Analyze for roughly monthly patterns in performance dips"""
    if player_data is None:
        return {}

    dip_games = player_data[player_data['is_dip'] == True].copy()

    if len(dip_games) < 3:
        return {}

    # Calculate days between consecutive dips
    dip_games = dip_games.sort_values('game_date')
    dip_games['days_since_last_dip'] = dip_games['game_date'].diff().dt.days

    # Look for patterns around 25-35 day cycles (accounting for game schedule)
    cycle_intervals = dip_games['days_since_last_dip'].dropna()

    # Find intervals that suggest monthly cycles
    monthly_cycles = cycle_intervals[(cycle_intervals >= 20) & (cycle_intervals <= 40)]

    if len(monthly_cycles) > 0:
        avg_cycle = monthly_cycles.mean()
        cycle_std = monthly_cycles.std()

        cycle_patterns = {
            'average_cycle_days': avg_cycle,
            'cycle_std': cycle_std,
            'cycle_count': len(monthly_cycles),
            'last_dip_date': dip_games['game_date'].iloc[-1],
            'dip_games': dip_games
        }
        return cycle_patterns
    else:
        return {}


def predict_next_dip_window(cycle_patterns):
    """Predict when the next performance dip might occur"""
    if not cycle_patterns:
        return None

    last_dip = cycle_patterns['last_dip_date']
    avg_cycle = cycle_patterns['average_cycle_days']
    cycle_std = cycle_patterns['cycle_std']

    # Predict next dip window
    predicted_date = last_dip + timedelta(days=avg_cycle)
    window_start = predicted_date - timedelta(days=cycle_std)
    window_end = predicted_date + timedelta(days=cycle_std)

    return {
        'predicted_date': predicted_date,
        'window_start': window_start,
        'window_end': window_end,
        'confidence': min(100, cycle_patterns['cycle_count'] * 20)  # Max 100%
    }


def is_in_predicted_dip_window(game_date, cycle_patterns):
    """Check if a given date falls within a predicted performance dip window"""
    prediction = predict_next_dip_window(cycle_patterns)
    if not prediction:
        return False, 0

    game_date = pd.to_datetime(game_date)

    if prediction['window_start'] <= game_date <= prediction['window_end']:
        # Calculate how close to the center of the window
        center = prediction['predicted_date']
        days_from_center = abs((game_date - center).days)
        window_size = (prediction['window_end'] - prediction['window_start']).days

        # Proximity score (1.0 = exactly at center, 0.0 = at edge of window)
        proximity = 1.0 - (days_from_center / (window_size / 2))
        return True, proximity

    return False, 0


def calculate_betting_confidence(player_data, cycle_patterns, predicted_points, over_line, under_line, 
                               in_dip_window, proximity):
    """Calculate confidence score for betting and determine bet amount"""
    confidence_factors = {}
    
    # 1. Pattern strength (0-1): Much stricter requirements for cyclical patterns
    if cycle_patterns and cycle_patterns['cycle_count'] >= 3:
        # Require more cycles for high confidence, penalize high variance
        cycle_consistency = 1.0 / (1.0 + cycle_patterns['cycle_std'] / cycle_patterns['average_cycle_days'])
        pattern_strength = min(1.0, (cycle_patterns['cycle_count'] / 10) * cycle_consistency)  # Max at 10+ cycles
        confidence_factors['pattern_strength'] = pattern_strength * 0.4  # Increased weight
    else:
        confidence_factors['pattern_strength'] = 0
    
    # 2. Proximity to predicted dip (0-1): Must be very close to center for high confidence
    if in_dip_window:
        # Use exponential scaling - only very close to center gets high scores
        proximity_score = proximity ** 2  # Square it to make it more selective
        confidence_factors['proximity'] = proximity_score * 0.3
    else:
        confidence_factors['proximity'] = 0
    
    # 3. Prediction margin (0-1): Much stricter margin requirements
    line_used = over_line if predicted_points > over_line else under_line
    margin = abs(predicted_points - line_used)
    player_avg = player_data['points'].mean()
    player_std = player_data['points'].std()
    
    # Require larger margins for confidence, normalize more strictly
    if player_std > 0 and margin > 1.0:  # Must have at least 1 point margin
        # Require margin to be significant portion of player's standard deviation
        normalized_margin = min(1.0, (margin - 1.0) / (player_std * 1.5))  # Subtract baseline 1 pt
        confidence_factors['margin'] = normalized_margin * 0.2
    else:
        confidence_factors['margin'] = 0
    
    # 4. Dip game performance vs line: How well do dip games perform relative to the line?
    if cycle_patterns and len(cycle_patterns['dip_games']) > 0:
        dip_performances = cycle_patterns['dip_games']['points']
        line_used = under_line if predicted_points < over_line else over_line
        
        # What percentage of dip games would have hit our bet?
        if predicted_points < over_line:  # We're betting UNDER
            hit_rate = (dip_performances <= under_line).mean()
        else:  # We're betting OVER
            hit_rate = (dip_performances >= over_line).mean()
        
        # Only confident if high hit rate
        confidence_factors['historical_accuracy'] = max(0, (hit_rate - 0.6) / 0.4) * 0.1  # Need 60%+ hit rate
    else:
        confidence_factors['historical_accuracy'] = 0
    
    # Calculate total confidence score
    total_confidence = sum(confidence_factors.values())
    
    # Much stricter thresholds for bet amounts
    if total_confidence >= 0.85:  # Very rare, need nearly perfect conditions
        amount = 5
    elif total_confidence >= 0.70:  # High confidence, strong pattern + close proximity + good margin
        amount = 4
    elif total_confidence >= 0.50:  # Moderate confidence, decent pattern + some proximity
        amount = 3
    elif total_confidence >= 0.30:  # Low-moderate confidence, weak pattern or poor timing
        amount = 2
    elif total_confidence >= 0.15:  # Minimal confidence, very weak signal
        amount = 1
    else:
        amount = None  # Not confident enough to bet
    
    return amount, total_confidence, confidence_factors


def predict(player):
    """
    Main prediction function
    player should contain:
    - 'name': player name
    - 'game_date': date of the game to predict
    - 'over_line': betting over line
    - 'under_line': betting under line
    """
    player_name = player['name']
    game_date = player['date']
    over_line = float(player['over_line'])
    under_line = float(player['under_line'])

    print(f"\n--- CL2: Running prediction for {player['name']} on {player['date']} ---")

    # Load and analyze player data
    player_data = load_player_data(player_name)
    if player_data is None:
        print(f"{player_name} not found in data")
        print(f"Prediction not generated for {player_name}")
        print()
        return None  # Skip player if no data

    # Calculate performance metrics
    player_data = calculate_rolling_averages(player_data)
    player_data = identify_performance_dips(player_data)
    cycle_patterns = find_cyclical_patterns(player_data)

    # Check if we found a clear menstrual pattern
    if not cycle_patterns or cycle_patterns['cycle_count'] < 3:
        print(f"{player_name} not in dip results")
        print(f"Prediction not generated for {player_name}")
        print()
        return None  # Skip player if no clear cyclical pattern

    # Get baseline prediction (season average)
    season_avg_points = player_data['points'].mean()
    recent_avg_points = player_data['points'].tail(10).mean()  # Last 10 games
    baseline_prediction = (season_avg_points + recent_avg_points) / 2

    # Check if game date is in predicted dip window
    in_dip_window, proximity = is_in_predicted_dip_window(game_date, cycle_patterns)

    if in_dip_window and cycle_patterns:
        # Adjust prediction based on typical dip performance
        dip_games = cycle_patterns['dip_games']
        avg_dip_performance = dip_games['points'].mean()

        # Weight the adjustment by proximity to dip center and pattern confidence
        confidence = min(1.0, cycle_patterns['cycle_count'] / 5)  # Max confidence at 5+ cycles
        adjustment_factor = proximity * confidence * 0.7  # Max 70% adjustment

        predicted_points = (baseline_prediction * (1 - adjustment_factor) +
                            avg_dip_performance * adjustment_factor)

        performance_note = "Period Game" if proximity > 0.7 else "Below Average"

    else:
        # No cyclical pattern or outside dip window
        predicted_points = baseline_prediction

        # Determine performance note based on recent form
        recent_performance = player_data['performance_score'].tail(5).mean()
        if recent_performance > 0.5:
            performance_note = "Good Game"
        elif recent_performance > 0:
            performance_note = "Average Game"
        else:
            performance_note = "Below Average"

    # Make betting decision
    bet = "OVER" if predicted_points > over_line else "UNDER"

    # Calculate betting confidence and amount
    amount, confidence_score, confidence_factors = calculate_betting_confidence(
        player_data, cycle_patterns, predicted_points, over_line, under_line, 
        in_dip_window, proximity
    )

    print(f"Prediction successful for {player_name}: {round(predicted_points)} pts")
    if amount:
        print(f"Betting confidence: {confidence_score:.2f}, Amount: ${amount}")
    else:
        print(f"Betting confidence too low: {confidence_score:.2f}, No bet recommended")
    print()

    return {
        "predicted_points": round(predicted_points),
        "bet": bet,
        "over_line": over_line,
        "under_line": under_line,
        "note": performance_note,
        "amount": amount
    }


"""
# Testing block
if __name__ == "__main__":
    import json

    # Debug: First check if we can load any data at all
    try:
        test_df = pd.read_csv("playerboxes/player_box_2024.csv")
        sample_players = test_df['athlete_display_name'].unique()[:5]
        print(f"Sample players in data: {sample_players}")
    except:
        print("Could not load sample data file")

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
