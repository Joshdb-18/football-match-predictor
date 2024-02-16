import pandas as pd
import numpy as np
import ujson as json
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

scaling_factor = 2

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def load_and_preprocess_data():
    data = pd.read_csv("cleaned_matches.csv")

    le_home = LabelEncoder()
    le_away = LabelEncoder()
    data['home_team'] = le_home.fit_transform(data['home_team'])
    data['away_team'] = le_away.fit_transform(data['away_team'])

    data['home_recent_wins'] = data['home_team'].apply(lambda x: get_recent_form(data, x)[0]) * scaling_factor
    data['home_recent_draws'] = data['home_team'].apply(lambda x: get_recent_form(data, x)[1]) * scaling_factor
    data['home_recent_losses'] = data['home_team'].apply(lambda x: get_recent_form(data, x)[2]) * scaling_factor
    data['away_recent_wins'] = data['away_team'].apply(lambda x: get_recent_form(data, x)[0]) * scaling_factor
    data['away_recent_draws'] = data['away_team'].apply(lambda x: get_recent_form(data, x)[1]) * scaling_factor
    data['away_recent_losses'] = data['away_team'].apply(lambda x: get_recent_form(data, x)[2]) * scaling_factor

    features = [
        'home_team',
        'away_team',
        'home_recent_wins',
        'home_recent_draws',
        'home_recent_losses',
        'away_recent_wins',
        'away_recent_draws',
        'away_recent_losses'
    ]

    ml_features = data[features]
    ml_target = data['winner']

    return data, le_home, le_away, ml_features, ml_target

def get_recent_form(data, team, n_matches=5):
    team_data = data[(data['home_team'] == team) | (data['away_team'] == team)].tail(n_matches)
    wins = 0
    draws = 0
    losses = 0
    for _, row in team_data.iterrows():
        if row['winner'] == 'home' and row['home_team'] == team:
            wins += 1
        elif row['winner'] == 'away' and row['away_team'] == team:
            wins += 1
        elif row['winner'] == 'draw':
            draws += 1
        else:
            losses += 1

    return wins / n_matches, draws / n_matches, losses / n_matches

def train_model(ml_features_train, ml_target_train):
    clf = GradientBoostingClassifier()
    clf.fit(ml_features_train, ml_target_train)

    return clf

def evaluate_model(clf, ml_features, ml_target, ml_features_test, ml_target_test):
    y_pred = clf.predict(ml_features_test)

    cv_scores = cross_val_score(clf, ml_features, ml_target, cv=5)

    accuracy = accuracy_score(ml_target_test, y_pred)
    precision = precision_score(ml_target_test, y_pred, average='weighted')
    recall = recall_score(ml_target_test, y_pred, average='weighted')
    f1 = f1_score(ml_target_test, y_pred, average='weighted')

    print("MODEL PERFORMANCE & METRICS")
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean cross-validation score: {cv_scores.mean():.4f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-score: {f1 * 100:.2f}%")


def predict_outcome(data, clf, le_home, le_away, home_team, away_team):
    home_recent_wins, home_recent_draws, home_recent_losses = get_recent_form(data, le_home.transform([home_team])[0])
    away_recent_wins, away_recent_draws, away_recent_losses = get_recent_form(data, le_away.transform([away_team])[0])
    
    input_data = np.array([
        [le_home.transform([home_team])[0], le_away.transform([away_team])[0]] +
        [home_recent_wins * scaling_factor, home_recent_draws * scaling_factor, home_recent_losses * scaling_factor,
        away_recent_wins * scaling_factor, away_recent_draws * scaling_factor, away_recent_losses * scaling_factor]
    ])
    prediction = clf.predict(input_data)[0]
    probabilities = clf.predict_proba(input_data)[0]

    return prediction, probabilities

def predictor(home_team, away_team):
    data, le_home, le_away, ml_features, ml_target = load_and_preprocess_data()
    ml_features_train, ml_features_test, ml_target_train, ml_target_test = train_test_split(ml_features, ml_target, test_size=0.2, random_state=42)
    clf = train_model(ml_features_train, ml_target_train)
    evaluate_model(clf, ml_features, ml_target, ml_features_test, ml_target_test)

    prediction, probabilities = predict_outcome(data, clf, le_home, le_away, home_team, away_team)

    print(f"\n{home_team} (Home) Win - {probabilities[2] * 100:.2f}%")
    print(f"{away_team} (Away) Win - {probabilities[0] * 100:.2f}%")
    print(f"Draw - {probabilities[1] * 100:.2f}%\n")

    save_prediction(home_team, away_team, prediction, probabilities)


def save_prediction(home_team, away_team, prediction, probabilities):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {
        "timestamp": timestamp,
        "home_team": home_team,
        "away_team": away_team,
        "prediction": prediction,
        "probabilities": {
            "home_win": probabilities[2],
            "away_win": probabilities[0],
            "draw": probabilities[1]
        }
    }
    
    with open("predictions.csv", "a+") as file:
        file.write(json.dumps(data) + '\n')

def output_previous_prediction():
    try:
        with open("predictions.csv", "r") as file:
            lines = file.readlines()
            last_prediction = json.loads(lines[-1])

            print(f"\nPrevious Prediction (Timestamp: {last_prediction['timestamp']}):")
            print(f"{last_prediction['home_team']} (Home) Win - {last_prediction['probabilities']['home_win'] * 100:.2f}%")
            print(f"{last_prediction['away_team']} (Away) Win - {last_prediction['probabilities']['away_win'] * 100:.2f}%")
            print(f"Draw - {last_prediction['probabilities']['draw'] * 100:.2f}%\n")

    except FileNotFoundError:
        print("No previous prediction found.")

# Example usage
predictor("Manchester Reds", "Chelsea")
output_previous_prediction()
