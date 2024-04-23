import os
import pickle
from collections import defaultdict
import warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

scaling_factor = 2


def warn(*args, **kwargs):
    pass


warnings.warn = warn


def load_datasets():
    cleaned_matches = pd.read_csv("cleaned matches.csv")
    mre_dataset = pd.read_csv("MRE dataset.csv")
    league_table = pd.read_csv("league table.csv")
    league_team_form = pd.read_csv("league team-form.csv")
    current_fixtures = pd.read_csv("current fixtures.csv")
    return (
        cleaned_matches,
        mre_dataset,
        league_table,
        league_team_form,
        current_fixtures,
    )


def preprocess_data(cleaned_matches, mre_dataset, league_table, league_team_form):
    # Drop unnecessary columns from datasets
    cleaned_matches = cleaned_matches[
        [
            "home_team",
            "away_team",
            "winner",
            "home_team_abbr",
            "away_team_abbr",
            "home_score",
            "away_score",
        ]
    ]
    mre_dataset = mre_dataset[["home_team", "away_team", "winner"]]
    league_table = league_table[["Team", "Home wins", "Away wins"]]
    league_team_form = league_team_form[
        [
            "Team",
            "Round 1",
            "Round 2",
            "Round 3",
            "Round 4",
            "Round 5",
            "Round 6",
            "Round 7",
            "Round 8",
            "Round 9",
            "Round 10",
            "Round 11",
            "Round 12",
            "Round 13",
            "Round 14",
            "Round 15",
            "Round 16",
            "Round 17",
            "Round 18",
            "Round 19",
            "Round 20",
            "Round 21",
            "Round 22",
            "Round 23",
            "Round 24",
            "Round 25",
            "Round 26",
            "Round 27",
            "Round 28",
            "Round 29",
            "Round 30",
        ]
    ]

    # Merge league_table and league_team_form on 'Team' column
    league_data = pd.merge(league_table, league_team_form, on="Team")

    # Calculate team win rate based on historical matches
    cleaned_matches['home_team_win_rate'] = cleaned_matches.groupby('home_team')['winner'].apply(lambda x: (x == 'home').mean())
    cleaned_matches['away_team_win_rate'] = cleaned_matches.groupby('away_team')['winner'].apply(lambda x: (x == 'away').mean())

    # Calculate goal difference for home and away teams
    cleaned_matches['home_goal_diff'] = cleaned_matches['home_score'] - cleaned_matches['away_score']
    cleaned_matches['away_goal_diff'] = cleaned_matches['away_score'] - cleaned_matches['home_score']

    # Calculate goal average for home and away teams
    cleaned_matches['home_goal_avg'] = cleaned_matches['home_score'] / cleaned_matches.groupby('home_team')['home_team'].transform('count')
    cleaned_matches['away_goal_avg'] = cleaned_matches['away_score'] / cleaned_matches.groupby('away_team')['away_team'].transform('count')

    # Calculate win streak and lose streak for each team
    cleaned_matches['home_win_streak'] = (cleaned_matches['winner'] == 'home').groupby(cleaned_matches['home_team']).cumsum()
    cleaned_matches['home_lose_streak'] = (cleaned_matches['winner'] != 'home').groupby(cleaned_matches['home_team']).cumsum()
    cleaned_matches['away_win_streak'] = (cleaned_matches['winner'] == 'away').groupby(cleaned_matches['away_team']).cumsum()
    cleaned_matches['away_lose_streak'] = (cleaned_matches['winner'] != 'away').groupby(cleaned_matches['away_team']).cumsum()

    # Calculate head-to-head performance
    head_to_head = cleaned_matches.groupby(['home_team', 'away_team']).agg(
        home_wins=('winner', lambda x: (x == 'home').sum()),
        away_wins=('winner', lambda x: (x == 'away').sum()),
        draws=('winner', lambda x: (x == 'draw').sum())
    ).reset_index()
    head_to_head['home_win_pct'] = head_to_head['home_wins'] / (head_to_head['home_wins'] + head_to_head['away_wins'])
    head_to_head['away_win_pct'] = head_to_head['away_wins'] / (head_to_head['home_wins'] + head_to_head['away_wins'])

    # Merge head-to-head performance with cleaned_matches
    cleaned_matches = cleaned_matches.merge(head_to_head[['home_team', 'away_team', 'home_win_pct', 'away_win_pct']], on=['home_team', 'away_team'])

    # Calculate recent performance metrics
    cleaned_matches['home_avg_goals_scored'] = cleaned_matches.groupby('home_team')['home_score'].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
    cleaned_matches['away_avg_goals_scored'] = cleaned_matches.groupby('away_team')['away_score'].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
    cleaned_matches['home_avg_goals_conceded'] = cleaned_matches.groupby('home_team')['away_score'].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
    cleaned_matches['away_avg_goals_conceded'] = cleaned_matches.groupby('away_team')['home_score'].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)

    # Calculate goal difference ratio
    cleaned_matches['home_goal_diff_ratio'] = cleaned_matches['home_score'] / (cleaned_matches['home_score'] + cleaned_matches['away_score'])
    cleaned_matches['away_goal_diff_ratio'] = cleaned_matches['away_score'] / (cleaned_matches['home_score'] + cleaned_matches['away_score'])

    # Calculate win rate against common opponents
    common_opponents = cleaned_matches.groupby(['home_team', 'away_team'])['winner'].apply(lambda x: (x == 'home').mean()).reset_index()
    common_opponents.rename(columns={'winner': 'home_win_rate_common_opponents'}, inplace=True)
    cleaned_matches = cleaned_matches.merge(common_opponents, on=['home_team', 'away_team'], how='left')

    # Calculate venue-based performance
    home_performance = cleaned_matches.groupby('home_team')['winner'].apply(lambda x: (x == 'home').mean()).reset_index()
    home_performance.rename(columns={'winner': 'home_win_rate'}, inplace=True)
    away_performance = cleaned_matches.groupby('away_team')['winner'].apply(lambda x: (x == 'away').mean()).reset_index()
    away_performance.rename(columns={'winner': 'away_win_rate'}, inplace=True)
    cleaned_matches = cleaned_matches.merge(home_performance, left_on='home_team', right_on='home_team', how='left')
    cleaned_matches = cleaned_matches.merge(away_performance, left_on='away_team', right_on='away_team', how='left')

    # Calculate win rate for each team in the current season
    team_win_rate = cleaned_matches.groupby('home_team')['winner'].apply(lambda x: (x =='home').mean()).reset_index()
    team_win_rate.rename(columns={'winner': 'win_rate'}, inplace=True)
    cleaned_matches = cleaned_matches.merge(team_win_rate, on='home_team', how='left')

    return cleaned_matches, mre_dataset, league_data, league_team_form


def train_model(ml_features_train, ml_target_train, model_filename='trained_model.pkl'):
    # Check if the model file already exists
    if os.path.exists(model_filename):
        # If the model file exists, load the model
        with open(model_filename, 'rb') as f:
            ensemble_clf = pickle.load(f)
    else:
        # If the model file does not exist, train a new model

        model1 = GradientBoostingClassifier()
        model2 = RandomForestClassifier()

        # Combine models using Voting...
        ensemble_clf = VotingClassifier(estimators=[('gb', model1), ('rf', model2)],voting='soft')
        ensemble_clf.fit(ml_features_train, ml_target_train)

        # Save the trained model to a file
        with open(model_filename, 'wb') as f:
            pickle.dump(ensemble_clf, f)

    return ensemble_clf

def train_new_model(ml_features, ml_target, model_filename='trained_model.pkl'):
    clf = GradientBoostingClassifier()
    clf2 = RandomForestClassifier()

    ensemble_clf = VotingClassifier(estimators=[('gb', clf), ('rf', clf2)],voting='soft')
    ensemble_clf.fit(ml_features, ml_target)

    with open(model_filename, 'wb') as f:
        pickle.dump(ensemble_clf, f)

    return ensemble_clf


def evaluate_model(clf, ml_features, ml_target, ml_features_test, ml_target_test):
    y_pred = clf.predict(ml_features_test)
    cv_scores = cross_val_score(clf, ml_features, ml_target, cv=5)
    accuracy = accuracy_score(ml_target_test, y_pred)
    precision = precision_score(ml_target_test, y_pred, average="weighted")
    recall = recall_score(ml_target_test, y_pred, average="weighted")
    f1 = f1_score(ml_target_test, y_pred, average="weighted")

    print("MODEL PERFORMANCE & METRICS")
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean cross-validation score: {cv_scores.mean():.4f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-score: {f1 * 100:.2f}%")

    confusion_matrix = pd.crosstab(index=ml_target_test, columns=y_pred)
    print("Confusion Matrix:")
    print(confusion_matrix)


def predict_outcome(clf, cleaned_matches, home_team, away_team, le_home, le_away):
    try:
        home_team_encoded = le_home.transform([home_team])[0]
        away_team_encoded = le_away.transform([away_team])[0]
    except (KeyError, ValueError) as e:
        print(f"Error: Unseen label encountered - {e}")
        return None, None

    if home_team_encoded is None or away_team_encoded is None:
        print("Error: Unable to encode team labels.")
        return None, None

    input_data = np.array([[home_team_encoded, away_team_encoded]])

    prediction = clf.predict(input_data)[0]
    probabilities = clf.predict_proba(input_data)[0]

    return prediction, probabilities


def get_recent_form(data, team, n_matches=5):
    team_data = data[(data["home_team"] == team) | (data["away_team"] == team)].tail(
        n_matches
    )
    wins = 0
    draws = 0
    losses = 0
    for _, row in team_data.iterrows():
        if row["winner"] == "home" and row["home_team"] == team:
            wins += 1
        elif row["winner"] == "away" and row["away_team"] == team:
            wins += 1
        elif row["winner"] == "draw":
            draws += 1
        else:
            losses += 1

    return wins / n_matches, draws / n_matches, losses / n_matches


def main():
    (
        cleaned_matches,
        mre_dataset,
        league_table,
        league_team_form,
        current_fixtures,
    ) = load_datasets()

    # Preprocess data
    cleaned_matches, mre_dataset, league_table, league_team_form = preprocess_data(
        cleaned_matches, mre_dataset, league_table, league_team_form
    )

    # Check if 'home_team_abbr' and 'away_team_abbr' columns are present
    if (
        "home_team_abbr" not in cleaned_matches.columns
        or "away_team_abbr" not in cleaned_matches.columns
    ):
        print(
            "Error: 'home_team_abbr' or 'away_team_abbr' columns are missing in the cleaned_matches DataFrame."
        )
        return

    # Encode categorical variables
    le_home = LabelEncoder()
    le_away = LabelEncoder()

    # Fit label encoders with all team abbreviations
    all_team_abbreviations = set(cleaned_matches["home_team_abbr"]).union(
        set(cleaned_matches["away_team_abbr"])
    )
    le_home.fit(list(all_team_abbreviations))
    le_away.fit(list(all_team_abbreviations))

    # Encode home_team_abbr and away_team_abbr
    cleaned_matches['home_team'] = le_home.transform(cleaned_matches['home_team_abbr'])
    cleaned_matches['away_team'] = le_away.transform(cleaned_matches['away_team_abbr'])

    print("Options:")
    print("1. Enter specific teams")
    print("2. Load entire season predictions")

    option = input("Choose an option (1/2): ")

    if option == "1":
        home_team = input("Enter the home team abbreviation: ")
        away_team = input("Enter the away team abbreviation: ")

        # Define original and additional features
        original_features = ["home_team", "away_team"]

        # Create input data with all features
        ml_features = cleaned_matches[original_features].copy()
        ml_target = cleaned_matches["winner"]

        clf = train_model(ml_features, ml_target)
        prediction, probabilities = predict_outcome(
            clf, cleaned_matches, home_team, away_team, le_home, le_away
        )
        print(f"Prediction for {home_team} vs {away_team}: {prediction}")
    elif option == "2":
        # Define features and target
        original_features = ["home_team", "away_team"]
        features = original_features
        ml_features = cleaned_matches[
            original_features
        ].copy()
        ml_target = cleaned_matches["winner"]

        # Split data into training and testing sets
        ml_features_train, ml_features_test, ml_target_train, ml_target_test = train_test_split(ml_features, ml_target, test_size=0.2, random_state=42)
        # Train model
        clf = train_model(ml_features_train, ml_target_train)

        # Evaluate model
        evaluate_model(clf, ml_features, ml_target, ml_features_test, ml_target_test)

        # Predict outcomes for current fixtures
        predictions = defaultdict(dict)
        for _, row in current_fixtures.iterrows():
            home_team = row["home_team_abbr"]
            away_team = row["away_team_abbr"]
            prediction, probabilities = predict_outcome(
                clf, cleaned_matches, home_team, away_team, le_home, le_away
            )
            predictions[(home_team, away_team)] = prediction

        # Print predictions to console
        print("\nPredictions for current season's fixtures:")
        for (home_team, away_team), prediction in predictions.items():
            print(f"{home_team} vs {away_team}: {prediction}")

        # Save predictions to CSV file
        df_predictions = pd.DataFrame(
            list(predictions.items()), columns=["Fixture", "Prediction"]
        )
        df_predictions.to_csv("predictions.csv", index=False)

    else:
        print("Invalid option. Please choose either 1 or 2.")

    edit_option = input("Do you want to edit any predicted result? (yes/no): ")

    if edit_option.lower() == "yes":
        # Prompt the user to input the edited prediction
        edited_home_team = input("Enter the edited home team abbreviation: ")
        edited_away_team = input("Enter the edited away team abbreviation: ")
        edited_prediction = input("Enter the edited prediction (home/draw/away): ")

        # Update the dataset with the user's edits
        idx = (cleaned_matches['home_team_abbr'] == edited_home_team) & (cleaned_matches['away_team_abbr'] == edited_away_team)
        cleaned_matches.loc[idx, 'winner'] = edited_prediction

        # Define features and target
        features = ['home_team_abbr', 'away_team_abbr']
        ml_features = cleaned_matches[features]
        ml_target = cleaned_matches['winner']

        # Reinitialize and refit label encoders with the updated dataset
        le_home = LabelEncoder()
        le_away = LabelEncoder()
        all_team_abbreviations = set(ml_features['home_team_abbr']).union(set(ml_features['away_team_abbr']))
        le_home.fit(list(all_team_abbreviations))
        le_away.fit(list(all_team_abbreviations))

        # Encode home_team_abbr and away_team_abbr
        cleaned_matches['home_team'] = le_home.transform(cleaned_matches['home_team_abbr'])
        cleaned_matches['away_team'] = le_away.transform(cleaned_matches['away_team_abbr'])

        original_features = ["home_team", "away_team"]
        ml_features = cleaned_matches[original_features].copy()

        # Train a new model using the updated dataset
        clf = train_new_model(ml_features, ml_target)

        # Display the updated prediction based on the new trained model
        updated_prediction, _ = predict_outcome(clf, cleaned_matches, edited_home_team, edited_away_team, le_home, le_away)
        print(f"Updated prediction for {edited_home_team} vs {edited_away_team}: {updated_prediction}")
    elif edit_option.lower() == "no":
        print("Thank you for using the predictor.")
    else:
        print("Invalid option. Please choose either 'yes' or 'no'.")


if __name__ == "__main__":
    main()
