import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.utils.class_weight import compute_sample_weight
import os
import glob

def load_data(path: str, num_columns=60) -> pd.DataFrame:
    """
    Loads and merges CSV files from the specified directory.
    
    Parameters:
    path (str): The directory path containing the CSV files.
    num_columns (int): Number of columns to keep from the CSV files.

    Returns:
    pandas.DataFrame: The merged DataFrame containing data from the selected CSV files.
    """
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"The directory '{path}' does not exist.")

        all_files = [
            file for file in glob.glob(f"{path}/*.csv") if 'player positioning' not in file
        ]

        if not all_files:
            raise ValueError(f"No valid CSV files found in the directory '{path}'.")

        columns_to_keep = list(range(num_columns))
        df_list = [pd.read_csv(filename, usecols=columns_to_keep) for filename in all_files]
        merged_df = pd.concat(df_list, ignore_index=True)

        output_path = "/Users/tommayer/Desktop/games_test.csv"
        merged_df.to_csv(output_path, index=False)

        return merged_df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def preprocess_data(data: pd.DataFrame) -> tuple:
    """
    Preprocesses the data by identifying column types, encoding categorical data, and scaling numerical data.
    Returns train/test/validation splits of features and target.
    """
    # Identify column types
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = data.select_dtypes(include=['object']).columns

    # Handle categorical data with One-Hot Encoding
    data = pd.get_dummies(data, columns=categorical_cols)

    # Scale numerical features
    scaler = StandardScaler()
    numerical_features = [col for col in numerical_cols if col != 'RunsScored']
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    # Feature engineering
    data['SpeedSpin'] = data['RelSpeed'] * data['SpinRate']
    data['BreakComposite'] = data['InducedVertBreak'] * data['HorzBreak']
    data['RelSpeed_Squared'] = data['RelSpeed'] ** 2
    data['SpinRate_Squared'] = data['SpinRate'] ** 2

    # Split into features and target
    X = data.drop(['RunsScored'], axis=1)
    y = data['RunsScored']

    # Split into train/test/validation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

    return X_train, X_test, X_val, y_train, y_test, y_val

def train_random_forest(X_train, X_val, y_train, y_val):
    """
    Trains and evaluates a Random Forest model.
    """
    # Compute sample weights
    sample_weights = compute_sample_weight('balanced', y_train)

    # Create and train the model
    model_rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_split=2,
        random_state=1,
        n_jobs=-1
    )
    
    model_rf.fit(X_train, y_train, sample_weight=sample_weights)
    predictions_rf = model_rf.predict(X_val)

    # Evaluate the model
    mse = mean_squared_error(y_val, predictions_rf)
    rmse = mean_squared_error(y_val, predictions_rf, squared=False)
    r2 = r2_score(y_val, predictions_rf)

    print(f'Random Forest Performance:')
    print(f'Mean Squared Error: {mse:.4f}')
    print(f'Root Mean Squared Error: {rmse:.4f}')
    print(f'R² Score: {r2:.4f}')

    return model_rf

def train_xgboost(X_train, X_val, y_train, y_val):
    """
    Trains and evaluates an XGBoost model.
    """
    model_xgb = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    
    model_xgb.fit(X_train, y_train)
    predictions_xgb = model_xgb.predict(X_val)

    mse = mean_squared_error(y_val, predictions_xgb)
    rmse = mean_squared_error(y_val, predictions_xgb, squared=False)
    r2 = r2_score(y_val, predictions_xgb)

    print(f'XGBoost Performance:')
    print(f'Mean Squared Error: {mse:.4f}')
    print(f'Root Mean Squared Error: {rmse:.4f}')
    print(f'R² Score: {r2:.4f}')

    return model_xgb

def plot_target_distribution(data):
    """
    Plots the distribution of the target variable (RunsScored).
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data['RunsScored'], bins=30, kde=True)
    plt.title('Distribution of RunsScored')
    plt.xlabel('RunsScored')
    plt.ylabel('Frequency')
    plt.show()

def main():
    # Define required columns
    required_columns = [
        'TaggedPitchType', 'AutoPitchType', 'RunsScored', 'RelSpeed', 
        'RelHeight', 'VertRelAngle', 'HorzRelAngle', 'SpinRate', 'SpinAxis', 
        'Tilt', 'Extension', 'InducedVertBreak', 'HorzBreak', 'VertApprAngle', 
        'HorzApprAngle'
    ]

    # Load and preprocess data
    path = "/Users/tommayer/Desktop/training_data.csv"
    data = pd.read_csv(path, usecols=required_columns)
    
    # Plot target distribution
    plot_target_distribution(data)

    # Preprocess data
    X_train, X_test, X_val, y_train, y_test, y_val = preprocess_data(data)

    # Train and evaluate models
    rf_model = train_random_forest(X_train, X_val, y_train, y_val)
    xgb_model = train_xgboost(X_train, X_val, y_train, y_val)

if __name__ == "__main__":
    main()