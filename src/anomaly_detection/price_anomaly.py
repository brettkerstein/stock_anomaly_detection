import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
from optuna import create_study
from optuna.integration import LightGBMPruningCallback

# Load the data
def load_data(file_path):
    return pd.read_csv(file_path)

# Preprocess the data
def preprocess_data(df, id_columns, categorical_columns, target_column):
    # Drop identification columns
    df = df.drop(columns=id_columns)
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col].astype(str))
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return X, y

# Define the objective function for Optuna
def objective(trial, X, y):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 3000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1.0),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.1, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.1, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0)
    }
    
    cv_scores = cross_val_score(
        lgb.LGBMRegressor(**params),
        X, y,
        cv=5,
        scoring='neg_mean_squared_error'
    )
    
    return -np.mean(cv_scores)

# Train the model with the best hyperparameters
def train_model(X, y, best_params):
    model = lgb.LGBMRegressor(**best_params)
    model.fit(X, y)
    return model

# Evaluate the model
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    return mse, rmse, r2

# Main function
def main():
    # Load and preprocess data
    df = load_data('your_data.csv')
    id_columns = ['id']  # Add your identification columns here
    categorical_columns = ['category1', 'category2']  # Add your categorical columns here
    target_column = 'target'  # Replace with your target column name
    
    X, y = preprocess_data(df, id_columns, categorical_columns, target_column)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Hyperparameter tuning
    study = create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=100)
    
    best_params = study.best_params
    print("Best hyperparameters:", best_params)
    
    # Train the model with best parameters
    model = train_model(X_train, y_train, best_params)
    
    # Evaluate the model
    mse, rmse, r2 = evaluate_model(model, X_test, y_test)
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"R-squared Score: {r2}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10))

if __name__ == "__main__":
    main()
