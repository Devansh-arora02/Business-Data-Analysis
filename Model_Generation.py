import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, ShuffleSplit

import logging
import sys
import os
import datetime

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
directory = f"C:\\Users\\arora\\OneDrive\\Desktop\\BDM Capstone\\{current_time}_Models"

if not os.path.exists(directory):
    os.makedirs(directory)

log_file = os.path.join(directory, "output_file.log")

# Configure logging
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,  # Log level (INFO, DEBUG, ERROR, etc.)
    format='%(asctime)s - %(message)s'
)

# Redirect print statements to the log file
class PrintLogger:
    def write(self, message):
        if message.strip() != "":
            logging.info(message.strip())
    def flush(self):
        pass

sys.stdout = PrintLogger()  # Redirect stdout (print output)
sys.stderr = PrintLogger()  # Redirect stderr (error output)
print('Log file set up')

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created at {path}")
    else:
        pass


numerical_features = ['MRP', 'ProductLevelDisc%', 'ProductLevelDiscAmount', 'Hour', 'DayOfWeek', 'IsWeekend', 'AmountperProduct']
categorical_features = ['StoreName', 'New customer Type', 'ProductName']

def preprocess(df):
  # Feature Engineering: Create temporal features (if applicable)
  df['BillTime'] = pd.to_datetime(df['BillTime'], format='%I:%M %p', errors='coerce')
  df['Hour'] = df['BillTime'].dt.hour
  df['DayOfWeek'] = df['BillTime'].dt.dayofweek
  df['IsWeekend'] = (df['BillTime'].dt.weekday >= 5).astype(int)
  df['AmountperProduct'] = df['Amount']/df['Quantity']

  X = df[['StoreName', 'MRP', 'ProductLevelDisc%', 'ProductLevelDiscAmount', 'ProductName', 'New customer Type', 'Hour', 'DayOfWeek', 'IsWeekend', 'AmountperProduct']]

  unique_categories = [np.append(df[key].dropna().unique(), np.nan) for key in categorical_features]

  numerical_transformer = SimpleImputer(strategy='mean')
  categorical_transformer = Pipeline(steps=[
      ('imputer', SimpleImputer(strategy='most_frequent')),
      ('onehot', OneHotEncoder(categories = unique_categories))
  ])

  preprocessor = ColumnTransformer(
      transformers=[
          ('num', numerical_transformer, numerical_features),
          ('cat', categorical_transformer, categorical_features)
      ])

  y = df['Quantity']
  # Split the dataset into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  return preprocessor, X_train, X_test, y_train, y_test


def model(df, path, name):
    print(f"Running Model {name}")
    preprocessor, X_train, X_test, y_train, y_test = preprocess(df)

    param_grid = {
    'regressor__n_estimators': [100, 200, 300, 400],
    'regressor__max_depth': [None, 10, 20, 30, 40, 50, 100],
    'regressor__min_samples_split': [2, 5, 10, 15],
    'regressor__min_samples_leaf': [1, 2, 4],
    'regressor__max_features': [None, 'sqrt', 'log2', 0.5, 0.75]
    }


    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    shuffle_split = ShuffleSplit(n_splits=3, test_size=0.3, random_state=42)

    grid_search = GridSearchCV(model, param_grid, cv=shuffle_split, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    print(f"Best parameters for {name}: {grid_search.best_params_}")
    joblib.dump(best_model, os.path.join(path, f'{name}_model.pkl'))
    return best_model, X_test, y_test


def residuals(y_test, y_pred_rf, path, name):
  residuals_rf = y_test - y_pred_rf
  plt.figure(figsize=(8, 6))
  plt.scatter(y_test, residuals_rf)
  plt.axhline(0, color='red', linestyle='--')
  plt.title(f"Residual Plot {name}")
  plt.xlabel("Actual Quantity")
  plt.ylabel("Residuals (Actual - Predicted)")
  plt.savefig(f'{path}\\residuals_plot.png', dpi=300, bbox_inches='tight')

def importance(best_model, path, name):
    # Access the regressor inside the pipeline
    model = best_model.named_steps['regressor']

    feature_names = numerical_features + list(
        best_model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features)
    )

    # Extract feature importances
    importances = model.feature_importances_

    # Combine feature names and importances in a DataFrame
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10)

    # Plot feature importances
    plt.figure(figsize=(12, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='salmon')
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Top 10 Feature Importances {name}")
    plt.gca().invert_yaxis()
    plt.savefig(f'{path}\\Feature_importance.png', dpi=300, bbox_inches='tight')


def test_model(rf_reg, df, X_test, y_test, path, name):
    y_pred_rf = rf_reg.predict(X_test)

    # Evaluate the model
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)

    print(f"Random Forest RMSE: {np.sqrt(mse_rf)}")
    print(f"Random Forest R-squared: {r2_rf}")
    residuals(y_test, y_pred_rf, path, name)
    importance(rf_reg, path, name)


df = pd.read_csv('combined_files.csv')

overall_path = f'{current_time}_Models\\Overall'
ensure_directory(overall_path)
best_model, X_test, y_test = model(df, overall_path, 'Overall')
print('Overall Model Metrics :')
test_model(best_model, df, X_test, y_test, overall_path, 'Overall')

for i in range(3):
    cluster_df = pd.read_csv(f'Cluster_{i}_Dataset.csv')
    cluster_path = f'{current_time}_Models\\Cluster_{i}'
    ensure_directory(cluster_path)
    best_model_i, X_test_i, y_test_i = model(cluster_df, cluster_path, f'Cluster {i}')
    print(f'Cluster {i} Model Metrics :')
    test_model(best_model_i, df, X_test_i, y_test_i, cluster_path, f'For Cluster {i}')
print("COMPLETED")
