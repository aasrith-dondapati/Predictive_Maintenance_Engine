import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Comments: This script loads the simulated data, performs preprocessing steps
# such as handling missing values, calculating time-to-failure, and engineering
# new features like rolling averages and standard deviations. It then splits
# the data into training and testing sets.

# Load the simulated data
try:
    df = pd.read_csv('simulated_sensor_data.csv', parse_dates=['timestamp'])
    print("Simulated data loaded successfully.")
except FileNotFoundError:
    print("Error: 'simulated_sensor_data.csv' not found. Please run Step 1 (Data Simulation) first.")
    exit() # Exit if data is not available

# Sort data by equipment and timestamp
df = df.sort_values(by=['equipment_id', 'timestamp']).reset_index(drop=True)

# 1. Handle Missing Values (Example: If missing values were present, we'd interpolate) ---
# For our simulated data, we don't expect missing values, but this is where you'd add it.
# df.fillna(method='ffill', inplace=True) # Forward fill
# df.fillna(method='bfill', inplace=True) # Backward fill any remaining

print("\nChecking for missing values after initial load:")
print(df.isnull().sum())

# --- 2. Outlier Detection and Handling (Example: Simple IQR method) ---
# For simplicity, we won't remove outliers from the simulated data as they represent degradation.
# However, in real-world scenarios, you might cap or transform extreme values.
# Example for a single column:
# Q1 = df['temperature_C'].quantile(0.25)
# Q3 = df['temperature_C'].quantile(0.75)
# IQR = Q3 - Q1
# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR
# df['temperature_C'] = np.where(df['temperature_C'] > upper_bound, upper_bound, df['temperature_C'])
# df['temperature_C'] = np.where(df['temperature_C'] < lower_bound, lower_bound, df['temperature_C'])


# --- 3. Feature Engineering ---

# Define the prediction horizon (how many days in advance we want to predict failure)
# Let's say we want to predict if a failure will occur within the next 7 days.
PREDICTION_HORIZON_DAYS = 7
SENSOR_FREQ_MINUTES = 10
TIME_TO_FAILURE_THRESHOLD_MINUTES = PREDICTION_HORIZON_DAYS * 24 * 60

print(f"\nEngineering features with a prediction horizon of {PREDICTION_HORIZON_DAYS} days...")

# Calculate Time-to-Failure (TTF) for each equipment
# This is a critical step: for each row, we want to know how many time units until the *next* failure.
df['time_to_failure_minutes'] = np.nan

for equipment_id in df['equipment_id'].unique():
    eq_df = df[df['equipment_id'] == equipment_id].copy()
    failure_timestamps = eq_df[eq_df['is_failure'] == 1]['timestamp'].tolist()

    if not failure_timestamps:
        # If no failures, TTF is effectively infinite (or a large number indicating no imminent failure)
        df.loc[df['equipment_id'] == equipment_id, 'time_to_failure_minutes'] = eq_df.shape[0] * SENSOR_FREQ_MINUTES # Assign a large value
        continue

    for i, row in eq_df.iterrows():
        current_time = row['timestamp']
        # Find the next failure after the current timestamp
        future_failures = [ft for ft in failure_timestamps if ft > current_time]

        if future_failures:
            next_failure_time = min(future_failures)
            ttf = (next_failure_time - current_time).total_seconds() / 60 # TTF in minutes
            df.loc[i, 'time_to_failure_minutes'] = ttf
        else:
            # If no future failures, assign a large value or NaN if it signifies "no failure"
            # We'll use a large value consistent with "not failing soon"
            df.loc[i, 'time_to_failure_minutes'] = eq_df.shape[0] * SENSOR_FREQ_MINUTES # Or a predefined max TTF

# Create the binary target variable: `is_failing_soon`
# `is_failing_soon` is 1 if time_to_failure is within the prediction horizon, else 0.
df['is_failing_soon'] = (df['time_to_failure_minutes'] <= TIME_TO_FAILURE_THRESHOLD_MINUTES).astype(int)

# Rolling window features
# We'll use a window of 6 hours (36 readings for 10-minute frequency)
WINDOW_SIZE = 6 * 60 // SENSOR_FREQ_MINUTES # 36 readings for 10-min freq (6 hours)
features_to_roll = ['temperature_C', 'vibration_ms2', 'pressure_kPa', 'voltage_V']

for col in features_to_roll:
    df[f'{col}_roll_mean_{WINDOW_SIZE}h'] = df.groupby('equipment_id')[col].transform(lambda x: x.rolling(window=WINDOW_SIZE, min_periods=1).mean())
    df[f'{col}_roll_std_{WINDOW_SIZE}h'] = df.groupby('equipment_id')[col].transform(lambda x: x.rolling(window=WINDOW_SIZE, min_periods=1).std())
    df[f'{col}_roll_min_{WINDOW_SIZE}h'] = df.groupby('equipment_id')[col].transform(lambda x: x.rolling(window=WINDOW_SIZE, min_periods=1).min())
    df[f'{col}_roll_max_{WINDOW_SIZE}h'] = df.groupby('equipment_id')[col].transform(lambda x: x.rolling(window=WINDOW_SIZE, min_periods=1).max())

# Lagged features (e.g., previous readings)
LAG_SIZE = 1 # Lag by 1 reading (10 minutes ago)
for col in features_to_roll:
    df[f'{col}_lag_{LAG_SIZE}'] = df.groupby('equipment_id')[col].shift(LAG_SIZE)

# Ensure no NaN values remain from rolling/lagging at the beginning of each equipment's data
# For simplicity, we'll fill with 0 or forward/backward fill based on context.
# In a real scenario, these initial rows might be dropped or handled more carefully.
df.fillna(0, inplace=True) # Fill NaNs introduced by rolling/lagging at start of groups

# Time-based features
df['hour_of_day'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month

print("\nDataFrame head after feature engineering:")
print(df.head())
print("\nDataFrame info after feature engineering:")
print(df.info())

# --- 4. Data Splitting ---
# For time-series data, it's crucial to split chronologically, not randomly.
# We'll use the first 80% of data for training and the last 20% for testing.
# This ensures that our model is evaluated on future, unseen data.

# Determine the split point (e.g., 80% of the total unique timestamps)
split_point = int(len(df) * 0.8)
train_df = df.iloc[:split_point]
test_df = df.iloc[split_point:]

# Define features (X) and target (y)
# Drop original sensor values and 'is_failure' (which we use to calculate TTF), equipment_id, timestamp
# Also drop 'days_since_last_failure' as 'time_to_failure_minutes' and 'is_failing_soon' are our targets/proxies
features = [col for col in df.columns if col not in ['equipment_id', 'timestamp', 'is_failure', 'failure_type', 'time_to_failure_minutes', 'is_failing_soon', 'days_since_last_failure']]
target = 'is_failing_soon' # Binary classification for predicting imminent failure

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

print(f"\nFeatures used for training: {len(features)}")
print(f"Number of training samples: {len(X_train)}")
print(f"Number of testing samples: {len(X_test)}")
print(f"Proportion of 'is_failing_soon' in training set: {y_train.value_counts(normalize=True)}")
print(f"Proportion of 'is_failing_soon' in testing set: {y_test.value_counts(normalize=True)}")

# Save processed data for next steps
train_df.to_csv('processed_train_data.csv', index=False)
test_df.to_csv('processed_test_data.csv', index=False)
print("\nProcessed training and testing data saved.")

# Visualize the distribution of the target variable
plt.figure(figsize=(8, 5))
y_train.value_counts().plot(kind='bar', title='Distribution of is_failing_soon (Training Set)')
plt.xlabel('Is Failing Soon (0=No, 1=Yes)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

