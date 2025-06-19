import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Comments: This script generates synthetic time-series sensor data for predictive maintenance.
# It simulates normal operation with slight fluctuations and introduces anomalies leading to "failures."

def generate_sensor_data(num_days=365, sensor_freq_minutes=10, equipment_id='EQP001'):
    """
    Generates synthetic sensor data for a given equipment.

    Args:
        num_days (int): Number of days for which to generate data.
        sensor_freq_minutes (int): Frequency of sensor readings in minutes.
        equipment_id (str): Identifier for the equipment.

    Returns:
        pd.DataFrame: DataFrame containing simulated sensor data.
    """
    print(f"Generating data for {equipment_id}...")
    start_date = datetime(2023, 1, 1)
    end_date = start_date + timedelta(days=num_days)
    time_index = pd.date_range(start=start_date, end=end_date, freq=f'{sensor_freq_minutes}min')

    # Sensor parameters (simulate normal operation values)
    temp_base = 50  # degrees Celsius
    vibration_base = 1.5 # m/s^2
    pressure_base = 100 # kPa
    voltage_base = 220 # Volts

    # Introduce some gradual degradation and abrupt failure events
    data = []
    failure_events = {} # To store start time and type of failure for later annotation

    # Simulate random failure events for a few instances within the data range
    num_failures = np.random.randint(1, 3) # 1 to 2 failures per equipment in a year
    failure_points = np.sort(np.random.choice(len(time_index) - 100, num_failures, replace=False)) + 50 # Ensure not at very beginning

    failure_types = {
        0: {"name": "Overheating", "temp_factor": 1.5, "vib_factor": 1.1, "pressure_factor": 1.0, "volt_factor": 1.0},
        1: {"name": "Bearing Failure", "temp_factor": 1.1, "vib_factor": 2.0, "pressure_factor": 1.0, "volt_factor": 0.9},
        2: {"name": "Pressure Leak", "temp_factor": 1.0, "vib_factor": 1.0, "pressure_factor": 0.5, "volt_factor": 1.0},
        3: {"name": "Electrical Fault", "temp_factor": 1.2, "vib_factor": 1.0, "pressure_factor": 1.0, "volt_factor": 0.7}
    }

    for i, timestamp in enumerate(time_index):
        # Normal fluctuations
        temp = temp_base + np.random.normal(0, 1.5)
        vibration = vibration_base + np.random.normal(0, 0.1)
        pressure = pressure_base + np.random.normal(0, 0.5)
        voltage = voltage_base + np.random.normal(0, 0.8)
        
        is_failure = 0 # Default to no failure
        failure_type = "None"
        days_since_failure = np.nan

        # Introduce degradation and failure
        for fp_idx, failure_point_idx in enumerate(failure_points):
            # Degradation starts 2 days before actual failure
            degradation_start_idx = failure_point_idx - (2 * 24 * 60 // sensor_freq_minutes) # 2 days before

            if i >= degradation_start_idx:
                failure_type_idx = fp_idx % len(failure_types) # Cycle through failure types
                failure_info = failure_types[failure_type_idx]
                failure_name = failure_info["name"]

                # Gradual degradation
                if i < failure_point_idx:
                    # Linear increase towards failure point
                    progress = (i - degradation_start_idx) / (failure_point_idx - degradation_start_idx)
                    temp += progress * (temp_base * (failure_info["temp_factor"] - 1))
                    vibration += progress * (vibration_base * (failure_info["vib_factor"] - 1))
                    pressure += progress * (pressure_base * (failure_info["pressure_factor"] - 1))
                    voltage += progress * (voltage_base * (failure_info["volt_factor"] - 1))
                # Abrupt failure
                else:
                    temp *= failure_info["temp_factor"] + np.random.normal(0, 0.5) # Add more noise post-failure
                    vibration *= failure_info["vib_factor"] + np.random.normal(0, 0.2)
                    pressure *= failure_info["pressure_factor"] + np.random.normal(0, 0.1)
                    voltage *= failure_info["volt_factor"] + np.random.normal(0, 0.1)
                    is_failure = 1
                    failure_type = failure_name
                    
                    if timestamp not in failure_events:
                        failure_events[timestamp] = failure_name
        
        # Calculate days since last failure for current timestamp
        current_failure_time = None
        if is_failure == 1 and timestamp not in failure_events:
            failure_events[timestamp] = failure_type
            current_failure_time = timestamp

        # If there's a recorded failure, calculate days since it happened for all subsequent points
        if current_failure_time is not None:
             days_since_failure = 0
        elif len(failure_events) > 0:
            most_recent_failure_time = max(failure_events.keys())
            if timestamp > most_recent_failure_time:
                days_since_failure = (timestamp - most_recent_failure_time).total_seconds() / (24 * 3600)
            else:
                days_since_failure = 0 # It's a failure event or before any failure
        
        data.append([
            equipment_id,
            timestamp,
            temp,
            vibration,
            pressure,
            voltage,
            is_failure,
            failure_type,
            days_since_failure # Will be NaN until a failure occurs, then counts up
        ])

    df = pd.DataFrame(data, columns=[
        'equipment_id', 'timestamp', 'temperature_C', 'vibration_ms2',
        'pressure_kPa', 'voltage_V', 'is_failure', 'failure_type', 'days_since_last_failure'
    ])
    
    # Fill days_since_last_failure for records before any failure event with a large number or 0, or calculate retrospectively
    # For simplicity, let's fill NaN with 0 or a large value for no failure context
    # A more robust approach would be to calculate time_to_failure if we knew future failures, but for prediction,
    # days_since_last_failure as a feature makes sense.
    # We will adjust 'days_since_last_failure' to reflect 'time_until_next_failure' during preprocessing if a future label is defined.

    return df

# Generate data for a few pieces of equipment
equipment_data = []
for i in range(5):
    eq_id = f'EQP{i:03d}'
    equipment_data.append(generate_sensor_data(num_days=365, equipment_id=eq_id))

raw_df = pd.concat(equipment_data).reset_index(drop=True)

# Save the simulated data to a CSV file
raw_df.to_csv('simulated_sensor_data.csv', index=False)

print("\nSimulated data head:")
print(raw_df.head())
print("\nSimulated data info:")
print(raw_df.info())
print(f"\nData saved to simulated_sensor_data.csv with {len(raw_df)} rows.")

# Visualize a sample of the data to see the degradation and failure points
plt.figure(figsize=(15, 8))
sample_eq_id = raw_df['equipment_id'].sample(1).iloc[0]
sample_df = raw_df[raw_df['equipment_id'] == sample_eq_id].copy()

plt.subplot(2, 2, 1)
plt.plot(sample_df['timestamp'], sample_df['temperature_C'])
plt.title(f'{sample_eq_id} - Temperature')
plt.ylabel('Temperature (°C)')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(sample_df['timestamp'], sample_df['vibration_ms2'])
plt.title(f'{sample_eq_id} - Vibration')
plt.ylabel('Vibration (m/s²)')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(sample_df['timestamp'], sample_df['pressure_kPa'])
plt.title(f'{sample_eq_id} - Pressure')
plt.ylabel('Pressure (kPa)')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(sample_df['timestamp'], sample_df['voltage_V'])
plt.title(f'{sample_eq_id} - Voltage')
plt.ylabel('Voltage (V)')
plt.xlabel('Timestamp')
plt.grid(True)

plt.tight_layout()
plt.suptitle(f"Sample Sensor Data for {sample_eq_id} with Simulated Failures", y=1.02, fontsize=16)
plt.show()

# Plotting failure points
plt.figure(figsize=(15, 5))
failure_points_df = sample_df[sample_df['is_failure'] == 1]
plt.plot(sample_df['timestamp'], sample_df['temperature_C'], label='Temperature')
plt.scatter(failure_points_df['timestamp'], failure_points_df['temperature_C'], color='red', marker='X', s=100, label='Failure Point')
plt.title(f'{sample_eq_id} - Temperature with Failure Points')
plt.xlabel('Timestamp')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()

