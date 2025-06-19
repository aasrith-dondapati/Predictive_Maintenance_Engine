# Predictive_Maintenance_Engine
Time-series forecasting model using XGBoost and Azure ML to predict equipment failure and reduce operational downtime

# Running the Project
It's crucial to run the scripts in the specified order as each subsequent script relies on the outputs of the previous one.

Run Data Simulation:
This script will generate simulated_sensor_data.csv in your project directory and display plots of the simulated sensor data.
python data_simulation.py

Run Data Preprocessing:
This script will load the simulated data, perform feature engineering, define the target variable, and save processed_train_data.csv and processed_test_data.csv. It will also print information about the processed data.
python data_preprocessing.py

Run Model Training and Evaluation:
This script will load the processed data, train the XGBoost model (including hyperparameter tuning), evaluate its performance, and save the trained model (xgboost_predictive_maintenance_model.joblib) and the scaler_for_predictive_maintenance.joblib to your project directory. It will also display various evaluation plots and feature importances.
python model_training.py

After successfully running all three scripts, you will have the generated datasets, the trained XGBoost model, and the scaler saved in your project folder.
