import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

class BikeSharePredictor:
    def __init__(self, hour_file, day_file, station_column=None):
        """
        Initialize the BikeSharePredictor with datasets and station information.
        """
        self.hour_data = pd.read_csv(hour_file)
        self.day_data = pd.read_csv(day_file)
        self.station_column = station_column

    def preprocess_data(self, data, is_hourly=True):
        """
        Preprocess the data for model training:
        - Handle missing values
        - Perform one-hot encoding for categorical variables
        - Scale features
        """
        # Add new features
        data['day_of_week'] = pd.to_datetime(data['dteday']).dt.dayofweek  # Day of the week
        data['day_of_year'] = pd.to_datetime(data['dteday']).dt.dayofyear  # Day of the year
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)  # Weekend flag
        
        # Define 'workingday' feature: 1 if it's neither weekend nor holiday
        if is_hourly:
            data['workingday'] = (data['workingday'] == 0).astype(int)
        
        if is_hourly:
            data['part_of_day'] = pd.cut(
                data['hr'],
                bins=[-1, 5, 11, 17, 23],
                labels=['night', 'morning', 'afternoon', 'evening'],
                right=True
            )
        else:
            data['part_of_day'] = 'daytime'  # Single category for daily data

        # Handle missing values
        data.ffill(inplace=True)  # Forward fill missing values
        
        # One-hot encoding for categorical variables
        categorical_features = ['season', 'weathersit', 'weekday', 'part_of_day', 'workingday'] if is_hourly else ['season', 'weathersit', 'weekday', 'part_of_day']
        if is_hourly:
            categorical_features.append('hr')
        if self.station_column and self.station_column in data.columns:
            categorical_features.append(self.station_column)

        data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

        # Add polynomial and interaction terms
        data['temp_hum_interaction'] = data['temp'] * data['hum']
        data['windspeed_squared'] = data['windspeed'] ** 2
        
        # Include `instant` as a scaled feature
        data['instant_scaled'] = StandardScaler().fit_transform(data[['instant']])
        
        # Select the final features
        features = [
            'instant_scaled', 'yr', 'mnth', 'day_of_week', 'day_of_year', 
            'is_weekend', 'temp', 'atemp', 'hum', 'windspeed', 'temp_hum_interaction', 
            'windspeed_squared', 'cnt', 'workingday'
        ] if is_hourly else [
            'instant_scaled', 'yr', 'mnth', 'day_of_week', 'day_of_year', 
            'is_weekend', 'temp', 'atemp', 'hum', 'windspeed', 'temp_hum_interaction', 
            'windspeed_squared', 'cnt'
        ]
        
        data = data[features]
        
        # Split into features and target
        X = data.drop('cnt', axis=1)
        y = data['cnt']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Reshape data for CNN input
        X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
        X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
        
        return X_train_scaled, X_test_scaled, y_train, y_test

    def build_cnn_model(self, input_shape):
        """
        Build a CNN model for predicting bike demand.
        """
        model = Sequential()
        
        # Add convolutional layers
        model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        
        model.add(Conv1D(32, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        
        # Flatten and add dense layers
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1))  # Output layer for regression
        
        # Compile the model
        model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae', 'mse'])
        return model

    def train_and_evaluate(self, X_train, y_train, X_test, y_test, station_id=None):
        """
        Train the CNN model and evaluate its performance.
        Optionally specify station_id for station-specific training.
        """
        input_shape = (X_train.shape[1], 1)
        model = self.build_cnn_model(input_shape)
        
        # Train the model
        print(f"Training model{' for station: ' + str(station_id) if station_id else ''}...")
        history = model.fit(
            X_train, y_train, 
            epochs=20, batch_size=32, 
            validation_data=(X_test, y_test), 
            verbose=1
        )
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Performance Metrics{' for station: ' + str(station_id) if station_id else ''}:")
        print(f"MAE: {mae}")
        print(f"RMSE: {rmse}")
        print(f"R2 Score: {r2}")
        
        # Plot the training history with performance metrics
        self.plot_training_history(history, mae, rmse, r2)
        
        return model, history

    def plot_training_history(self, history, mae, rmse, r2):
        """
        Plot the training and validation loss over epochs and display performance metrics.
        """
        # Plot the training and validation loss over epochs
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        # Add performance metrics as text to the plot
        metrics_text = f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR2 Score: {r2:.4f}"
        plt.gca().text(0.95, 0.95, metrics_text, horizontalalignment='right', verticalalignment='top', 
                       transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
        
        plt.show()

    def show_tabular_and_graphical_features(self, data):
        """
        Display the dataset features in a tabular format and graphical representations.
        """
        # Tabular format: Show first few rows of the data with features
        print("\nDataset Features (Tabular View):")
        print(data.head())
        
        # Graphical representation: Distribution of key features
        plt.figure(figsize=(18, 10))
        
        # Plot the distribution of 'temp', 'atemp', 'hum', 'windspeed', 'cnt'
        plt.subplot(2, 3, 1)
        sns.histplot(data['temp'], kde=True, color='skyblue')
        plt.title('Temperature Distribution')

        plt.subplot(2, 3, 2)
        sns.histplot(data['atemp'], kde=True, color='salmon')
        plt.title('Feeling Temperature Distribution')

        plt.subplot(2, 3, 3)
        sns.histplot(data['hum'], kde=True, color='green')
        plt.title('Humidity Distribution')

        plt.subplot(2, 3, 4)
        sns.histplot(data['windspeed'], kde=True, color='purple')
        plt.title('Wind Speed Distribution')

        plt.subplot(2, 3, 5)
        sns.histplot(data['cnt'], kde=True, color='orange')
        plt.title('Total Bike Rentals (cnt)')

        plt.subplot(2, 3, 6)
        sns.barplot(x='season', y='cnt', data=data, hue='season')
        plt.title('Bike Rentals by Season')

        plt.tight_layout()
        plt.show()

    def predict_for_station(self, station_id, data):
        """
        Predict bike demand for a specific station.
        """
        if self.station_column not in data.columns:
            raise ValueError(f"Station column '{self.station_column}' is not in the dataset.")
        
        station_data = data[data[self.station_column] == station_id]
        if station_data.empty:
            raise ValueError(f"No data available for station ID: {station_id}")
        
        X_train, X_test, y_train, y_test = self.preprocess_data(station_data, is_hourly=True)
        model, history = self.train_and_evaluate(X_train, y_train, X_test, y_test, station_id=station_id)
        return model, history


# Main function
def main():
    """
    Main function to execute the AI solution for predicting bike usage demand at each station.
    """
    # Paths to datasets
    hour_file_path = 'hour.csv'  # Hourly dataset
    day_file_path = 'day.csv'    # Daily dataset
    station_column = 'station_id'  # Replace with actual column name if available
    
    predictor = BikeSharePredictor(hour_file_path, day_file_path, station_column)
    #######################################
    # Preprocess daily data
    print("\nProcessing Daily Data...")
    X_train_day, X_test_day, y_train_day, y_test_day = predictor.preprocess_data(predictor.day_data, is_hourly=False)
    predictor.train_and_evaluate(X_train_day, y_train_day, X_test_day, y_test_day)
    # Preprocess hourly data
    print("Processing Hourly Data...")
    if station_column in predictor.hour_data.columns:
        unique_stations = predictor.hour_data[station_column].unique()
        for station_id in unique_stations:
            print(f"\nTraining and Evaluating for Station ID: {station_id}")
            predictor.predict_for_station(station_id, predictor.hour_data)
    # Display tabular and graphical features for hour data
    predictor.show_tabular_and_graphical_features(predictor.hour_data)

if __name__ == "__main__":
    main()
