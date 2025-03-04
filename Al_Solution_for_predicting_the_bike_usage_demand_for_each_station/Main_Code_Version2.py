import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import seaborn as sns
import plotly.express as x
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
from lightgbm import LGBMRegressor
class BikeShareData:
    def __init__(self, day_file, hour_file):
        """
        Initialize the BikeShareData object with file paths to the datasets.
        """
        self.day_df = pd.read_csv(day_file)  # Load the "day" DataFrame
        self.hour_df = pd.read_csv(hour_file)  # Load the "hour" DataFrame
        self.data_dict = {'Day': self.day_df, 'Hour': self.hour_df}  # Create a dictionary for easy access

    def df_to_pretty_table(self, df, num_rows=5):
        """
        Convert a DataFrame to a PrettyTable for better visualization.
        Args:
            df (pd.DataFrame): The DataFrame to convert.
            num_rows (int): Number of rows to display. Default is 5.
        Returns:
            PrettyTable: A PrettyTable object displaying the data.
        """
        table = PrettyTable()
        table.field_names = df.columns.tolist()  # Set the column headers
        for _, row in df.head(num_rows).iterrows():
            table.add_row(row.tolist())  # Add rows
        return table

    def display_day_data(self, num_rows=5):
        """
        Display data from the "day" DataFrame.
        Args:
            num_rows (int): Number of rows to display. Default is 5.
        """
        print("All of Data of Day\n")
        print("===============================================================\n")
        print(self.df_to_pretty_table(self.day_df, num_rows))
        print("\n===============================================================\n")

    def display_hour_data(self, num_rows=5):
        """
        Display data from the "hour" DataFrame.
        Args:
            num_rows (int): Number of rows to display. Default is 5.
        """
        print("All of Data of Hour\n")
        print("===============================================================\n")
        print(self.df_to_pretty_table(self.hour_df, num_rows))
        print("\n===============================================================\n")

    def display_data_summary(self):
        """
        Display additional information about the datasets, such as shape, info, duplicates, and missing values.
        """
        print("Dataset Rows & Columns Count\n")
        for key, value in self.data_dict.items():
            print(f'{key} dataframe shape : {value.shape}\n')

        print("Dataset Info\n")
        for key, value in self.data_dict.items():
            print(f'--------{key} dataframe info--------\n')
            value.info()
            print("\n")
        print("Dataset Duplicate Value Count\n")
        for key, value in self.data_dict.items():
            print(f'--------{key} dataframe duplicates--------\n')
            print(f'{value[value.duplicated()].shape[0]}\n')

        print("Missing Values/Null Values Count\n")
        for key, value in self.data_dict.items():
            print(f'--------{key} dataframe Missing/Null Value--------\n')
            print(value.isna().sum())
            print("\n")
        print("Dataset Columns\n")
        for key, value in self.data_dict.items():
            print(f'--------{key} dataframe columns--------\n')
            print(value.columns.tolist())
            print("\n")
        print("Hour Dataset Describe\n")
        print(self.hour_df.describe(include='all'))
        print("\nUnique Values in Hour Dataset Columns\n")
        for col in self.hour_df.columns:
            unique_count = self.hour_df[col].nunique()
            print(f"Unique {col}'s count: {unique_count}")
            if unique_count < 25:
                print(f"{col}'s unique values: {self.hour_df[col].unique()}\n")

    def detect_and_visualize_outliers(self):
        """
        Detect outliers in the "cnt" (bike count) column using z-scores and visualize them.
        """
        # Check for outliers in the "count" variable
        plt.figure(figsize=(10, 6))
        plt.boxplot(self.hour_df['cnt'])
        plt.title('Boxplot of Bike Sharing Count')
        plt.ylabel('Count')
        plt.show()
        # Calculate the z-scores for each data point in the "count" variable
        z_scores = (self.hour_df['cnt'] - self.hour_df['cnt'].mean()) / self.hour_df['cnt'].std()
        # Identify outliers based on a threshold (e.g., z-score > 3)
        outliers = self.hour_df[z_scores > 3]
        # Print the outliers
        print("Outliers:")
        print(outliers)
        # Visualize the outliers
        plt.figure(figsize=(10, 6))
        plt.scatter(self.hour_df.index, self.hour_df['cnt'], label='Data')
        plt.scatter(outliers.index, outliers['cnt'], color='red', label='Outliers')
        plt.title('Bike Sharing Count with Outliers')
        plt.xlabel('Index')
        plt.ylabel('Count')
        plt.legend()
        plt.show()

    def evaluate_model(self, true, predicted):
        """
        Function to evaluate model using MAE, RMSE, and R2 score.
        """
        mae = mean_absolute_error(true, predicted)
        mse = mean_squared_error(true, predicted)
        rmse = np.sqrt(mse)
        r2_square = r2_score(true, predicted)
        return mae, rmse, r2_square

    def train_models(self, X_train, y_train, X_test, y_test):
        """
        Train multiple models and evaluate their performance.
        """
        models = {
            'LinearRegression': LinearRegression(),
            'Random Forest': RandomForestRegressor(),
            'Extra Trees Regressor': ExtraTreesRegressor(),
            'Lightgbm': LGBMRegressor(),
            'XGboost': xgb.XGBRegressor()
        }

        model_list = []
        r2_list = []
        rmse_list = []
        mae_list = []

        for model_name, model in models.items():
            model.fit(X_train, y_train)

            # Make Predictions
            y_pred = model.predict(X_test)

            mae, rmse, r2_square = self.evaluate_model(y_test, y_pred)

            print(f"{model_name}")
            model_list.append(model_name)

            print("Model Training Performance")
            print(f"RMSE: {rmse}")
            print(f"MAE: {mae}")
            print(f"R2 score: {r2_square * 100}")

            r2_list.append(r2_square)
            rmse_list.append(rmse)
            mae_list.append(mae)

            print("=" * 35)
            print("\n")

        # Plotting the metrics for all models
        metrics_df = pd.DataFrame({
            'Model': model_list,
            'RMSE': rmse_list,
            'MAE': mae_list,
            'R2 Score': r2_list
        })

        # Plotting bar charts
        metrics_df.set_index('Model', inplace=True)
        metrics_df.plot(kind='bar', figsize=(10, 6))
        plt.title('Model Evaluation Metrics')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.show()

        return model_list, r2_list

    def rename_columns(self):
        """
        Rename columns for better readability.
        """
        self.hour_df.rename(columns={
            'instant': 'rec_id',
            'dteday': 'datetime',
            'holiday': 'is_holiday',
            'workingday': 'is_workingday',
            'weathersit': 'weather_condition',
            'hum': 'humidity',
            'mnth': 'month',
            'cnt': 'total_count',
            'hr': 'hour',
            'yr': 'yr'
        }, inplace=True)
        self.hour_df['datetime'] = pd.to_datetime(self.hour_df['datetime'])
        # Convert columns to appropriate categories
        self.hour_df['season'] = self.hour_df['season'].astype('category')
        self.hour_df['yr'] = self.hour_df['yr'].astype('category')
        self.hour_df['month'] = self.hour_df['month'].astype('category')
        self.hour_df['hour'] = self.hour_df['hour'].astype('category')
        self.hour_df['is_holiday'] = self.hour_df['is_holiday'].astype('category')
        self.hour_df['weekday'] = self.hour_df['weekday'].astype('category')
        self.hour_df['is_workingday'] = self.hour_df['is_workingday'].astype('category')
        self.hour_df['weather_condition'] = self.hour_df['weather_condition'].astype('category')
#######################################################################################################################################################
    def plot_datewise_demand(self):
        """
        Plot the date-wise bike sharing demand.
        """
        count_date = self.hour_df.groupby(['hr'])['cnt'].sum().reset_index()
        fig = plt.figure(figsize=(25, 6))
        ax = plt.axes()
        x = count_date['hr']
        ax.plot(x, count_date['cnt'])
        plt.title('Date-wise Bike Sharing Demand')
        plt.show()

    def plot_monthwise_demand(self):
        """
        Plot the monthly bike sharing demand.
        """
        count_date = self.hour_df.groupby(['mnth'])['cnt'].sum().reset_index()
        fig = plt.figure(figsize=(25, 6))
        ax = plt.axes()
        x = count_date['mnth']
        ax.plot(x, count_date['cnt'])
        plt.title('Monthly Bike Sharing Demand')
        plt.show()
############################################################################################################################################
    # Preprocessing method: Handle categorical variables
    def preprocess_data(self):
        """
        Preprocess data by one-hot encoding categorical features and splitting into X and y.
        """
        # One-hot encode categorical variables
        self.hour_df = pd.get_dummies(self.hour_df, columns=['season', 'yr', 'month', 'hour', 'is_holiday', 'weekday', 'is_workingday', 'weather_condition'], drop_first=True)
        # Target variable
        X = self.hour_df.drop(columns=['total_count', 'rec_id', 'datetime'])
        y = self.hour_df['total_count']
        return X, y
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
# Main function
def main():
    """
    Main function to load and display the bike share data.
    """
    # File paths
    day_file = 'day.csv'
    hour_file = 'hour.csv'
    # Instantiate the BikeShareData class
    bike_data = BikeShareData(day_file, hour_file)
#######(Complete all of Visulization)#############################################
    # Display "day" dataframe
    bike_data.display_day_data()
    # Display "hour" dataframe
    bike_data.display_hour_data()
    # Display additional dataset summaries
    bike_data.display_data_summary()
    # Detect and visualize outliers
    bike_data.detect_and_visualize_outliers()
    # Show tabular and graphical features
    bike_data.show_tabular_and_graphical_features(bike_data.hour_df)
    # Plot various visualizations
    bike_data.plot_datewise_demand()
    bike_data.plot_monthwise_demand()
    # Rename columns for easier access
##############################################################################################
    bike_data.rename_columns()
    # Print the column names to verify them
    print("Column names in hour_df:")
    print(bike_data.hour_df.columns)
    ##########################################################################################################################################
######################################(Complete Model Processing)########################################################### 
###################################################Preprocess data and get X, y#######################################
    X, y = bike_data.preprocess_data()
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train models and evaluate
    model_list, r2_list = bike_data.train_models(X_train, y_train, X_test, y_test)
################################################################################################################################
#########################################################################################################################################
if __name__ == "__main__":
    main()
