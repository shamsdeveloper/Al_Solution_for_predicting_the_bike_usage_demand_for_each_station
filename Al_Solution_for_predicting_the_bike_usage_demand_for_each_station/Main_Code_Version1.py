import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import seaborn as sns
import plotly.express as x
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from lightgbm import LGBMRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# BikeShareData Class
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
###########################################################################################################################
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
            'yr': 'year'
        }, inplace=True)
        self.hour_df['datetime'] = pd.to_datetime(self.hour_df['datetime'])
        # Convert columns to appropriate categories
        self.hour_df['season'] = self.hour_df['season'].astype('category')
        self.hour_df['year'] = self.hour_df['year'].astype('category')
        self.hour_df['month'] = self.hour_df['month'].astype('category')
        self.hour_df['hour'] = self.hour_df['hour'].astype('category')
        self.hour_df['is_holiday'] = self.hour_df['is_holiday'].astype('category')
        self.hour_df['weekday'] = self.hour_df['weekday'].astype('category')
        self.hour_df['is_workingday'] = self.hour_df['is_workingday'].astype('category')
        self.hour_df['weather_condition'] = self.hour_df['weather_condition'].astype('category')
    
    def plot_seasonwise_hourly_distribution(self):
        """
        Plot season-wise hourly distribution of bike sharing counts.
        """
        fig, ax = plt.subplots(figsize=(15, 6))
        sns.set_style('white')
        sns.pointplot(x='hour', y='total_count', data=self.hour_df[['hour', 'total_count', 'season']], hue='season', ax=ax)
        ax.set_title('Season-wise Hourly Distribution of Counts')
        plt.show()

    def plot_weekdaywise_hourly_distribution(self):
        """
        Plot weekday-wise hourly distribution of bike sharing counts.
        """
        fig, ax = plt.subplots(figsize=(20, 8))
        sns.pointplot(x='hour', y='total_count', data=self.hour_df[['hour', 'total_count', 'weekday']], hue='weekday')
        ax.set_title('Weekday-wise Hourly Distribution of Counts')
        plt.show()

    def plot_monthly_distribution(self):
        """
        Plot the monthly distribution of bike sharing counts.
        """
        fig, ax = plt.subplots(figsize=(20, 8))
        sns.barplot(x='month', y='total_count', data=self.hour_df[['month', 'total_count']], ax=ax)
        ax.set_title('Monthly Distribution of Counts')
        plt.show()

    def plot_seasonwise_monthly_distribution(self):
        """
        Plot season-wise monthly distribution of bike sharing counts.
        """
        fig, ax = plt.subplots(figsize=(20, 8))
        sns.barplot(x='month', y='total_count', data=self.hour_df[['month', 'total_count', 'season']], hue='season', ax=ax)
        ax.set_title('Season-wise Monthly Distribution of Counts')
        plt.show()

    def plot_yearly_distribution(self):
        """
        Plot the yearly distribution of bike sharing counts.
        """
        fig, ax = plt.subplots(figsize=(20, 8))
        sns.violinplot(x='year', y='total_count', data=self.hour_df[['year', 'total_count']])
        ax.set_title('Yearly-wise Distribution of Counts')
        plt.show()

    def plot_holiday_workingday_distribution(self):
        """
        Plot the holiday and working day distribution of bike sharing counts.
        """
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 5))
        sns.barplot(data=self.hour_df, x='is_holiday', y='total_count', hue='season', ax=ax1)
        ax1.set_title('Holiday-wise Distribution of Counts')
        sns.barplot(data=self.hour_df, x='is_workingday', y='total_count', hue='season', ax=ax2)
        ax2.set_title('Working Day-wise Distribution of Counts')
        plt.show()
    
    def plot_correlation_matrix(self):
        """
        Plot the correlation matrix of bike sharing data attributes.
        """
        correMtr = self.hour_df[["temp", "atemp", "humidity", "windspeed", "total_count"]].corr()
        mask = np.array(correMtr)
        mask[np.tril_indices_from(mask)] = False
        fig, ax = plt.subplots(figsize=(20, 5))
        sns.heatmap(correMtr, mask=mask, vmax=0.8, square=True, annot=True, ax=ax)
        ax.set_title('Correlation Matrix of Attributes')
        plt.show()

    # New visualizations

    def plot_avg_bike_demand_during_holidays(self):
        """
        Plot barplot of average bike demand during holidays.
        """
        y = self.hour_df.groupby('is_holiday')['total_count'].mean().reset_index()
        fig = plt.subplots(figsize=(6, 4))
        sns.barplot(x='is_holiday', y='total_count', data=y).set_title('Average Bike Demand during Holidays')
        plt.show()

    def plot_hourwise_demand_during_holidays(self):
        """
        Plot hourwise bike demand during holidays.
        """
        c = self.hour_df.groupby('hour')['total_count'].mean().reset_index()
        fig = plt.subplots(figsize=(12, 6))
        sns.barplot(x='hour', y='total_count', data=c).set_title('Hourwise Bike Sharing Demand during Holidays')
        plt.show()

    def plot_hourwise_demand_percentage_during_holidays(self):
        """
        Pie chart showing percentage of demand per hour during holidays.
        """
        c = self.hour_df.groupby('hour')['total_count'].mean().reset_index()
        fig = x.pie(c, values='total_count', names=c['hour'].unique(), title='Percentage of Hourwise Bike Sharing Demand during Holidays')
        fig.update_layout(autosize=False, width=600, height=400, legend_title_text="Hour of the Day")
        fig.update_layout(autosize=True, width=750, height=600)
        fig.show()

    def plot_weekend_hourwise_demand(self):
        """
        Barplot and pie chart of hourwise bike demand during weekends.
        """
        y = self.hour_df.groupby(['weekday', 'hour'])['total_count'].mean().reset_index()
        y = y[y['weekday'] == 0]  # Select weekend (0 represents weekend)
        
        fig = plt.subplots(figsize=(10, 6))
        sns.barplot(x='hour', y='total_count', data=y).set_title('Average Bike Demand during Weekend')
        plt.show()

        fig = x.pie(y, values='total_count', names=y['hour'].unique(), title='Percentage of Average Hourwise Bike Demand during Weekend')
        fig.update_layout(autosize=False, width=600, height=400, legend_title_text="Hour of the Day")
        fig.update_layout(autosize=True, width=750, height=600)
        fig.show()
#########################################################################################################
# ModelTraining Class
class ModelTraining:
    def __init__(self, day_file, hour_file):
        # Load the data
        self.day_file = day_file
        self.hour_file = hour_file
        self.df = pd.read_csv(self.day_file)
        self._preprocess_data()
        
        # Models
        self.models = {
            'LinearRegression': LinearRegression(),
            'Random Forest': RandomForestRegressor(),
            'Extra Trees Regressor': ExtraTreesRegressor(),
            'Lightgbm': LGBMRegressor(),
            'XGboost': xgb.XGBRegressor()
        }
        
    def _preprocess_data(self):
        # Dropping irrelevant columns
        self.df = self.df.drop(labels=['instant', 'dteday', 'casual', 'registered'], axis=1)
        # Separating Independent and Dependent Features
        self.X = self.df.drop(labels=['cnt'], axis=1)
        self.Y = self.df[['cnt']]
        # Defining categorical and numerical columns
        self.categorical_cols = self.X.select_dtypes(include='object').columns
        self.numerical_cols = self.X.select_dtypes(exclude='object').columns
        # Pipelines for Preprocessing
        self.num_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
        self.cat_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('ordinalencoder', OrdinalEncoder(categories=[])), ('scaler', StandardScaler())])
        # Column transformer for combining pipelines
        self.preprocessor = ColumnTransformer([('num_pipeline', self.num_pipeline, self.numerical_cols), ('cat_pipeline', self.cat_pipeline, self.categorical_cols)])
        # Train-Test Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.30, random_state=30)
        # Scaling datasets
        self.X_train = pd.DataFrame(self.preprocessor.fit_transform(self.X_train), columns=self.preprocessor.get_feature_names_out())
        self.X_test = pd.DataFrame(self.preprocessor.transform(self.X_test), columns=self.preprocessor.get_feature_names_out())

    def evaluate_model(self, true, predicted):
        mae = mean_absolute_error(true, predicted)
        mse = mean_squared_error(true, predicted)
        rmse = np.sqrt(mse)
        r2_square = r2_score(true, predicted)
        return mae, rmse, r2_square
    
    def train_models(self):
        r2_list = []
        rmse_list = []
        mae_list = []
        
        for model_name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            # Make Predictions
            y_pred = model.predict(self.X_test)
            # Evaluate the model
            mae, rmse, r2_square = self.evaluate_model(self.y_test, y_pred)
            # Append metrics for later visualization
            r2_list.append(r2_square)
            rmse_list.append(rmse)
            mae_list.append(mae)
            
            print(f'{model_name} Model Training Performance')
            print(f"RMSE: {rmse}")
            print(f"MAE: {mae}")
            print(f"R2 score: {r2_square * 100:.2f}%")
            print('=' * 35)

        # Create a DataFrame for the table
        df = pd.DataFrame({
            'Model': list(self.models.keys()),
            'RMSE': rmse_list,
            'MAE': mae_list,
            'R2 Score': r2_list
        })

        # Plotting the graph
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot RMSE, MAE, and R2 Score as bars
        ax.bar(self.models.keys(), rmse_list, label='RMSE', color='blue', alpha=0.6, width=0.2, align='center')
        ax.bar(self.models.keys(), mae_list, label='MAE', color='green', alpha=0.6, width=0.2, align='edge')
        ax.bar(self.models.keys(), r2_list, label='R2 Score', color='red', alpha=0.6, width=0.2, align='edge')

        # Adding labels and title
        ax.set_xlabel('Models')
        ax.set_ylabel('Values')
        ax.set_title('Model Performance Comparison')
        ax.legend()

        # Display the table below the graph
        plt.table(cellText=df.values, colLabels=df.columns, loc='bottom', cellLoc='center', bbox=[0.1, -0.4, 0.8, 0.3])
        # Adjust layout to make space for the table
        plt.subplots_adjust(bottom=0.3)
        # Show the plot and table
        plt.show()
        return r2_list
def main():
    # File paths
    day_file = 'day.csv'
    hour_file = 'hour.csv'
    # Initialize and run the model training
    model_trainer = ModelTraining(day_file, hour_file)
    r2_scores = model_trainer.train_models()
    print(f"R2 Scores of All Models: {r2_scores}")
    ################################################################################################
    # Instantiate the BikeShareData class
    bike_data = BikeShareData(day_file, hour_file)
    # Display "day" dataframe
    bike_data.display_day_data()
    # Display "hour" dataframe
    bike_data.display_hour_data()
    # Display additional dataset summaries
    bike_data.display_data_summary()
    # Detect and visualize outliers
    bike_data.detect_and_visualize_outliers()
    ############################################################################################
    # Plot datewise bike sharing demand
    bike_data.plot_datewise_demand()
    # Plot monthly bike sharing demand
    bike_data.plot_monthwise_demand()
    # Rename columns for better readability
    bike_data.rename_columns()
    # Plot season-wise hourly distribution
    bike_data.plot_seasonwise_hourly_distribution()
    # Plot weekday-wise hourly distribution
    bike_data.plot_weekdaywise_hourly_distribution()
    # Plot monthly distribution
    bike_data.plot_monthly_distribution()
    # Plot season-wise monthly distribution
    bike_data.plot_seasonwise_monthly_distribution()
    # Plot yearly distribution
    bike_data.plot_yearly_distribution()
    # Plot holiday and working day distribution
    bike_data.plot_holiday_workingday_distribution()
    # Plot correlation matrix
    bike_data.plot_correlation_matrix()
    # New visualizations
    bike_data.plot_avg_bike_demand_during_holidays()
    bike_data.plot_hourwise_demand_during_holidays()
    bike_data.plot_hourwise_demand_percentage_during_holidays()
    bike_data.plot_weekend_hourwise_demand()
if __name__ == "__main__":
    main()
