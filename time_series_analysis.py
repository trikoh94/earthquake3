import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

class TimeSeriesAnalyzer:
    def __init__(self, df):
        """
        Initialize the TimeSeriesAnalyzer with preprocessed earthquake data
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Preprocessed earthquake data
        """
        self.df = df
        self.prepare_time_series()
        
    def prepare_time_series(self):
        """
        Prepare time series data for analysis:
        - Aggregate data by day
        - Create features for time series analysis
        """
        # Aggregate earthquakes by day
        self.daily_data = self.df.groupby(self.df['time'].dt.date).agg({
            'mag': ['count', 'mean', 'max'],
            'depth': 'mean',
            'energy': 'sum'
        }).reset_index()
        
        self.daily_data.columns = ['date', 'count', 'mean_mag', 'max_mag', 'mean_depth', 'total_energy']
        self.daily_data['date'] = pd.to_datetime(self.daily_data['date'])
        self.daily_data.set_index('date', inplace=True)
        
    def decompose_time_series(self, column='count', period=30):
        """
        Decompose time series into trend, seasonal, and residual components
        
        Parameters:
        -----------
        column : str
            Column to analyze
        period : int
            Period for seasonal decomposition
        """
        decomposition = seasonal_decompose(self.daily_data[column], period=period)
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(411)
        plt.plot(self.daily_data[column])
        plt.title('Original Time Series')
        
        plt.subplot(412)
        plt.plot(decomposition.trend)
        plt.title('Trend')
        
        plt.subplot(413)
        plt.plot(decomposition.seasonal)
        plt.title('Seasonal')
        
        plt.subplot(414)
        plt.plot(decomposition.resid)
        plt.title('Residual')
        
        plt.tight_layout()
        plt.show()
        
    def fit_arima(self, column='count', order=(5,1,0)):
        """
        Fit ARIMA model to the time series
        
        Parameters:
        -----------
        column : str
            Column to analyze
        order : tuple
            ARIMA order parameters (p,d,q)
        """
        model = ARIMA(self.daily_data[column], order=order)
        self.arima_model = model.fit()
        
        # Make predictions
        forecast = self.arima_model.forecast(steps=30)
        
        # Plot results
        plt.figure(figsize=(15, 6))
        plt.plot(self.daily_data[column], label='Actual')
        plt.plot(pd.date_range(start=self.daily_data.index[-1], periods=31)[1:],
                forecast, label='Forecast', color='red')
        plt.title('ARIMA Forecast')
        plt.legend()
        plt.show()
        
    def fit_prophet(self, column='count'):
        """
        Fit Prophet model to the time series
        
        Parameters:
        -----------
        column : str
            Column to analyze
        """
        # Prepare data for Prophet
        prophet_data = self.daily_data.reset_index()
        prophet_data.columns = ['ds', 'y'] if column == 'count' else ['ds', 'y', 'mean_mag', 'max_mag', 'mean_depth', 'total_energy']
        
        # Fit model
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        model.fit(prophet_data[['ds', 'y']])
        
        # Make predictions
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        
        # Plot results
        fig = model.plot(forecast)
        plt.title('Prophet Forecast')
        plt.show()
        
        # Plot components
        fig = model.plot_components(forecast)
        plt.show()
        
    def evaluate_models(self, column='count', test_size=30):
        """
        Evaluate ARIMA and Prophet models
        
        Parameters:
        -----------
        column : str
            Column to analyze
        test_size : int
            Number of days to use for testing
        """
        # Split data
        train = self.daily_data[column][:-test_size]
        test = self.daily_data[column][-test_size:]
        
        # ARIMA
        arima_model = ARIMA(train, order=(5,1,0))
        arima_fit = arima_model.fit()
        arima_pred = arima_fit.forecast(steps=test_size)
        
        # Prophet
        prophet_data = pd.DataFrame({
            'ds': train.index,
            'y': train.values
        })
        prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        prophet_model.fit(prophet_data)
        
        future = prophet_model.make_future_dataframe(periods=test_size)
        prophet_forecast = prophet_model.predict(future)
        prophet_pred = prophet_forecast['yhat'][-test_size:]
        
        # Calculate metrics
        arima_rmse = np.sqrt(mean_squared_error(test, arima_pred))
        prophet_rmse = np.sqrt(mean_squared_error(test, prophet_pred))
        
        arima_mae = mean_absolute_error(test, arima_pred)
        prophet_mae = mean_absolute_error(test, prophet_pred)
        
        print(f"\nModel Evaluation for {column}:")
        print(f"ARIMA RMSE: {arima_rmse:.2f}")
        print(f"Prophet RMSE: {prophet_rmse:.2f}")
        print(f"ARIMA MAE: {arima_mae:.2f}")
        print(f"Prophet MAE: {prophet_mae:.2f}")
        
        # Plot comparison
        plt.figure(figsize=(15, 6))
        plt.plot(test.index, test, label='Actual')
        plt.plot(test.index, arima_pred, label='ARIMA')
        plt.plot(test.index, prophet_pred, label='Prophet')
        plt.title('Model Comparison')
        plt.legend()
        plt.show()

# Example usage
if __name__ == "__main__":
    from earthquake_analysis import EarthquakeAnalyzer
    
    # Initialize analyzer and get preprocessed data
    analyzer = EarthquakeAnalyzer('japanearthquake_cleaned.csv')
    ts_analyzer = TimeSeriesAnalyzer(analyzer.df)
    
    # Perform time series analysis
    ts_analyzer.decompose_time_series()
    ts_analyzer.fit_arima()
    ts_analyzer.fit_prophet()
    ts_analyzer.evaluate_models() 