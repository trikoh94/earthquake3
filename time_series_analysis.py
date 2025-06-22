import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
from scipy import stats
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
        - Calculate inter-event times
        """
        # Sort by time
        self.df = self.df.sort_values('time')
        
        # Calculate inter-event times (more meaningful for earthquakes)
        self.df['time_diff'] = self.df['time'].diff()
        self.df['time_diff_hours'] = self.df['time_diff'].dt.total_seconds() / 3600
        
        # Aggregate earthquakes by day
        self.daily_data = self.df.groupby(self.df['time'].dt.date).agg({
            'mag': ['count', 'mean', 'max'],
            'depth': 'mean',
            'energy': 'sum',
            'time_diff_hours': 'mean'
        }).reset_index()
        
        self.daily_data.columns = ['date', 'count', 'mean_mag', 'max_mag', 'mean_depth', 'total_energy', 'avg_interval']
        self.daily_data['date'] = pd.to_datetime(self.daily_data['date'])
        self.daily_data.set_index('date', inplace=True)
        
        # Fill missing days with 0 counts
        self.daily_data = self.daily_data.asfreq('D').fillna(0)
        
    def analyze_inter_event_times(self):
        """
        Analyze time intervals between earthquakes (more meaningful than daily counts)
        """
        # Remove first row (no previous event)
        intervals = self.df['time_diff_hours'].dropna()
        
        # Basic statistics
        stats_dict = {
            'mean_interval': intervals.mean(),
            'median_interval': intervals.median(),
            'std_interval': intervals.std(),
            'min_interval': intervals.min(),
            'max_interval': intervals.max(),
            'total_events': len(self.df),
            'total_time_hours': intervals.sum()
        }
        
        # Test for exponential distribution (Poisson process)
        # If earthquakes follow Poisson process, intervals should be exponentially distributed
        lambda_est = 1 / intervals.mean()
        ks_stat, p_value = stats.kstest(intervals, 'expon', args=(0, 1/lambda_est))
        
        stats_dict['poisson_p_value'] = p_value
        stats_dict['is_poisson_like'] = p_value > 0.05
        
        # Detect clustering periods
        # Short intervals indicate clustering (aftershock sequences)
        short_interval_threshold = intervals.quantile(0.1)  # Bottom 10%
        clustered_events = intervals[intervals <= short_interval_threshold]
        stats_dict['clustered_ratio'] = len(clustered_events) / len(intervals)
        
        return stats_dict, intervals
        
    def analyze_magnitude_time_patterns(self):
        """
        Analyze how earthquake magnitudes change over time
        """
        # Calculate magnitude differences between consecutive events
        mag_diffs = self.df['mag'].diff().dropna()
        
        # Detect foreshock-aftershock patterns
        # Increasing magnitudes might indicate foreshock activity
        # Decreasing magnitudes might indicate aftershock sequences
        
        increasing_sequences = (mag_diffs > 0).sum()
        decreasing_sequences = (mag_diffs < 0).sum()
        total_sequences = len(mag_diffs)
        
        # Calculate rolling average of magnitudes
        self.df['mag_rolling_avg'] = self.df['mag'].rolling(window=10, center=True).mean()
        
        # Detect significant magnitude changes
        mag_change_threshold = mag_diffs.std() * 2
        significant_changes = mag_diffs[abs(mag_diffs) > mag_change_threshold]
        
        return {
            'increasing_ratio': increasing_sequences / total_sequences,
            'decreasing_ratio': decreasing_sequences / total_sequences,
            'significant_changes': len(significant_changes),
            'avg_mag_change': mag_diffs.mean(),
            'mag_change_std': mag_diffs.std()
        }
        
    def analyze_stress_accumulation_patterns(self):
        """
        Analyze patterns that might indicate stress accumulation and release
        """
        # Calculate cumulative energy release
        self.df['cumulative_energy'] = self.df['energy'].cumsum()
        
        # Detect periods of high energy release (stress release)
        energy_threshold = self.df['energy'].quantile(0.9)  # Top 10%
        high_energy_events = self.df[self.df['energy'] > energy_threshold]
        
        # Calculate energy release rate
        self.df['energy_rate'] = self.df['energy'] / self.df['time_diff_hours']
        
        # Detect quiescent periods (low activity before major events)
        # Look for periods of low activity followed by high magnitude events
        quiescent_threshold = self.df['time_diff_hours'].quantile(0.8)  # Top 20% intervals
        quiescent_periods = self.df[self.df['time_diff_hours'] > quiescent_threshold]
        
        # Check if quiescent periods are followed by high magnitude events
        quiescent_followed_by_major = 0
        for i, period in quiescent_periods.iterrows():
            # Look for high magnitude events within 24 hours after quiescent period
            future_events = self.df[
                (self.df['time'] > period['time']) & 
                (self.df['time'] <= period['time'] + pd.Timedelta(hours=24)) &
                (self.df['mag'] >= 5.0)
            ]
            if len(future_events) > 0:
                quiescent_followed_by_major += 1
        
        return {
            'high_energy_events': len(high_energy_events),
            'avg_energy_rate': self.df['energy_rate'].mean(),
            'quiescent_periods': len(quiescent_periods),
            'quiescent_followed_by_major': quiescent_followed_by_major,
            'quiescent_major_ratio': quiescent_followed_by_major / len(quiescent_periods) if len(quiescent_periods) > 0 else 0
        }
        
    def test_stationarity(self, column='count'):
        """
        Test if the time series is stationary (important for ARIMA)
        """
        series = self.daily_data[column]
        
        # Augmented Dickey-Fuller test
        adf_result = adfuller(series)
        
        # KPSS test
        kpss_result = kpss(series)
        
        return {
            'adf_statistic': adf_result[0],
            'adf_p_value': adf_result[1],
            'adf_stationary': adf_result[1] < 0.05,
            'kpss_statistic': kpss_result[0],
            'kpss_p_value': kpss_result[1],
            'kpss_stationary': kpss_result[1] > 0.05
        }
        
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
        # Test stationarity first
        stationarity = self.test_stationarity(column)
        
        decomposition = seasonal_decompose(self.daily_data[column], period=period)
        
        plt.figure(figsize=(15, 12))
        
        plt.subplot(511)
        plt.plot(self.daily_data[column])
        plt.title(f'Original Time Series ({column})')
        plt.text(0.02, 0.98, f'Stationary: {stationarity["adf_stationary"]}', 
                transform=plt.gca().transAxes, verticalalignment='top')
        
        plt.subplot(512)
        plt.plot(decomposition.trend)
        plt.title('Trend')
        
        plt.subplot(513)
        plt.plot(decomposition.seasonal)
        plt.title('Seasonal')
        
        plt.subplot(514)
        plt.plot(decomposition.resid)
        plt.title('Residual')
        
        plt.subplot(515)
        plt.hist(decomposition.resid.dropna(), bins=30, alpha=0.7)
        plt.title('Residual Distribution')
        
        plt.tight_layout()
        plt.show()
        
        return stationarity
        
    def fit_arima(self, column='count', order=(5,1,0)):
        """
        Fit ARIMA model to the time series with enhanced diagnostics
        """
        # Test stationarity first
        stationarity = self.test_stationarity(column)
        
        if not stationarity['adf_stationary']:
            print("Warning: Series is not stationary. Consider differencing.")
        
        model = ARIMA(self.daily_data[column], order=order)
        self.arima_model = model.fit()
        
        # Model diagnostics
        residuals = self.arima_model.resid
        
        # Test residuals for normality
        _, normality_p = stats.normaltest(residuals.dropna())
        
        # Ljung-Box test for autocorrelation in residuals
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_stat, lb_p = acorr_ljungbox(residuals.dropna(), lags=10, return_df=False)
        
        # Make predictions
        forecast = self.arima_model.forecast(steps=30)
        
        # Plot results with diagnostics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Original series and forecast
        axes[0,0].plot(self.daily_data[column], label='Actual')
        axes[0,0].plot(pd.date_range(start=self.daily_data.index[-1], periods=31)[1:],
                forecast, label='Forecast', color='red')
        axes[0,0].set_title('ARIMA Forecast')
        axes[0,0].legend()
        
        # Residuals
        axes[0,1].plot(residuals)
        axes[0,1].set_title('Residuals')
        axes[0,1].axhline(y=0, color='r', linestyle='--')
        
        # Residuals histogram
        axes[1,0].hist(residuals.dropna(), bins=30, alpha=0.7)
        axes[1,0].set_title(f'Residual Distribution (p={normality_p:.3f})')
        
        # Q-Q plot
        stats.probplot(residuals.dropna(), dist="norm", plot=axes[1,1])
        axes[1,1].set_title('Q-Q Plot')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'stationarity': stationarity,
            'normality_p': normality_p,
            'ljung_box_p': lb_p[0],
            'model_aic': self.arima_model.aic,
            'model_bic': self.arima_model.bic
        }
        
    def evaluate_models(self, test_size=30):
        """
        Evaluate ARIMA model with earthquake-specific metrics
        """
        # Split data
        train = self.daily_data['count'][:-test_size]
        test = self.daily_data['count'][-test_size:]
        
        # ARIMA
        arima_model = ARIMA(train, order=(5,1,0))
        arima_fit = arima_model.fit()
        arima_pred = arima_fit.forecast(steps=test_size)
        
        # Calculate metrics
        arima_rmse = np.sqrt(mean_squared_error(test, arima_pred))
        arima_mae = mean_absolute_error(test, arima_pred)
        
        # Earthquake-specific metrics
        # Direction accuracy (predicting increase/decrease correctly)
        actual_direction = np.diff(test) > 0
        pred_direction = np.diff(arima_pred) > 0
        direction_accuracy = np.mean(actual_direction == pred_direction)
        
        # High activity detection accuracy
        high_activity_threshold = test.quantile(0.8)
        actual_high = test > high_activity_threshold
        pred_high = arima_pred > high_activity_threshold
        high_activity_accuracy = np.mean(actual_high == pred_high)
        
        # Print results
        print("=== ARIMA Model Evaluation ===")
        print(f"RMSE: {arima_rmse:.2f}")
        print(f"MAE: {arima_mae:.2f}")
        print(f"Direction Accuracy: {direction_accuracy:.3f}")
        print(f"High Activity Detection Accuracy: {high_activity_accuracy:.3f}")
        
        # Plot predictions
        plt.figure(figsize=(15, 6))
        plt.plot(test.index, test.values, label='Observed', marker='o')
        plt.plot(test.index, arima_pred, label='ARIMA', marker='s')
        plt.axhline(y=high_activity_threshold, color='r', linestyle='--', label='High Activity Threshold')
        plt.title('Model Comparison with Earthquake-Specific Metrics')
        plt.xlabel('Date')
        plt.ylabel('Earthquake Count')
        plt.legend()
        plt.show()
        
        return {
            'rmse': arima_rmse,
            'mae': arima_mae,
            'direction_accuracy': direction_accuracy,
            'high_activity_accuracy': high_activity_accuracy
        }

# Example usage
if __name__ == "__main__":
    from earthquake_analysis import EarthquakeAnalyzer
    
    # Initialize analyzer and get preprocessed data
    analyzer = EarthquakeAnalyzer('japanearthquake_cleaned.csv')
    ts_analyzer = TimeSeriesAnalyzer(analyzer.df)
    
    # Perform enhanced time series analysis
    print("=== Inter-Event Time Analysis ===")
    interval_stats, intervals = ts_analyzer.analyze_inter_event_times()
    print(interval_stats)
    
    print("\n=== Magnitude Time Patterns ===")
    mag_patterns = ts_analyzer.analyze_magnitude_time_patterns()
    print(mag_patterns)
    
    print("\n=== Stress Accumulation Patterns ===")
    stress_patterns = ts_analyzer.analyze_stress_accumulation_patterns()
    print(stress_patterns)
    
    # Traditional time series analysis (with caveats)
    print("\n=== Traditional Time Series Analysis ===")
    stationarity = ts_analyzer.decompose_time_series()
    print(f"Stationarity test results: {stationarity}")
    
    diagnostics = ts_analyzer.fit_arima()
    print(f"Model diagnostics: {diagnostics}")
    
    evaluation = ts_analyzer.evaluate_models()
    print(f"Model evaluation: {evaluation}") 