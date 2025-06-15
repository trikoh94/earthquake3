import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EarthquakeAnalyzer:
    def __init__(self, file_path):
        """
        Initialize the EarthquakeAnalyzer with the data file path
        
        Parameters:
        -----------
        file_path : str
            Path to the earthquake data CSV file
        """
        self.df = pd.read_csv(file_path)
        self.preprocess_data()
        
    def preprocess_data(self):
        """
        Preprocess the earthquake data:
        - Convert time to datetime
        - Extract additional features
        - Handle missing values
        """
        # Convert time to datetime with mixed format handling
        self.df['time'] = pd.to_datetime(self.df['time'], format='mixed')
        
        # Extract year, month, day, hour
        self.df['year'] = self.df['time'].dt.year
        self.df['month'] = self.df['time'].dt.month
        self.df['day'] = self.df['time'].dt.day
        self.df['hour'] = self.df['time'].dt.hour
        
        # Calculate energy (using Gutenberg-Richter relation)
        self.df['energy'] = 10**(1.5 * self.df['mag'] + 4.8)
        
        # Handle missing values
        self.df = self.df.fillna({
            'depth': self.df['depth'].median(),
            'mag': self.df['mag'].median()
        })
        
    def basic_statistics(self):
        """
        Calculate and return basic statistics of the earthquake data
        
        Returns:
        --------
        dict
            Dictionary containing basic statistics
        """
        stats = {
            'total_earthquakes': len(self.df),
            'date_range': (self.df['time'].min(), self.df['time'].max()),
            'magnitude_range': (self.df['mag'].min(), self.df['mag'].max()),
            'depth_range': (self.df['depth'].min(), self.df['depth'].max()),
            'mean_magnitude': self.df['mag'].mean(),
            'mean_depth': self.df['depth'].mean()
        }
        return stats
    
    def plot_magnitude_distribution(self):
        """
        Plot the distribution of earthquake magnitudes
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.df, x='mag', bins=30)
        plt.title('Distribution of Earthquake Magnitudes')
        plt.xlabel('Magnitude')
        plt.ylabel('Count')
        plt.show()
        
    def plot_depth_distribution(self):
        """
        Plot the distribution of earthquake depths
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.df, x='depth', bins=30)
        plt.title('Distribution of Earthquake Depths')
        plt.xlabel('Depth (km)')
        plt.ylabel('Count')
        plt.show()
        
    def plot_time_series(self):
        """
        Plot the time series of earthquake occurrences
        """
        # Group by date and count earthquakes
        daily_counts = self.df.groupby(self.df['time'].dt.date).size()
        
        plt.figure(figsize=(15, 6))
        daily_counts.plot()
        plt.title('Daily Earthquake Frequency')
        plt.xlabel('Date')
        plt.ylabel('Number of Earthquakes')
        plt.show()
        
    def plot_magnitude_vs_depth(self):
        """
        Plot the relationship between magnitude and depth
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.df, x='depth', y='mag', alpha=0.5)
        plt.title('Magnitude vs Depth')
        plt.xlabel('Depth (km)')
        plt.ylabel('Magnitude')
        plt.show()

# Example usage
if __name__ == "__main__":
    analyzer = EarthquakeAnalyzer('japanearthquake_cleaned.csv')
    
    # Print basic statistics
    stats = analyzer.basic_statistics()
    print("\nBasic Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Generate plots
    analyzer.plot_magnitude_distribution()
    analyzer.plot_depth_distribution()
    analyzer.plot_time_series()
    analyzer.plot_magnitude_vs_depth() 