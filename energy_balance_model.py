import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

class EnergyBalanceAnalyzer:
    def __init__(self, df):
        """
        Initialize the EnergyBalanceAnalyzer with preprocessed earthquake data
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Preprocessed earthquake data
        """
        self.df = df
        self.prepare_energy_data()
        
    def prepare_energy_data(self):
        """
        Prepare data for energy analysis:
        - Calculate energy release
        - Calculate energy accumulation
        - Create time windows
        """
        # Calculate energy release using Gutenberg-Richter relation
        self.df['energy_release'] = 10**(1.5 * self.df['mag'] + 4.8)
        
        # Calculate cumulative energy
        self.df['cumulative_energy'] = self.df['energy_release'].cumsum()
        
        # Create time windows for analysis
        self.df['time_window'] = pd.Grouper(key='time', freq='M')
        
    def analyze_energy_distribution(self):
        """
        Analyze the distribution of earthquake energy
        """
        plt.figure(figsize=(12, 6))
        sns.histplot(np.log10(self.df['energy_release']), bins=50)
        plt.title('Distribution of Earthquake Energy (log scale)')
        plt.xlabel('Log10(Energy Release)')
        plt.ylabel('Count')
        plt.show()
        
    def analyze_energy_accumulation(self):
        """
        Analyze energy accumulation over time
        """
        # Calculate monthly energy release
        monthly_energy = self.df.groupby('time_window')['energy_release'].sum()
        
        plt.figure(figsize=(15, 6))
        plt.plot(monthly_energy.index, monthly_energy.values)
        plt.title('Monthly Energy Release')
        plt.xlabel('Time')
        plt.ylabel('Energy Release')
        plt.xticks(rotation=45)
        plt.show()
        
    def analyze_energy_depth_relationship(self):
        """
        Analyze relationship between energy release and depth
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.df, x='depth', y='energy_release', alpha=0.5)
        plt.title('Energy Release vs Depth')
        plt.xlabel('Depth (km)')
        plt.ylabel('Energy Release')
        plt.yscale('log')
        plt.show()
        
    def analyze_regional_energy_balance(self, region_size=1.0):
        """
        Analyze energy balance in different regions
        
        Parameters:
        -----------
        region_size : float
            Size of regions in degrees
        """
        # Create grid of regions
        self.df['region_lat'] = (self.df['latitude'] / region_size).astype(int) * region_size
        self.df['region_lon'] = (self.df['longitude'] / region_size).astype(int) * region_size
        
        # Calculate energy statistics by region
        regional_energy = self.df.groupby(['region_lat', 'region_lon']).agg({
            'energy_release': ['sum', 'mean', 'count'],
            'mag': ['mean', 'max']
        }).reset_index()
        
        # Plot regional energy release
        plt.figure(figsize=(12, 8))
        plt.scatter(
            regional_energy['region_lon'],
            regional_energy['region_lat'],
            c=regional_energy[('energy_release', 'sum')],
            cmap='YlOrRd',
            s=regional_energy[('mag', 'count')] * 10,
            alpha=0.6
        )
        plt.colorbar(label='Total Energy Release')
        plt.title('Regional Energy Release')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()
        
    def analyze_energy_magnitude_relationship(self):
        """
        Analyze relationship between energy release and magnitude
        """
        # Fit linear regression
        X = self.df['mag'].values.reshape(-1, 1)
        y = np.log10(self.df['energy_release'])
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate R-squared
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        
        # Plot relationship
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df['mag'], np.log10(self.df['energy_release']), alpha=0.5)
        plt.plot(X, y_pred, color='red', label=f'R² = {r2:.3f}')
        plt.title('Energy Release vs Magnitude')
        plt.xlabel('Magnitude')
        plt.ylabel('Log10(Energy Release)')
        plt.legend()
        plt.show()
        
    def analyze_energy_time_patterns(self):
        """
        Analyze temporal patterns in energy release
        """
        # Calculate daily energy release
        daily_energy = self.df.groupby(self.df['time'].dt.date)['energy_release'].sum()
        
        # Calculate rolling average
        rolling_avg = daily_energy.rolling(window=30).mean()
        
        plt.figure(figsize=(15, 6))
        plt.plot(daily_energy.index, daily_energy.values, alpha=0.5, label='Daily Energy')
        plt.plot(rolling_avg.index, rolling_avg.values, color='red', label='30-day Average')
        plt.title('Daily Energy Release')
        plt.xlabel('Date')
        plt.ylabel('Energy Release')
        plt.legend()
        plt.show()
        
    def generate_energy_statistics(self):
        """
        Generate statistics about energy release
        """
        stats = {
            'total_energy_release': self.df['energy_release'].sum(),
            'mean_energy_release': self.df['energy_release'].mean(),
            'max_energy_release': self.df['energy_release'].max(),
            'energy_release_per_region': self.df.groupby(['region_lat', 'region_lon'])['energy_release'].sum().mean(),
            'energy_magnitude_correlation': self.df['energy_release'].corr(self.df['mag']),
            'energy_depth_correlation': self.df['energy_release'].corr(self.df['depth'])
        }
        
        return stats

# Example usage
if __name__ == "__main__":
    from earthquake_analysis import EarthquakeAnalyzer
    
    # Initialize analyzer and get preprocessed data
    analyzer = EarthquakeAnalyzer('japanearthquake_cleaned.csv')
    energy_analyzer = EnergyBalanceAnalyzer(analyzer.df)
    
    # Perform energy analysis
    energy_analyzer.analyze_energy_distribution()
    energy_analyzer.analyze_energy_accumulation()
    energy_analyzer.analyze_energy_depth_relationship()
    energy_analyzer.analyze_regional_energy_balance()
    energy_analyzer.analyze_energy_magnitude_relationship()
    energy_analyzer.analyze_energy_time_patterns()
    
    # Print statistics
    stats = energy_analyzer.generate_energy_statistics()
    print("\nEnergy Balance Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}") 