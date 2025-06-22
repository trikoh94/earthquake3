import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from math import radians, sin, cos, sqrt, atan2
from scipy.optimize import curve_fit

class EarthquakeChainAnalyzer:
    def __init__(self, df):
        """
        Initialize the EarthquakeChainAnalyzer with preprocessed earthquake data
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Preprocessed earthquake data
        """
        self.df = df
        self.prepare_data()
        
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees)
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        distance = 6371 * c  # Radius of earth in kilometers
        
        return distance
    
    def prepare_data(self):
        """
        Prepare data for chain analysis:
        - Sort by time
        - Calculate time differences
        - Calculate spatial distances using Haversine
        """
        # Sort by time
        self.df = self.df.sort_values('time')
        
        # Calculate time differences
        self.df['time_diff'] = self.df['time'].diff()
        
        # Calculate spatial distances using Haversine
        self.df['spatial_diff'] = self.df.apply(
            lambda row: self.haversine_distance(
                row['latitude'], row['longitude'],
                self.df.loc[row.name - 1, 'latitude'] if row.name > 0 else row['latitude'],
                self.df.loc[row.name - 1, 'longitude'] if row.name > 0 else row['longitude']
            ) if row.name > 0 else 0,
            axis=1
        )
        
    def omori_law(self, t, K, c, p):
        """
        Omori's Law for aftershock decay
        n(t) = K/(t+c)^p
        """
        return K / (t + c)**p
    
    def fit_omori_law(self, main_shock_time, time_window=30):
        """
        Fit Omori's Law to aftershock sequence
        
        Parameters:
        -----------
        main_shock_time : datetime
            Time of the main shock
        time_window : int
            Number of days to analyze after main shock
        """
        # Get aftershocks within time window
        mask = (
            (self.df['time'] > main_shock_time) &
            (self.df['time'] <= main_shock_time + timedelta(days=time_window))
        )
        aftershocks = self.df[mask].copy()
        
        if len(aftershocks) < 3:  # Reduced from 5 to 3 for more lenient fitting
            return None, None, None
        
        # Calculate time differences in hours
        t = (aftershocks['time'] - main_shock_time).dt.total_seconds() / 3600
        
        # Count cumulative aftershocks
        n = np.arange(1, len(t) + 1)
        
        try:
            # Fit Omori's Law with more robust initial parameters
            popt, pcov = curve_fit(
                self.omori_law,
                t,
                n,
                p0=[len(aftershocks), 1.0, 1.0],  # Better initial guess
                maxfev=10000,
                bounds=([0, 0, 0.5], [np.inf, np.inf, 3.0])  # Reasonable bounds
            )
            
            # Calculate p-value from covariance matrix
            # For curve fitting, we can use the reduced chi-squared statistic
            residuals = n - self.omori_law(t, *popt)
            chi_squared = np.sum(residuals**2)
            degrees_of_freedom = len(t) - len(popt)
            reduced_chi_squared = chi_squared / degrees_of_freedom if degrees_of_freedom > 0 else float('inf')
            
            # Calculate p-value (simplified approach)
            # A lower reduced chi-squared indicates a better fit
            # We'll use a threshold-based approach for p-value
            if reduced_chi_squared < 1.0:
                p_value = 0.01  # Very good fit
            elif reduced_chi_squared < 2.0:
                p_value = 0.05  # Good fit
            elif reduced_chi_squared < 5.0:
                p_value = 0.1   # Moderate fit
            else:
                p_value = 0.5   # Poor fit
            
            return popt, pcov, p_value
        except Exception as e:
            print(f"Error fitting Omori's Law: {e}")
            return None, None, None
    
    def identify_aftershocks(self, main_shock_mag=5.0, time_window=7, distance_window=100, max_mainshocks=100):
        """
        Identify aftershocks following main shocks using improved criteria
        
        Parameters:
        -----------
        main_shock_mag : float
            Minimum magnitude for main shock
        time_window : int
            Number of days to look for aftershocks
        distance_window : float
            Maximum distance (km) for aftershocks
        max_mainshocks : int, optional
            Maximum number of mainshocks to analyze (None for all)
        """
        main_shocks = self.df[self.df['mag'] >= main_shock_mag].copy()
        # Apply user-specified limit for performance
        main_shocks = main_shocks.sort_values('time', ascending=False)
        if max_mainshocks is not None:
            main_shocks = main_shocks.head(max_mainshocks)
        aftershocks = []
        omori_params = []
        
        for _, main_shock in main_shocks.iterrows():
            # Find potential aftershocks
            mask = (
                (self.df['time'] > main_shock['time']) &
                (self.df['time'] <= main_shock['time'] + timedelta(days=time_window)) &
                (self.df['mag'] < main_shock['mag'])
            )
            
            potential_aftershocks = self.df[mask]
            
            # Calculate distances using Haversine
            distances = potential_aftershocks.apply(
                lambda row: self.haversine_distance(
                    row['latitude'], row['longitude'],
                    main_shock['latitude'], main_shock['longitude']
                ),
                axis=1
            )
            
            # Filter by distance
            aftershocks.extend(
                potential_aftershocks[distances <= distance_window].index.tolist()
            )
            
            # Fit Omori's Law
            params, _, p_value = self.fit_omori_law(main_shock['time'], time_window)
            if params is not None:
                omori_params.append({
                    'main_shock_time': main_shock['time'],
                    'main_shock_mag': main_shock['mag'],
                    'K': params[0],
                    'c': params[1],
                    'p': params[2],
                    'p_value': p_value
                })
        
        self.df['is_aftershock'] = self.df.index.isin(aftershocks)
        self.omori_params = pd.DataFrame(omori_params)
        
    def analyze_clusters(self, eps=0.5, min_samples=5, time_eps=24):
        """
        Analyze spatiotemporal clusters of earthquakes using ST-DBSCAN
        
        Parameters:
        -----------
        eps : float
            Maximum spatial distance between samples (km)
        min_samples : int
            Minimum number of samples for DBSCAN
        time_eps : float
            Maximum time difference in hours
        """
        # Prepare features for clustering
        features = self.df[['latitude', 'longitude', 'depth', 'mag']].copy()
        
        # Convert time to hours since start
        features['time_hours'] = (self.df['time'] - self.df['time'].min()).dt.total_seconds() / 3600
        
        # Scale features - normalize spatial and temporal features separately
        scaler_spatial = StandardScaler()
        scaler_temporal = StandardScaler()
        
        # Scale spatial features
        spatial_features = features[['latitude', 'longitude', 'depth', 'mag']].values
        spatial_features_scaled = scaler_spatial.fit_transform(spatial_features)
        
        # Scale temporal features
        temporal_features = features[['time_hours']].values
        temporal_features_scaled = scaler_temporal.fit_transform(temporal_features)
        
        # Combine features
        features_scaled = np.column_stack([spatial_features_scaled, temporal_features_scaled])
        
        # Calculate eps values for scaled features
        # For spatial features: eps_km / typical spatial scale
        # For temporal features: time_eps_hours / typical temporal scale
        spatial_eps = eps / 100.0  # Assuming typical spatial scale of ~100km
        temporal_eps = time_eps / 24.0  # Assuming typical temporal scale of ~24 hours
        
        # Use the smaller eps for DBSCAN (more restrictive)
        eps_scaled = min(spatial_eps, temporal_eps)
        
        # Perform DBSCAN clustering
        clustering = DBSCAN(
            eps=eps_scaled,
            min_samples=min_samples,
            metric='euclidean'
        ).fit(features_scaled)
        
        self.df['cluster'] = clustering.labels_
        
    def plot_clusters(self):
        """
        Plot earthquake clusters with improved visualization
        """
        plt.figure(figsize=(12, 8))
        
        # Plot non-clustered points
        mask = self.df['cluster'] == -1
        plt.scatter(
            self.df[mask]['longitude'],
            self.df[mask]['latitude'],
            c='gray',
            alpha=0.5,
            label='Noise'
        )
        
        # Plot clusters
        for cluster in sorted(self.df['cluster'].unique()):
            if cluster == -1:
                continue
                
            mask = self.df['cluster'] == cluster
            plt.scatter(
                self.df[mask]['longitude'],
                self.df[mask]['latitude'],
                alpha=0.5,
                label=f'Cluster {cluster}'
            )
        
        plt.title('Earthquake Clusters')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.show()
        
    def plot_omori_law(self, main_shock_time):
        """
        Plot Omori's Law fit for a specific main shock
        """
        try:
            # Validate that omori_params exists and has the required data
            if not hasattr(self, 'omori_params') or self.omori_params is None or self.omori_params.empty:
                raise ValueError("No Omori parameters available")
            
            # Check if the main shock time exists in omori_params
            if main_shock_time not in self.omori_params['main_shock_time'].values:
                # If the exact time is not found, use the first available main shock time
                print(f"Main shock time {main_shock_time} not found, using first available main shock")
                main_shock_time = self.omori_params['main_shock_time'].iloc[0]
                print(f"Using main shock time: {main_shock_time}")
            
            params = self.omori_params[
                self.omori_params['main_shock_time'] == main_shock_time
            ].iloc[0]
            
            # Validate required parameters
            required_params = ['K', 'c', 'p']
            for param in required_params:
                if param not in params or pd.isna(params[param]):
                    raise ValueError(f"Missing or invalid parameter: {param}")
            
            # Get aftershocks
            mask = (
                (self.df['time'] > main_shock_time) &
                (self.df['is_aftershock'])
            )
            aftershocks = self.df[mask]
            
            # Check if we have enough aftershocks
            if len(aftershocks) < 2:
                raise ValueError("Not enough aftershocks for plotting")
            
            # Calculate time differences
            t = (aftershocks['time'] - main_shock_time).dt.total_seconds() / 3600
            n = np.arange(1, len(t) + 1)
            
            # Validate that t and n have the same length
            if len(t) != len(n):
                raise ValueError(f"Length mismatch: t has {len(t)} elements, n has {len(n)} elements")
            
            # Generate Omori's Law curve
            t_fit = np.linspace(0, t.max(), 1000)
            n_fit = self.omori_law(t_fit, params['K'], params['c'], params['p'])
            
            # Plot
            plt.figure(figsize=(10, 6))
            plt.scatter(t, n, label='Observed')
            plt.plot(t_fit, n_fit, 'r-', label='Omori\'s Law')
            plt.title('Aftershock Decay (Omori\'s Law)')
            plt.xlabel('Time (hours)')
            plt.ylabel('Cumulative Number of Aftershocks')
            plt.legend()
            plt.show()
            
        except Exception as e:
            # Return error message instead of raising exception
            print(f"Error in plot_omori_law: {e}")
            raise e
        
    def generate_chain_statistics(self):
        """
        Generate enhanced statistics about earthquake chains
        """
        stats = {
            'total_earthquakes': len(self.df),
            'main_shocks': len(self.df[self.df['mag'] >= 5.0]),
            'aftershocks': self.df['is_aftershock'].sum(),
            'clusters': len(self.df['cluster'].unique()) - 1,  # Exclude noise
            'avg_time_between_quakes': self.df['time_diff'].mean().total_seconds() / 3600,
            'max_magnitude': self.df['mag'].max(),
            'min_magnitude': self.df['mag'].min(),
            'avg_spatial_distance': self.df['spatial_diff'].mean(),
            'omori_p_mean': self.omori_params['p'].mean() if hasattr(self, 'omori_params') else None
        }
        
        return stats 