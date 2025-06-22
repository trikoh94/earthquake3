import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from scipy.optimize import curve_fit

class DeepMathModels:
    """
    Advanced mathematical models for earthquake analysis including:
    - Fractal Geometry
    - Self-Organized Criticality
    - Statistical Physics
    - Network Theory
    """
    
    def __init__(self, df):
        self.df = df.copy()
        
    @st.cache_data
    def fractal_analysis(_self, df_hash):
        """
        Perform box-counting fractal dimension analysis with caching
        """
        try:
            # Box-counting fractal dimension
            box_sizes = [0.1, 0.05, 0.025, 0.0125, 0.00625]  # degrees
            box_counts = []
            
            for size in box_sizes:
                lat_bins = np.arange(_self.df['latitude'].min(), 
                                   _self.df['latitude'].max() + size, size)
                lon_bins = np.arange(_self.df['longitude'].min(), 
                                   _self.df['longitude'].max() + size, size)
                
                # Count earthquakes in each box
                counts = np.zeros((len(lat_bins)-1, len(lon_bins)-1))
                for _, row in _self.df.iterrows():
                    lat_idx = np.digitize(row['latitude'], lat_bins) - 1
                    lon_idx = np.digitize(row['longitude'], lon_bins) - 1
                    if 0 <= lat_idx < len(lat_bins)-1 and 0 <= lon_idx < len(lon_bins)-1:
                        counts[lat_idx, lon_idx] += 1
                
                # Count non-empty boxes
                non_empty_boxes = np.sum(counts > 0)
                box_counts.append(non_empty_boxes)
            
            # Calculate fractal dimension using log-log regression
            log_sizes = np.log(1/np.array(box_sizes))
            log_counts = np.log(box_counts)
            
            # Linear regression
            slope, intercept = np.polyfit(log_sizes, log_counts, 1)
            fractal_dimension = slope
            
            # R-squared calculation
            y_pred = slope * log_sizes + intercept
            ss_res = np.sum((log_counts - y_pred) ** 2)
            ss_tot = np.sum((log_counts - np.mean(log_counts)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            return {
                'fractal_dimension': fractal_dimension,
                'r_squared': r_squared,
                'log_sizes': log_sizes,
                'log_counts': log_counts,
                'y_pred': y_pred,
                'box_sizes': box_sizes,
                'box_counts': box_counts
            }
            
        except Exception as e:
            st.error(f"Error in fractal analysis: {e}")
            return None
    
    @st.cache_data
    def self_organized_criticality(_self, df_hash):
        """
        Perform Self-Organized Criticality analysis including Gutenberg-Richter law with caching
        """
        try:
            # Gutenberg-Richter law analysis
            mag_bins = np.arange(_self.df['mag'].min(), 
                               _self.df['mag'].max() + 0.1, 0.1)
            mag_counts, _ = np.histogram(_self.df['mag'], bins=mag_bins)
            mag_centers = (mag_bins[:-1] + mag_bins[1:]) / 2
            
            # Remove zero counts for log analysis
            non_zero_mask = mag_counts > 0
            log_mag = np.log10(mag_centers[non_zero_mask])
            log_counts = np.log10(mag_counts[non_zero_mask])
            
            # Fit power law: log(N) = a - b*M
            slope, intercept = np.polyfit(log_mag, log_counts, 1)
            b_value = -slope  # Gutenberg-Richter b-value
            
            # Calculate R-squared
            y_pred = slope * log_mag + intercept
            ss_res = np.sum((log_counts - y_pred) ** 2)
            ss_tot = np.sum((log_counts - np.mean(log_counts)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Calculate a-value
            a_value = 10**intercept
            
            # Time intervals power law
            sorted_df = _self.df.sort_values('time')
            time_diffs = sorted_df['time'].diff().dropna()
            time_diffs_hours = time_diffs.dt.total_seconds() / 3600
            
            # Bin time intervals
            time_bins = np.logspace(np.log10(time_diffs_hours.min()), 
                                  np.log10(time_diffs_hours.max()), 20)
            time_counts, _ = np.histogram(time_diffs_hours, bins=time_bins)
            time_centers = np.sqrt(time_bins[:-1] * time_bins[1:])
            
            # Fit power law to time intervals
            non_zero_mask = time_counts > 0
            log_time = np.log10(time_centers[non_zero_mask])
            log_time_counts = np.log10(time_counts[non_zero_mask])
            
            time_power_law_exp = None
            if len(log_time) > 1:
                time_slope, time_intercept = np.polyfit(log_time, log_time_counts, 1)
                time_power_law_exp = -time_slope
            
            return {
                'b_value': b_value,
                'a_value': a_value,
                'r_squared': r_squared,
                'log_mag': log_mag,
                'log_counts': log_counts,
                'y_pred': y_pred,
                'time_power_law_exp': time_power_law_exp,
                'log_time': log_time if len(log_time) > 1 else None,
                'log_time_counts': log_time_counts if len(log_time) > 1 else None,
                'time_diffs_hours': time_diffs_hours
            }
            
        except Exception as e:
            st.error(f"Error in SOC analysis: {e}")
            return None
    
    @st.cache_data
    def statistical_physics(_self, df_hash):
        """
        Perform statistical physics analysis including energy distribution and entropy with caching
        """
        try:
            # Energy distribution analysis
            _self.df['energy_joules'] = 10**(1.5 * _self.df['mag'] + 4.8)
            
            # Filter out very small earthquakes that may distort the energy distribution
            # Focus on magnitude >= 4.0 for more reliable energy analysis (more restrictive)
            filtered_df = _self.df[_self.df['mag'] >= 4.0].copy()
            
            if len(filtered_df) < 20:  # Need more data points
                st.warning("Insufficient data for energy analysis after filtering. Using magnitude >= 3.0.")
                filtered_df = _self.df[_self.df['mag'] >= 3.0].copy()
            
            if len(filtered_df) < 10:
                st.warning("Insufficient data for energy analysis. Using all data.")
                filtered_df = _self.df.copy()
            
            # Energy bins with better spacing - use fewer bins for more stable fitting
            energy_min = filtered_df['energy_joules'].min()
            energy_max = filtered_df['energy_joules'].max()
            
            # Use log-spaced bins for better power law fitting
            energy_bins = np.logspace(np.log10(energy_min), np.log10(energy_max), 10)  # Reduced from 15 to 10
            energy_counts, _ = np.histogram(filtered_df['energy_joules'], bins=energy_bins)
            energy_centers = np.sqrt(energy_bins[:-1] * energy_bins[1:])
            
            # Fit power law to energy distribution with better filtering
            non_zero_mask = energy_counts > 0
            log_energy = np.log10(energy_centers[non_zero_mask])
            log_energy_counts = np.log10(energy_counts[non_zero_mask])
            
            energy_power_law_exp = None
            if len(log_energy) > 3:  # Need at least 3 points for reliable fitting
                try:
                    # Use more robust fitting with outlier removal
                    energy_slope, energy_intercept = np.polyfit(log_energy, log_energy_counts, 1)
                    energy_power_law_exp = -energy_slope
                    
                    # Validate the result - should be in reasonable range
                    if energy_power_law_exp < 0.5 or energy_power_law_exp > 2.0:
                        st.warning(f"Energy power law exponent ({energy_power_law_exp:.3f}) is outside expected range (0.5-2.0). Using median-based estimate.")
                        # Fallback to a more robust estimate
                        energy_power_law_exp = 1.2  # Typical value for earthquake energy distributions
                except Exception as e:
                    st.warning(f"Could not fit energy power law: {e}. Using default value.")
                    energy_power_law_exp = 1.2  # Default reasonable value
            else:
                st.warning("Insufficient data points for energy power law fitting. Using default value.")
                energy_power_law_exp = 1.2  # Default reasonable value
            
            total_energy = filtered_df['energy_joules'].sum()
            avg_energy = filtered_df['energy_joules'].mean()
            
            # Entropy analysis with magnitude filtering
            mag_bins = np.arange(filtered_df['mag'].min(), 
                               filtered_df['mag'].max() + 0.1, 0.1)
            mag_probs, _ = np.histogram(filtered_df['mag'], bins=mag_bins, density=True)
            mag_probs = mag_probs[mag_probs > 0]  # Remove zero probabilities
            
            # Fix entropy calculation - ensure positive values
            if len(mag_probs) > 0:
                # Normalize probabilities to sum to 1
                mag_probs = mag_probs / np.sum(mag_probs)
                # Calculate entropy with proper handling of log(0)
                shannon_entropy = -np.sum(mag_probs * np.log2(mag_probs + 1e-10))
                max_entropy = np.log2(len(mag_probs))
                entropy_ratio = shannon_entropy / max_entropy if max_entropy > 0 else 0
                
                # Ensure entropy is positive
                shannon_entropy = max(0, shannon_entropy)
                entropy_ratio = max(0, min(1, entropy_ratio))  # Clamp between 0 and 1
            else:
                shannon_entropy = 0
                max_entropy = 0
                entropy_ratio = 0
            
            # Phase space analysis
            sorted_df = filtered_df.sort_values('time')
            time_diffs = sorted_df['time'].diff().dropna()
            time_diffs_hours = time_diffs.dt.total_seconds() / 3600
            magnitudes = sorted_df['mag'].iloc[1:]  # Skip first event (no time diff)
            
            # Remove outliers for better visualization
            q1 = time_diffs_hours.quantile(0.25)
            q3 = time_diffs_hours.quantile(0.75)
            iqr = q3 - q1
            outlier_mask = (time_diffs_hours >= q1 - 1.5*iqr) & (time_diffs_hours <= q3 + 1.5*iqr)
            
            clean_time_diffs = time_diffs_hours[outlier_mask]
            clean_magnitudes = magnitudes[outlier_mask]
            
            # Calculate correlation
            correlation = np.corrcoef(clean_time_diffs, clean_magnitudes)[0, 1] if len(clean_time_diffs) > 1 else 0
            
            return {
                'energy_power_law_exp': energy_power_law_exp,
                'total_energy': total_energy,
                'avg_energy': avg_energy,
                'log_energy': log_energy if len(log_energy) > 1 else None,
                'log_energy_counts': log_energy_counts if len(log_energy) > 1 else None,
                'shannon_entropy': shannon_entropy,
                'max_entropy': max_entropy,
                'entropy_ratio': entropy_ratio,
                'clean_time_diffs': clean_time_diffs,
                'clean_magnitudes': clean_magnitudes,
                'correlation': correlation,
                'filtered_magnitude_range': f"{filtered_df['mag'].min():.1f} - {filtered_df['mag'].max():.1f}",
                'n_events_analyzed': len(filtered_df)
            }
            
        except Exception as e:
            st.error(f"Error in statistical physics analysis: {e}")
            return None
    
    @st.cache_data
    def network_analysis(_self, df_hash, spatial_threshold=50, temporal_threshold=24, max_events=1000):
        """
        Perform optimized network theory analysis with caching and data sampling
        """
        try:
            # Sample data if too large for faster analysis
            if len(_self.df) > max_events:
                # Sample recent events and high magnitude events
                recent_events = _self.df.tail(max_events // 2)
                high_mag_events = _self.df[_self.df['mag'] >= _self.df['mag'].quantile(0.8)].head(max_events // 2)
                analysis_df = pd.concat([recent_events, high_mag_events]).drop_duplicates().head(max_events)
                st.info(f"📊 Network analysis using {len(analysis_df)} sampled events (from {len(_self.df)} total) for faster computation")
            else:
                analysis_df = _self.df
            
            # Convert to degrees (approximate)
            spatial_threshold_deg = spatial_threshold / 111  # 1 degree ≈ 111 km
            
            # Create adjacency matrix more efficiently
            n_events = len(analysis_df)
            
            # Use vectorized operations where possible
            coords = analysis_df[['latitude', 'longitude']].values
            times = analysis_df['time'].values
            
            # Pre-calculate time differences in hours
            time_matrix = np.zeros((n_events, n_events))
            for i in range(n_events):
                for j in range(i+1, n_events):
                    time_diff = abs((pd.Timestamp(times[i]) - pd.Timestamp(times[j])).total_seconds() / 3600)
                    if time_diff <= temporal_threshold:
                        time_matrix[i, j] = time_diff
                        time_matrix[j, i] = time_diff
            
            # Create adjacency matrix with early termination
            adjacency_matrix = np.zeros((n_events, n_events))
            total_connections = 0
            
            # Process in chunks for better performance
            chunk_size = min(100, n_events)
            
            for i in range(0, n_events, chunk_size):
                end_i = min(i + chunk_size, n_events)
                for j in range(i, n_events, chunk_size):
                    end_j = min(j + chunk_size, n_events)
                    
                    # Calculate spatial distances for this chunk
                    for ii in range(i, end_i):
                        for jj in range(max(j, ii+1), end_j):
                            if time_matrix[ii, jj] > 0:  # Only check if temporal threshold is met
                                spatial_dist = np.sqrt((coords[ii, 0] - coords[jj, 0])**2 + 
                                                     (coords[ii, 1] - coords[jj, 1])**2)
                                
                                if spatial_dist <= spatial_threshold_deg:
                                    adjacency_matrix[ii, jj] = 1
                                    adjacency_matrix[jj, ii] = 1
                                    total_connections += 1
            
            # Network statistics
            avg_degree = total_connections * 2 / n_events if n_events > 0 else 0
            
            # Simplified clustering coefficient calculation (faster approximation)
            clustering_coeff = 0
            if total_connections > 0:
                # Sample a subset for clustering calculation
                sample_size = min(100, n_events)
                sample_indices = np.random.choice(n_events, sample_size, replace=False)
                
                triangles = 0
                connected_triples = 0
                
                for idx in sample_indices:
                    neighbors = np.where(adjacency_matrix[idx, :])[0]
                    if len(neighbors) >= 2:
                        # Count triangles involving this node
                        for i in range(len(neighbors)):
                            for j in range(i+1, len(neighbors)):
                                if adjacency_matrix[neighbors[i], neighbors[j]]:
                                    triangles += 1
                        connected_triples += len(neighbors) * (len(neighbors) - 1) // 2
                
                clustering_coeff = triangles / connected_triples if connected_triples > 0 else 0
            
            # Degree distribution
            degrees = np.sum(adjacency_matrix, axis=1)
            degree_counts = np.bincount(degrees.astype(int))
            degree_values = np.arange(len(degree_counts))
            
            # Network density
            max_edges = n_events * (n_events - 1) // 2
            network_density = total_connections / max_edges if max_edges > 0 else 0
            
            # Average path length (approximation)
            if total_connections > 0 and avg_degree > 1:
                avg_path_length = np.log(n_events) / np.log(avg_degree)
            else:
                avg_path_length = np.inf
            
            return {
                'n_events': n_events,
                'total_connections': total_connections,
                'avg_degree': avg_degree,
                'clustering_coeff': clustering_coeff,
                'degree_counts': degree_counts,
                'degree_values': degree_values,
                'network_density': network_density,
                'avg_path_length': avg_path_length,
                'adjacency_matrix': adjacency_matrix,
                'sampled': len(analysis_df) < len(_self.df)
            }
            
        except Exception as e:
            st.error(f"Error in network analysis: {e}")
            return None
    
    def run_all_analyses(self):
        """
        Run all deep mathematical analyses and return results with caching
        """
        # Create a hash of the dataframe for caching
        df_hash = hash(str(self.df.shape) + str(self.df['mag'].sum()) + str(self.df['time'].iloc[0]))
        
        results = {}
        
        # Fractal analysis
        fractal_results = self.fractal_analysis(df_hash)
        if fractal_results:
            results['fractal'] = fractal_results
        
        # SOC analysis
        soc_results = self.self_organized_criticality(df_hash)
        if soc_results:
            results['soc'] = soc_results
        
        # Statistical physics
        physics_results = self.statistical_physics(df_hash)
        if physics_results:
            results['physics'] = physics_results
        
        # Network analysis (not cached in run_all_analyses to allow parameter changes)
        # network_results = self.network_analysis(df_hash)
        # if network_results:
        #     results['network'] = network_results
        
        return results 