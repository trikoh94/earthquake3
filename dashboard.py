import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
from statsmodels.tsa.arima.model import ARIMA
from earthquake_analysis import EarthquakeAnalyzer
from time_series_analysis import TimeSeriesAnalyzer
from earthquake_chain_analysis import EarthquakeChainAnalyzer
from energy_balance_model import EnergyBalanceAnalyzer

# Place these at the top-level (outside the class)
@st.cache_data(show_spinner=False)
def cached_identify_aftershocks(_df, mainshock_mag, aftershock_window, distance_window):
    """
    캐시된 여진 식별 함수 - 디버그 메시지 개선
    """
    # 캐시 히트/미스 확인을 위한 고유 실행 ID
    import time
    execution_id = int(time.time() * 1000) % 10000
    
    print(f"DEBUG [{execution_id}]: cached_identify_aftershocks START - mag:{mainshock_mag}, time:{aftershock_window}, dist:{distance_window}")
    
    analyzer = EarthquakeChainAnalyzer(_df.copy())
    main_shocks = _df[_df['mag'] >= mainshock_mag].copy()
    main_shocks = main_shocks.sort_values('time', ascending=False).head(20)
    main_shocks_count = len(main_shocks)
    print(f"DEBUG [{execution_id}]: main_shocks count: {main_shocks_count}")
    
    if main_shocks_count == 0:
        print(f"DEBUG [{execution_id}]: No mainshocks found with mag >= {mainshock_mag}")
        return analyzer.df, None
    
    try:
        analyzer.identify_aftershocks(mainshock_mag, aftershock_window, distance_window)
        print(f"DEBUG [{execution_id}]: identify_aftershocks completed successfully")
    except Exception as e:
        print(f"DEBUG [{execution_id}]: identify_aftershocks ERROR: {e}")
        import traceback
        print(f"DEBUG [{execution_id}]: Traceback: {traceback.format_exc()}")
        return analyzer.df, None
    
    print(f"DEBUG [{execution_id}]: returning results")
    return analyzer.df, getattr(analyzer, 'omori_params', None)

@st.cache_data(show_spinner=False)
def cached_analyze_clusters(_df, eps_space, min_samples, eps_time):
    """
    캐시된 클러스터 분석 함수 - 디버그 메시지 개선
    """
    import time
    execution_id = int(time.time() * 1000) % 10000
    print(f"DEBUG [{execution_id}]: cached_analyze_clusters START")
    
    analyzer = EarthquakeChainAnalyzer(_df.copy())
    analyzer.analyze_clusters(eps_space, min_samples, eps_time)
    
    print(f"DEBUG [{execution_id}]: cached_analyze_clusters completed")
    return analyzer.df

class EarthquakeDashboard:
    def __init__(self):
        """
        Initialize the EarthquakeDashboard
        """
        self.analyzer = EarthquakeAnalyzer('japanearthquake_cleaned.csv')
        self.ts_analyzer = TimeSeriesAnalyzer(self.analyzer.df)
        self.chain_analyzer = EarthquakeChainAnalyzer(self.analyzer.df)
        self.energy_analyzer = EnergyBalanceAnalyzer(self.analyzer.df)
        
    def run(self):
        """
        Run the Streamlit dashboard
        """
        st.title('Japan Earthquake Analysis Dashboard')
        
        # Sidebar
        st.sidebar.title('Navigation')
        page = st.sidebar.selectbox(
            'Choose a page',
            ['Overview', 'Time Series Analysis', 'Chain Analysis', 'Energy Analysis', 'Interactive Map']
        )
        
        if page == 'Overview':
            self.show_overview()
        elif page == 'Time Series Analysis':
            self.show_time_series_analysis()
        elif page == 'Chain Analysis':
            self.show_chain_analysis()
        elif page == 'Energy Analysis':
            self.show_energy_analysis()
        else:
            self.show_interactive_map()
            
        # Streamlit 앱 시작 직후, 데이터프레임에서 Arrow로 직렬화 불가한 컬럼 제거
        for col in ['time_window']:
            if col in self.analyzer.df.columns:
                self.analyzer.df = self.analyzer.df.drop(columns=[col])
        
    def show_overview(self):
        """
        Show overview statistics and basic visualizations
        """
        st.header('Overview')
        
        # Basic statistics
        stats = self.analyzer.basic_statistics()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric('Total Earthquakes', stats['total_earthquakes'])
        with col2:
            st.metric('Mean Magnitude', f"{stats['mean_magnitude']:.2f}")
        with col3:
            st.metric('Mean Depth', f"{stats['mean_depth']:.2f} km")
            
        # Magnitude distribution
        st.subheader('Magnitude Distribution')
        fig = px.histogram(self.analyzer.df, x='mag', nbins=30)
        st.plotly_chart(fig)
        
        # Time series
        st.subheader('Earthquake Frequency Over Time')
        daily_counts = self.analyzer.df.groupby(self.analyzer.df['time'].dt.date).size()
        fig = px.line(x=daily_counts.index, y=daily_counts.values)
        st.plotly_chart(fig)
        
    def show_time_series_analysis(self):
        """
        Show time series analysis visualizations
        """
        st.header('Time Series Analysis')
        
        # Decomposition
        st.subheader('Time Series Decomposition')
        decomposition = self.ts_analyzer.daily_data['count']
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=decomposition.index, y=decomposition.values, name='Original'))
        st.plotly_chart(fig)
        
        # Forecasting
        st.subheader('Forecasting')
        forecast_days = st.slider('Forecast Days', 7, 90, 30)
        
        try:
            # ARIMA forecast
            model = ARIMA(self.ts_analyzer.daily_data['count'], order=(5,1,0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=forecast_days)
            
            # Get forecast standard errors
            forecast_std = model_fit.get_forecast(steps=forecast_days).conf_int()
            
            # Create forecast dates
            last_date = self.ts_analyzer.daily_data.index[-1]
            forecast_dates = pd.date_range(start=last_date, periods=forecast_days+1)[1:]
            
            # Create figure
            fig = go.Figure()
            
            # Add historical data
            fig.add_trace(go.Scatter(
                x=self.ts_analyzer.daily_data.index,
                y=self.ts_analyzer.daily_data['count'],
                name='Historical',
                line=dict(color='blue')
            ))
            
            # Add forecast
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast,
                name='Forecast',
                line=dict(color='red', dash='dash')
            ))
            
            # Add confidence intervals
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_std.iloc[:, 1],  # Upper bound
                fill=None,
                mode='lines',
                line=dict(color='rgba(255,0,0,0.1)'),
                name='Upper Bound'
            ))
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_std.iloc[:, 0],  # Lower bound
                fill='tonexty',
                mode='lines',
                line=dict(color='rgba(255,0,0,0.1)'),
                name='Lower Bound'
            ))
            
            # Update layout
            fig.update_layout(
                title='Earthquake Frequency Forecast',
                xaxis_title='Date',
                yaxis_title='Number of Earthquakes',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig)
            
            # Show forecast statistics
            st.subheader('Forecast Statistics')
            col1, col2 = st.columns(2)
            with col1:
                st.metric('Mean Forecast', f"{forecast.mean():.2f}")
            with col2:
                st.metric('Max Forecast', f"{forecast.max():.2f}")
                
        except Exception as e:
            st.error(f"Error in ARIMA forecasting: {str(e)}")
            st.info("Try adjusting the forecast parameters or using a different time range.")
        
    def show_chain_analysis(self):
        """
        Show chain (aftershock, Omori's Law, clustering, etc.) analysis with robust data checks, conversions, and caching for speed. Includes debug output for aftershock detection.
        """
        st.header('Chain Analysis')
        st.write('This section provides a comprehensive analysis of earthquake chains, including aftershock patterns, Omori\'s Law, spatiotemporal clustering, time intervals, and magnitude sequences.')

        # 캐시 클리어 버튼 (선택사항)
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🗑️ Clear Cache", help="Clear all cached data and rerun analysis"):
                st.cache_data.clear()
                st.experimental_rerun()

        # Legend/column explanations
        with st.expander('Legend & Column Explanations'):
            st.markdown('''
            - **time_diff_hours**: Time difference (in hours) between consecutive earthquakes. Useful for identifying temporal clustering.
            - **cluster**: Cluster label assigned by the ST-DBSCAN algorithm. -1 means noise (not in any cluster), 0, 1, 2, ... are cluster IDs.
            - **is_aftershock**: Boolean (True/False) indicating whether the event is classified as an aftershock based on the selected mainshock and time window.
            ''')

        # 1. Aftershock Analysis
        st.subheader('1. Aftershock Analysis')
        st.markdown('''
        **What is this?**
        This analysis identifies aftershocks following a mainshock and visualizes their temporal distribution. You can adjust the mainshock magnitude threshold and the time window to define aftershocks.
        ''')
        
        # Relaxed default parameters for easier aftershock detection
        mainshock_mag = st.slider('Mainshock Magnitude Threshold', 4.5, 8.0, 5.0, 0.1, help='Minimum magnitude to consider an event as a mainshock.', key='mainshock_mag_slider')
        aftershock_window = st.slider('Aftershock Time Window (days)', 1, 30, 14, 1, help='Number of days after the mainshock to consider aftershocks.', key='aftershock_window_slider')
        distance_window = st.slider('Aftershock Distance Window (km)', 50, 300, 150, 10, help='Maximum distance (km) for aftershocks.', key='distance_window_slider')

        # 세션 상태를 사용한 캐시 상태 표시
        if 'last_aftershock_params' not in st.session_state:
            st.session_state.last_aftershock_params = None

        # 현재 파라미터
        current_params = (mainshock_mag, aftershock_window, distance_window)
        
        # 파라미터 변경 감지 및 캐시 상태 표시
        if st.session_state.last_aftershock_params != current_params:
            st.info("🔄 Parameters changed. Running aftershock analysis...")
            st.session_state.last_aftershock_params = current_params
        else:
            st.success("✅ Using cached results")

        # 여진 분석 실행
        try:
            # 진행 표시기 추가
            with st.spinner('Analyzing aftershocks...'):
                df_with_aftershocks, omori_params = cached_identify_aftershocks(
                    self.analyzer.df, mainshock_mag, aftershock_window, distance_window
                )
            
            # 결과 업데이트
            self.chain_analyzer.df = df_with_aftershocks
            self.chain_analyzer.omori_params = omori_params
            
            # 결과 표시
            col1, col2 = st.columns(2)
            with col1:
                mainshock_count = len(self.chain_analyzer.df[self.chain_analyzer.df['mag'] >= mainshock_mag])
                st.metric('Number of Main Shocks', mainshock_count)
            with col2:
                if 'is_aftershock' in self.chain_analyzer.df.columns:
                    n_aftershocks = int(self.chain_analyzer.df['is_aftershock'].sum())
                    st.metric('Number of Aftershocks', n_aftershocks)
                    
                    # 디버그 정보는 expander로 숨김
                    with st.expander('Debug Information', expanded=False):
                        st.write(f"Aftershock indices (first 10): {self.chain_analyzer.df[self.chain_analyzer.df['is_aftershock']].index.tolist()[:10]}")
                        st.write('First few aftershock events:')
                        st.dataframe(self.chain_analyzer.df[self.chain_analyzer.df['is_aftershock']].head())
                else:
                    st.info('No aftershock data available.')
                    
        except Exception as e:
            st.error(f'Aftershock analysis could not be performed: {e}')
            with st.expander('Error Details', expanded=False):
                import traceback
                st.text(traceback.format_exc())

        # 아래에 반드시 다음 섹션이 실행되도록!
        st.write('---')  # 구분선
        st.write('Omori\'s Law Analysis 시작...')

        # Ensure time_diff_hours exists and is float
        if 'time_diff' in self.chain_analyzer.df.columns:
            if np.issubdtype(self.chain_analyzer.df['time_diff'].dtype, np.timedelta64):
                self.chain_analyzer.df['time_diff_hours'] = self.chain_analyzer.df['time_diff'].dt.total_seconds() / 3600
            else:
                self.chain_analyzer.df['time_diff_hours'] = self.chain_analyzer.df['time_diff'].astype(float)
        else:
            self.chain_analyzer.df['time_diff_hours'] = np.nan

        # 2. Omori's Law Analysis
        st.subheader("2. Omori's Law Analysis")
        st.markdown('''
        **What is this?**
        Omori's Law describes how the frequency of aftershocks decays with time after a mainshock. This section fits Omori's Law to the aftershock data and visualizes the decay curve.
        ''')
        try:
            if omori_params is not None and not omori_params.empty:
                latest_main_shock = omori_params['main_shock_time'].max()
                self.chain_analyzer.plot_omori_law(latest_main_shock)
                st.write('Omori\'s Law Parameters:')
                params_df = omori_params.copy()
                params_df['main_shock_time'] = params_df['main_shock_time'].dt.strftime('%Y-%m-%d %H:%M')
                params_df.columns = ['Main Shock Time', 'K (Decay Rate)', 'c (Time Constant)', 'p (Decay Exponent)']
                st.dataframe(params_df)
            else:
                st.info('Not enough data for Omori\'s Law analysis.')
        except Exception as e:
            st.warning(f'Omori\'s Law analysis could not be performed: {e}')

        # 3. Spatial Clustering Analysis
        st.subheader('3. Spatial Clustering Analysis')
        st.markdown('''
        **What is this?**
        This analysis uses ST-DBSCAN to find clusters of earthquakes in space and time, helping to reveal geographic and temporal patterns.
        ''')
        eps_space = st.slider('Spatial Epsilon (km)', 5, 100, 30, 1, help='Maximum distance (km) for events to be considered part of the same cluster.', key='eps_space_slider')
        eps_time = st.slider('Temporal Epsilon (days)', 1, 30, 7, 1, help='Maximum time difference (days) for events to be considered part of the same cluster.', key='eps_time_slider')
        min_samples = st.slider('Min Samples per Cluster', 2, 20, 5, 1, help='Minimum number of events to form a cluster.', key='min_samples_slider')
        
        # Use cached clustering
        with st.spinner('Analyzing clusters...'):
            clustered_df = cached_analyze_clusters(self.analyzer.df, eps_space, min_samples, eps_time)
        
        self.chain_analyzer.df['cluster'] = clustered_df['cluster']
        if 'cluster' in self.chain_analyzer.df.columns and not self.chain_analyzer.df['cluster'].isnull().all():
            fig = px.scatter(self.chain_analyzer.df,
                            x='longitude',
                            y='latitude',
                            color='cluster',
                            size='mag',
                            hover_data=['time', 'mag', 'depth', 'is_aftershock'] if 'is_aftershock' in self.chain_analyzer.df.columns else ['time', 'mag', 'depth'],
                            title='Earthquake Clustering Results')
            st.plotly_chart(fig)
        else:
            st.info('No cluster data available. Please check clustering step or data.')

        # 4. Time Interval Analysis
        st.subheader('4. Time Interval Analysis')
        st.markdown('''
        **What is this?**
        This section analyzes the distribution of time intervals between consecutive earthquakes, which can reveal temporal clustering or regularity.
        ''')
        try:
            if 'time_diff_hours' in self.chain_analyzer.df.columns and self.chain_analyzer.df['time_diff_hours'].notnull().any():
                fig = px.histogram(self.chain_analyzer.df, x='time_diff_hours', nbins=50, title='Distribution of Time Differences (hours)')
                st.plotly_chart(fig)
                mean_td = self.chain_analyzer.df['time_diff_hours'].mean()
                median_td = self.chain_analyzer.df['time_diff_hours'].median()
                max_td = self.chain_analyzer.df['time_diff_hours'].max()
                st.write(f"Mean: {float(mean_td):.2f}, Median: {float(median_td):.2f}, Max: {float(max_td):.2f}")
            else:
                st.info('No time difference data available.')
        except Exception as e:
            st.warning(f'Time interval analysis could not be performed: {e}')

        # 5. Magnitude Sequence Analysis
        st.subheader('5. Magnitude Sequence Analysis')
        st.markdown('''
        **What is this?**
        This analysis examines how earthquake magnitudes change in sequence, which can help identify foreshocks, aftershocks, or magnitude clustering.
        ''')
        try:
            if 'is_aftershock' in self.chain_analyzer.df.columns:
                fig = px.scatter(self.chain_analyzer.df, x='time', y='mag', color='is_aftershock', title='Magnitude Sequence Over Time')
            else:
                fig = px.scatter(self.chain_analyzer.df, x='time', y='mag', title='Magnitude Sequence Over Time')
            st.plotly_chart(fig)
        except Exception as e:
            st.warning(f'Magnitude sequence analysis could not be performed: {e}')

        # 6. Summary Statistics
        st.subheader('6. Summary Statistics')
        st.markdown('''
        **What is this?**
        A summary of key statistics from the above analyses for quick reference.
        ''')
        try:
            stats = self.chain_analyzer.generate_chain_statistics()
            st.json(stats)
        except Exception as e:
            st.warning(f'Statistics summary could not be generated: {e}')
        
    def show_energy_analysis(self):
        """
        Show energy analysis visualizations (fully implemented).
        """
        st.header('Energy Analysis')
        st.write('Analyze the energy release patterns of earthquakes using the Gutenberg-Richter relation.')

        # Calculate energy in Joules and convert to TNT equivalent
        self.analyzer.df['energy_joules'] = 10**(1.5 * self.analyzer.df['mag'] + 4.8)
        self.analyzer.df['energy_tnt'] = self.analyzer.df['energy_joules'] / 4.184e9  # Convert to tons of TNT

        # 1. Energy Distribution
        st.subheader('1. Energy Distribution (TNT equivalent)')
        fig = px.histogram(self.analyzer.df, x='energy_tnt', nbins=50, log_x=True, title='Distribution of Earthquake Energy (TNT equivalent)')
        fig.update_xaxes(title='Energy (tons of TNT)')
        fig.update_yaxes(title='Count')
        st.plotly_chart(fig)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('Total Energy Released', f"{self.analyzer.df['energy_tnt'].sum():.2e} tons TNT")
        with col2:
            st.metric('Average Energy per Quake', f"{self.analyzer.df['energy_tnt'].mean():.2e} tons TNT")
        with col3:
            st.metric('Max Energy Release', f"{self.analyzer.df['energy_tnt'].max():.2e} tons TNT")

        # 2. Energy vs Magnitude
        st.subheader('2. Energy vs Magnitude')
        fig = px.scatter(self.analyzer.df, x='mag', y='energy_tnt', log_y=True, title='Earthquake Magnitude vs Energy Release')
        fig.update_xaxes(title='Magnitude')
        fig.update_yaxes(title='Energy (tons of TNT)')
        st.plotly_chart(fig)

        # 3. Energy Release Over Time
        st.subheader('3. Energy Release Over Time')
        time_period = st.selectbox('Time Period', ['Daily', 'Weekly', 'Monthly'], key='energy_time_period')
        if time_period == 'Daily':
            energy_series = self.analyzer.df.groupby(self.analyzer.df['time'].dt.date)['energy_tnt'].sum()
            title = 'Daily Energy Release'
        elif time_period == 'Weekly':
            energy_series = self.analyzer.df.groupby(self.analyzer.df['time'].dt.isocalendar().week)['energy_tnt'].sum()
            title = 'Weekly Energy Release'
        else:
            energy_series = self.analyzer.df.groupby(self.analyzer.df['time'].dt.to_period('M'))['energy_tnt'].sum()
            title = 'Monthly Energy Release'
        fig = px.line(x=energy_series.index.astype(str), y=energy_series.values, title=title)
        fig.update_xaxes(title='Time')
        fig.update_yaxes(title='Energy (tons of TNT)')
        st.plotly_chart(fig)

        # 4. Regional Energy Analysis
        st.subheader('4. Regional Energy Analysis')
        st.write('Analyze energy release patterns across different regions')
        lat_bins = st.slider('Latitude Bins', 5, 20, 10, key='lat_bins_slider')
        lon_bins = st.slider('Longitude Bins', 5, 20, 10, key='lon_bins_slider')
        self.analyzer.df['lat_bin'] = pd.cut(self.analyzer.df['latitude'], bins=lat_bins)
        self.analyzer.df['lon_bin'] = pd.cut(self.analyzer.df['longitude'], bins=lon_bins)
        region_energy = self.analyzer.df.groupby(['lat_bin', 'lon_bin'])['energy_tnt'].sum().reset_index()
        # Convert Interval columns to string for plotting
        region_energy['lat_bin'] = region_energy['lat_bin'].astype(str)
        region_energy['lon_bin'] = region_energy['lon_bin'].astype(str)
        fig = px.density_heatmap(region_energy, x='lon_bin', y='lat_bin', z='energy_tnt', title='Regional Energy Release Distribution', labels={'lon_bin': 'Longitude', 'lat_bin': 'Latitude', 'energy_tnt': 'Energy (tons of TNT)'})
        st.plotly_chart(fig)

        # 5. Energy-Magnitude Relationship Statistics
        st.subheader('5. Energy-Magnitude Relationship Statistics')
        log_energy = np.log10(self.analyzer.df['energy_joules'])
        slope, intercept = np.polyfit(self.analyzer.df['mag'], log_energy, 1)
        col1, col2 = st.columns(2)
        with col1:
            st.metric('Gutenberg-Richter Slope', f"{slope:.2f}")
        with col2:
            st.metric('Gutenberg-Richter Intercept', f"{intercept:.2f}")
        fig = px.scatter(self.analyzer.df, x='mag', y='energy_joules', log_y=True, title='Theoretical vs Actual Energy-Magnitude Relationship')
        theoretical_energy = 10**(1.5 * self.analyzer.df['mag'] + 4.8)
        fig.add_scatter(x=self.analyzer.df['mag'], y=theoretical_energy, mode='lines', name='Theoretical', line=dict(color='red'))
        fig.update_xaxes(title='Magnitude')
        fig.update_yaxes(title='Energy (Joules)')
        st.plotly_chart(fig)

    def show_interactive_map(self):
        """
        Show interactive map visualizations (fully implemented).
        """
        st.header('Interactive Map')
        st.write('Explore earthquake locations on an interactive map. Use the filters to adjust the view.')
        min_mag = st.slider('Minimum Magnitude', float(self.analyzer.df['mag'].min()), float(self.analyzer.df['mag'].max()), float(self.analyzer.df['mag'].min()), 0.1)
        max_depth = st.slider('Maximum Depth (km)', float(self.analyzer.df['depth'].min()), float(self.analyzer.df['depth'].max()), float(self.analyzer.df['depth'].max()), 1.0)
        filtered_df = self.analyzer.df[(self.analyzer.df['mag'] >= min_mag) & (self.analyzer.df['depth'] <= max_depth)]
        if filtered_df.empty:
            st.warning('No earthquakes match the selected filters.')
            return
        m = folium.Map(location=[filtered_df['latitude'].mean(), filtered_df['longitude'].mean()], zoom_start=5)
        for idx, row in filtered_df.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=max(2, row['mag']),
                popup=f"Magnitude: {row['mag']}<br>Depth: {row['depth']} km<br>Time: {row['time']}",
                color='red' if row['mag'] >= 5.0 else 'orange',
                fill=True,
                fill_opacity=0.7
            ).add_to(m)
        folium_static(m)

if __name__ == "__main__":
    dashboard = EarthquakeDashboard()
    dashboard.run()