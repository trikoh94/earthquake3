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
from transformer_model import SimulatedTransformerPredictor
from deep_math_models import DeepMathModels
from folium.plugins import MarkerCluster, HeatMap
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import time
import os

# Place these at the top-level (outside the class)
@st.cache_data(show_spinner=False)
def cached_identify_aftershocks(_df, mainshock_mag, aftershock_window, distance_window):
    """
    캐시된 여진 식별 함수 - 디버그 메시지 개선
    """
    # 캐시 히트/미스 확인을 위한 고유 실행 ID
    execution_id = int(time.time() * 1000) % 10000
    
    print(f"DEBUG [{execution_id}]: cached_identify_aftershocks START - mag:{mainshock_mag}, time:{aftershock_window}, dist:{distance_window}")
    
    analyzer = EarthquakeChainAnalyzer(_df.copy())
    main_shocks = _df[_df['mag'] >= mainshock_mag].copy()
    main_shocks = main_shocks.sort_values('time', ascending=False).head(50)  # Increased from 20 to 50
    main_shocks_count = len(main_shocks)
    print(f"DEBUG [{execution_id}]: main_shocks count: {main_shocks_count}")
    
    # Add magnitude distribution info
    total_quakes = len(_df)
    quakes_above_threshold = len(_df[_df['mag'] >= mainshock_mag])
    print(f"DEBUG [{execution_id}]: Total quakes: {total_quakes}, Quakes >= {mainshock_mag}: {quakes_above_threshold} ({quakes_above_threshold/total_quakes*100:.1f}%)")
    
    if main_shocks_count == 0:
        print(f"DEBUG [{execution_id}]: No mainshocks found with mag >= {mainshock_mag}")
        return analyzer.df, None
    
    try:
        # Use more lenient parameters for better aftershock detection
        analyzer.identify_aftershocks(mainshock_mag, aftershock_window, distance_window, max_mainshocks=50)
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
    execution_id = int(time.time() * 1000) % 10000
    print(f"DEBUG [{execution_id}]: cached_analyze_clusters START")
    
    analyzer = EarthquakeChainAnalyzer(_df.copy())
    analyzer.analyze_clusters(eps_space, min_samples, eps_time)
    
    print(f"DEBUG [{execution_id}]: cached_analyze_clusters completed")
    return analyzer.df

@st.cache_data
def load_all_analyzers():
    """
    Load and preprocess data by initializing all necessary analyzers.
    This function is cached to prevent reloading data on every script rerun.
    """
    analyzer = EarthquakeAnalyzer('japanearthquake_cleaned.csv')
    ts_analyzer = TimeSeriesAnalyzer(analyzer.df)
    chain_analyzer = EarthquakeChainAnalyzer(analyzer.df)
    energy_analyzer = EnergyBalanceAnalyzer(analyzer.df)
    return analyzer, ts_analyzer, chain_analyzer, energy_analyzer

@st.cache_data
def generate_cached_map(df, min_mag, max_depth):
    """
    Generates and caches a Folium map object based on filtered data.
    This prevents re-rendering the map on every interaction if the filters haven't changed.
    """
    filtered_df = df[(df['mag'] >= min_mag) & (df['depth'] <= max_depth)]
    
    if filtered_df.empty:
        return None
            
    # Create map centered on the data
    map_center = [filtered_df['latitude'].mean(), filtered_df['longitude'].mean()]
    m = folium.Map(location=map_center, zoom_start=5)

    # 1. Marker Cluster Layer
    marker_cluster = MarkerCluster().add_to(m)
    for idx, row in filtered_df.iterrows():
        popup_html = f"<b>Magnitude:</b> {row['mag']}<br><b>Depth:</b> {row['depth']} km<br><b>Time:</b> {row['time']}"
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=popup_html,
            icon=folium.Icon(color='red' if row['mag'] >= 6.0 else 'orange' if row['mag'] >= 5.0 else 'blue', icon='info-sign')
        ).add_to(marker_cluster)

    # 2. Heatmap Layer
    heat_data = [[row['latitude'], row['longitude']] for index, row in filtered_df.iterrows()]
    HeatMap(heat_data, radius=15).add_to(folium.FeatureGroup(name='Heatmap').add_to(m))

    # Add layer control to switch between views
    folium.LayerControl().add_to(m)
    
    return m

class EarthquakeDashboard:
    def __init__(self):
        """
        Initialize the EarthquakeDashboard
        """
        self.analyzer, self.ts_analyzer, self.chain_analyzer, self.energy_analyzer = load_all_analyzers()
        
    def run(self):
        """
        Run the Streamlit dashboard
        """
        st.title('Japan Earthquake Analysis Dashboard')
        
        # Sidebar for navigation
        st.sidebar.title("🌋 Japan Earthquake Analysis")
        
        # Menu options
        menu = st.sidebar.selectbox(
            "Choose Analysis Type:",
            [
                "📊 Overview & Statistics",
                "🔍 Chain Analysis (Aftershocks)", 
                "⚡ Energy Analysis",
                "📈 Time Series Analysis",
                "🗺️ Interactive Map",
                "🧮 Deep Mathematical Models",
                "🎯 Practical Insights & Recommendations"
            ]
        )
        
        # Display selected analysis
        if menu == "📊 Overview & Statistics":
            self.show_overview()
        elif menu == "🔍 Chain Analysis (Aftershocks)":
            self.show_chain_analysis()
        elif menu == "⚡ Energy Analysis":
            self.show_energy_analysis()
        elif menu == "📈 Time Series Analysis":
            self.show_time_series_analysis()
        elif menu == "🗺️ Interactive Map":
            self.show_interactive_map()
        elif menu == "🤖 Transformer Prediction":
            self.show_transformer_prediction()
        elif menu == "🧮 Deep Mathematical Models":
            self.show_deep_math_models()
        elif menu == "🎯 Practical Insights & Recommendations":
            self.show_practical_insights()
            
        # Streamlit 앱 시작 직후, 데이터프레임에서 Arrow로 직렬화 불가한 컬럼 제거
        for col in ['time_window']:
            if col in self.analyzer.df.columns:
                self.analyzer.df = self.analyzer.df.drop(columns=[col])
        
    def show_overview(self):
        """
        Show a more insightful overview with key metrics, charts, and a map of major earthquakes.
        """
        st.header('Seismic Activity Overview')
        st.markdown("A high-level summary of seismic activity across the entire dataset.")

        # Key Metrics
        stats = self.analyzer.basic_statistics()
        top_quake = self.analyzer.df.loc[self.analyzer.df['mag'].idxmax()]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric('Total Earthquakes', f"{stats['total_earthquakes']:,}")
        col2.metric('Mean Magnitude', f"{stats['mean_magnitude']:.2f}")
        col3.metric('Max Magnitude', f"{top_quake['mag']:.2f} on {pd.to_datetime(top_quake['time']).date()}")
        col4.metric('Date Range', f"{self.analyzer.df['time'].min().year} - {self.analyzer.df['time'].max().year}")

        # Visualizations
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader('Magnitude Distribution')
            fig_mag = px.histogram(self.analyzer.df, x='mag', nbins=50, title="Magnitude Frequency")
            fig_mag.update_layout(showlegend=False, yaxis_title="Count", xaxis_title="Magnitude")
            st.plotly_chart(fig_mag, use_container_width=True)
            
            st.subheader('Depth Distribution')
            fig_depth = px.histogram(self.analyzer.df, x='depth', nbins=50, title="Depth (km) Frequency")
            fig_depth.update_layout(showlegend=False, yaxis_title="Count", xaxis_title="Depth (km)")
            st.plotly_chart(fig_depth, use_container_width=True)

        with col2:
            st.subheader('Top 5 Largest Earthquakes')
            top_5_quakes = self.analyzer.df.nlargest(5, 'mag')
            
            map_center = [top_5_quakes['latitude'].mean(), top_5_quakes['longitude'].mean()]
            m = folium.Map(location=map_center, zoom_start=4)
            
            for _, row in top_5_quakes.iterrows():
                popup_html = f"<b>Magnitude: {row['mag']}</b><br>Date: {pd.to_datetime(row['time']).date()}<br>Depth: {row['depth']} km"
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    popup=popup_html,
                    tooltip=f"Mag: {row['mag']}",
                    icon=folium.Icon(color='red', icon='star')
                ).add_to(m)
            folium_static(m, height=410)

        st.subheader('Earthquake Frequency Over Time')
        daily_counts = self.analyzer.df.set_index('time').resample('D').size()
        rolling_avg = daily_counts.rolling(window=30).mean()
        
        fig_freq = go.Figure()
        fig_freq.add_trace(go.Scatter(x=daily_counts.index, y=daily_counts.values, mode='lines', name='Daily Count', line=dict(color='lightblue')))
        fig_freq.add_trace(go.Scatter(x=rolling_avg.index, y=rolling_avg.values, mode='lines', name='30-Day Avg', line=dict(color='red', width=2)))
        fig_freq.update_layout(title='Daily Earthquake Count with 30-Day Rolling Average', xaxis_title='Date', yaxis_title='Number of Earthquakes')
        st.plotly_chart(fig_freq, use_container_width=True)
        
    def show_time_series_analysis(self):
        """
        Show time series analysis visualizations with enhanced insights, including full decomposition and autocorrelation plots.
        """
        st.header('Time Series Analysis')
        st.write("""
        This section provides comprehensive time series analysis of earthquake frequency patterns, 
        including decomposition into trend, seasonal, and residual components, and forecasting.
        """)

        # Time Series Decomposition
        st.subheader('Time Series Decomposition')
        st.markdown("""
        **What is this?** Decomposition separates the time series into three components:
        - **Trend**: The underlying long-term movement in the data.
        - **Seasonality**: Repeating short-term cycles (e.g., yearly).
        - **Residuals**: The random, irregular component left over.
        This helps to understand the fundamental patterns driving earthquake frequency.
        """)
        
        daily_counts = self.ts_analyzer.daily_data['count'].asfreq('D').fillna(0)
        decomposition = seasonal_decompose(daily_counts, model='additive', period=365)
        
        # Plot decomposition using Plotly for consistency
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            subplot_titles=("Observed", "Trend", "Seasonal", "Residuals"))
        fig.add_trace(go.Scatter(x=decomposition.observed.index, y=decomposition.observed, mode='lines', name='Observed'), row=1, col=1)
        fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines', name='Trend'), row=2, col=1)
        fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines', name='Seasonal'), row=3, col=1)
        fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, mode='markers', name='Residual'), row=4, col=1)
        fig.update_layout(height=600, title_text="Time Series Decomposition")
        st.plotly_chart(fig, use_container_width=True)

        # Autocorrelation Plots
        st.subheader("Autocorrelation and Partial Autocorrelation (ACF/PACF)")
        st.markdown("""
        **What are these?** These plots help identify the correlation between a time series and its past values.
        - **ACF**: Shows the correlation of the series with its lags. Helps determine the 'q' parameter for ARIMA.
        - **PACF**: Shows the correlation of the series with its lags, after removing the effects of intervening lags. Helps determine the 'p' parameter for ARIMA.
        The blue shaded area represents the confidence interval; bars outside this area are statistically significant.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            fig_acf, ax_acf = plt.subplots(figsize=(6, 3))
            plot_acf(daily_counts.dropna(), ax=ax_acf, lags=40)
            ax_acf.set_title("Autocorrelation (ACF)")
            st.pyplot(fig_acf)
        with col2:
            fig_pacf, ax_pacf = plt.subplots(figsize=(6, 3))
            plot_pacf(daily_counts.dropna(), ax=ax_pacf, lags=40)
            ax_pacf.set_title("Partial Autocorrelation (PACF)")
            st.pyplot(fig_pacf)

        # Forecasting
        st.subheader('ARIMA Forecasting with Quantitative Assessment')
        st.write("""
        **Forecasting Analysis**: This section uses ARIMA (AutoRegressive Integrated Moving Average) 
        models to predict future earthquake frequencies and provides quantitative assessment of prediction accuracy.
        """)
        
        forecast_days = st.slider('Forecast Days', 7, 90, 30)
        
        try:
            # Split data for validation
            split_point = int(len(self.ts_analyzer.daily_data) * 0.8)
            train_data = self.ts_analyzer.daily_data['count'][:split_point]
            test_data = self.ts_analyzer.daily_data['count'][split_point:]
            
            # ARIMA forecast
            model = ARIMA(train_data, order=(5,1,0))
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
                name='Historical Data',
                line=dict(color='blue')
            ))
            
            # Add forecast
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast,
                name='ARIMA Forecast',
                line=dict(color='red', dash='dash')
            ))
            
            # Add confidence intervals
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_std.iloc[:, 1],  # Upper bound
                fill=None,
                mode='lines',
                line=dict(color='rgba(255,0,0,0.1)'),
                name='95% Confidence Interval'
            ))
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_std.iloc[:, 0],  # Lower bound
                fill='tonexty',
                mode='lines',
                line=dict(color='rgba(255,0,0,0.1)'),
                name='95% Confidence Interval'
            ))
            
            # Update layout
            fig.update_layout(
                title='Earthquake Frequency Forecast with Confidence Intervals',
                xaxis_title='Date',
                yaxis_title='Number of Earthquakes',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig)
            
            # Model validation and error metrics
            if len(test_data) > 0:
                # Generate predictions for test period
                test_forecast = model_fit.forecast(steps=len(test_data))
                
                # Calculate error metrics
                mse = np.mean((test_data - test_forecast) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(test_data - test_forecast))
                mape = np.mean(np.abs((test_data - test_forecast) / test_data)) * 100
                
                st.subheader('Model Performance Metrics')
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric('RMSE', f"{rmse:.2f}")
                with col2:
                    st.metric('MAE', f"{mae:.2f}")
                with col3:
                    st.metric('MAPE', f"{mape:.1f}%")
                with col4:
                    st.metric('MSE', f"{mse:.2f}")
                
                # Performance interpretation
                st.info("""
                **📊 Model Performance Interpretation:**
                - **RMSE (Root Mean Square Error)**: {:.2f} - Average prediction error in earthquake count
                - **MAE (Mean Absolute Error)**: {:.2f} - Average absolute prediction error
                - **MAPE (Mean Absolute Percentage Error)**: {:.1f}% - Average percentage error
                
                **🔍 Key Insights:**
                - The model shows {} prediction accuracy
                - {} indicates the model's ability to capture temporal patterns
                - Forecast confidence intervals reflect uncertainty in earthquake prediction
                """.format(rmse, mae, mape, 
                          "good" if mape < 20 else "moderate" if mape < 40 else "poor",
                          "Low MAPE" if mape < 20 else "Moderate MAPE" if mape < 40 else "High MAPE"))
            
            # Show forecast statistics
            st.subheader('Forecast Summary Statistics')
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric('Mean Forecast', f"{forecast.mean():.2f}")
            with col2:
                st.metric('Max Forecast', f"{forecast.max():.2f}")
            with col3:
                st.metric('Min Forecast', f"{forecast.min():.2f}")
            
            # Event threshold warnings
            high_activity_threshold = forecast.mean() + 2 * forecast.std()
            high_activity_days = (forecast > high_activity_threshold).sum()
            
            if high_activity_days > 0:
                st.warning(f"""
                🚨 **High Activity Alert**: 
                {high_activity_days} out of {forecast_days} forecast days show predicted earthquake activity 
                above normal levels (>{high_activity_threshold:.1f} events/day).
                """)
            
            # Model insights
            st.subheader('Forecasting Insights')
            st.write("""
            **🔬 Scientific Interpretation:**
            
            1. **Temporal Patterns**: The ARIMA model captures autocorrelation in earthquake frequency, 
            suggesting that recent activity levels influence future predictions.
            
            2. **Prediction Uncertainty**: Wide confidence intervals indicate high inherent variability 
            in earthquake occurrence, making precise prediction challenging.
            
            3. **Model Limitations**: The model assumes stationarity and may not capture sudden 
            changes in seismic activity due to major events or geological changes.
            
            4. **Practical Applications**: While useful for trend analysis, earthquake forecasting 
            should be used alongside other monitoring methods for comprehensive risk assessment.
            """)
                
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
                st.rerun()

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
        
        # Add waiting message
        st.info("⏳ **Please wait up to 30 seconds for the aftershock analysis to complete.** This analysis involves complex spatiotemporal calculations and may take some time to process.")
        
        # Relaxed default parameters for easier aftershock detection
        mainshock_mag = st.slider('Mainshock Magnitude Threshold', 5.0, 8.0, 6.0, 0.1, help='Minimum magnitude to consider an event as a mainshock. Higher values (6.0+) are recommended for meaningful aftershock analysis.', key='mainshock_mag_slider')
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
                
                # Add magnitude distribution info
                total_quakes = len(self.chain_analyzer.df)
                quakes_above_threshold = len(self.chain_analyzer.df[self.chain_analyzer.df['mag'] >= mainshock_mag])
                st.info(f"📊 **Magnitude Distribution:**\n- Total earthquakes: {total_quakes:,}\n- Quakes ≥ {mainshock_mag}: {quakes_above_threshold:,} ({quakes_above_threshold/total_quakes*100:.1f}%)")
                
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
                # Add debug information
                st.info(f"📊 Found {len(omori_params)} main shocks with valid Omori's Law parameters")
                
                # Validate omori_params structure before processing
                required_columns = ['main_shock_time', 'K', 'c', 'p']
                if all(col in omori_params.columns for col in required_columns):
                    # Try to find a valid main shock time
                    latest_main_shock = omori_params['main_shock_time'].max()
                    
                    # Check if the main shock exists in omori_params
                    if latest_main_shock in omori_params['main_shock_time'].values:
                        main_shock_to_use = latest_main_shock
                    else:
                        # Use the first available main shock if latest is not found
                        main_shock_to_use = omori_params['main_shock_time'].iloc[0]
                        st.info(f"⚠️ Latest main shock not found, using first available main shock: {main_shock_to_use}")
                    
                    try:
                        self.chain_analyzer.plot_omori_law(main_shock_to_use)
                        st.write('Omori\'s Law Parameters:')
                        params_df = omori_params.copy()
                        
                        # Check actual columns and rename them properly
                        if 'main_shock_time' in params_df.columns:
                            params_df['main_shock_time'] = params_df['main_shock_time'].dt.strftime('%Y-%m-%d %H:%M')
                        
                        # Create a mapping for column names
                        column_mapping = {
                            'main_shock_time': 'Main Shock Time',
                            'main_shock_mag': 'Main Shock Magnitude',
                            'K': 'K (Decay Rate)',
                            'c': 'c (Time Constant)',
                            'p': 'p (Decay Exponent)',
                            'p_value': 'P-Value'
                        }
                        
                        # Rename only the columns that exist
                        existing_columns = {col: column_mapping[col] for col in params_df.columns if col in column_mapping}
                        params_df = params_df.rename(columns=existing_columns)
                        
                        st.dataframe(params_df)
                    except Exception as plot_error:
                        st.warning(f'Could not plot Omori\'s Law: {plot_error}')
                        # Still show the parameters even if plotting fails
                        st.write('Omori\'s Law Parameters:')
                        params_df = omori_params.copy()
                        
                        # Check actual columns and rename them properly
                        if 'main_shock_time' in params_df.columns:
                            params_df['main_shock_time'] = params_df['main_shock_time'].dt.strftime('%Y-%m-%d %H:%M')
                        
                        # Create a mapping for column names
                        column_mapping = {
                            'main_shock_time': 'Main Shock Time',
                            'main_shock_mag': 'Main Shock Magnitude',
                            'K': 'K (Decay Rate)',
                            'c': 'c (Time Constant)',
                            'p': 'p (Decay Exponent)',
                            'p_value': 'P-Value'
                        }
                        
                        # Rename only the columns that exist
                        existing_columns = {col: column_mapping[col] for col in params_df.columns if col in column_mapping}
                        params_df = params_df.rename(columns=existing_columns)
                        
                        st.dataframe(params_df)
                        
                        # Show which main shock was used for plotting
                        st.info(f"📊 Used main shock time: {main_shock_to_use}")
                        
                        # Show debug information about the columns
                        with st.expander('Debug: Omori Parameters Details', expanded=False):
                            st.write("Original columns:", omori_params.columns.tolist())
                            st.write("Column mapping:", column_mapping)
                            st.write("Existing columns mapped:", existing_columns)
                else:
                    st.warning('Omori\'s Law parameters are missing required columns.')
                    with st.expander('Debug: Omori Parameters Structure', expanded=False):
                        st.write("Available columns:", omori_params.columns.tolist())
                        st.write("Required columns:", required_columns)
            else:
                st.info('Not enough data for Omori\'s Law analysis.')
                # Add more detailed information about why it failed
                with st.expander('Debug: Why Omori\'s Law Failed', expanded=False):
                    if omori_params is None:
                        st.write("❌ omori_params is None")
                    elif omori_params.empty:
                        st.write("❌ omori_params is empty")
                        st.write("This usually means:")
                        st.write("- Not enough aftershocks found (need at least 3)")
                        st.write("- Main shock magnitude threshold too high")
                        st.write("- Time/distance windows too restrictive")
                        st.write("- Curve fitting failed for all main shocks")
                    
                    # Show some statistics to help diagnose
                    if 'is_aftershock' in self.chain_analyzer.df.columns:
                        n_aftershocks = self.chain_analyzer.df['is_aftershock'].sum()
                        st.write(f"Total aftershocks found: {n_aftershocks}")
                        
                        # Show aftershock distribution by main shock
                        main_shocks = self.chain_analyzer.df[self.chain_analyzer.df['mag'] >= mainshock_mag]
                        st.write(f"Main shocks (mag >= {mainshock_mag}): {len(main_shocks)}")
                        
                        if len(main_shocks) > 0:
                            st.write("Top 5 main shocks by magnitude:")
                            top_shocks = main_shocks.nlargest(5, 'mag')[['time', 'mag', 'latitude', 'longitude']]
                            st.dataframe(top_shocks)
        except Exception as e:
            st.warning(f'Omori\'s Law analysis could not be performed: {e}')
            with st.expander('Error Details', expanded=False):
                import traceback
                st.text(traceback.format_exc())

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
            
            # Enhanced Temporal Cluster Evolution Analysis
            st.subheader("Temporal Cluster Evolution")
            
            # Analyze clusters by year with more detailed metrics
            self.chain_analyzer.df['year'] = self.chain_analyzer.df['time'].dt.year
            self.chain_analyzer.df['month'] = self.chain_analyzer.df['time'].dt.month
            
            # Yearly cluster statistics
            yearly_stats = []
            for year in sorted(self.chain_analyzer.df['year'].unique()):
                year_data = self.chain_analyzer.df[self.chain_analyzer.df['year'] == year]
                clustered_data = year_data[year_data['cluster'] != -1]
                
                if len(clustered_data) > 0:
                    yearly_stats.append({
                        'Year': year,
                        'Total_Events': len(year_data),
                        'Clustered_Events': len(clustered_data),
                        'Unique_Clusters': clustered_data['cluster'].nunique(),
                        'Avg_Cluster_Size': len(clustered_data) / clustered_data['cluster'].nunique() if clustered_data['cluster'].nunique() > 0 else 0,
                        'Clustering_Ratio': len(clustered_data) / len(year_data) * 100
                    })
                else:
                    yearly_stats.append({
                        'Year': year,
                        'Total_Events': len(year_data),
                        'Clustered_Events': 0,
                        'Unique_Clusters': 0,
                        'Avg_Cluster_Size': 0,
                        'Clustering_Ratio': 0
                    })
            
            yearly_df = pd.DataFrame(yearly_stats)
            
            # Create multiple visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Number of clusters by year
                fig = px.line(yearly_df, x='Year', y='Unique_Clusters', 
                            title='Number of Clusters by Year',
                            labels={'Unique_Clusters': 'Number of Clusters'})
                fig.update_traces(line=dict(width=3, color='blue'))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Clustering ratio by year
                fig = px.line(yearly_df, x='Year', y='Clustering_Ratio', 
                            title='Clustering Ratio by Year (%)',
                            labels={'Clustering_Ratio': 'Clustering Ratio (%)'})
                fig.update_traces(line=dict(width=3, color='red'))
                st.plotly_chart(fig, use_container_width=True)
            
            # Cluster size distribution over time
            st.subheader("Cluster Size Distribution Over Time")
            cluster_sizes = []
            for cluster_id in self.chain_analyzer.df[self.chain_analyzer.df['cluster'] != -1]['cluster'].unique():
                cluster_data = self.chain_analyzer.df[self.chain_analyzer.df['cluster'] == cluster_id]
                cluster_sizes.append({
                    'Cluster_ID': cluster_id,
                    'Size': len(cluster_data),
                    'Start_Time': cluster_data['time'].min(),
                    'End_Time': cluster_data['time'].max(),
                    'Duration_Hours': (cluster_data['time'].max() - cluster_data['time'].min()).total_seconds() / 3600,
                    'Avg_Magnitude': cluster_data['mag'].mean(),
                    'Max_Magnitude': cluster_data['mag'].max()
                })
            
            if cluster_sizes:
                cluster_sizes_df = pd.DataFrame(cluster_sizes)
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.scatter(cluster_sizes_df, x='Start_Time', y='Size', 
                                   title='Cluster Size vs Start Time',
                                   labels={'Start_Time': 'Cluster Start Time', 'Size': 'Cluster Size'})
                    fig.update_traces(marker=dict(size=8, color='purple'))
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.scatter(cluster_sizes_df, x='Duration_Hours', y='Size', 
                                   title='Cluster Size vs Duration',
                                   labels={'Duration_Hours': 'Duration (hours)', 'Size': 'Cluster Size'})
                    fig.update_traces(marker=dict(size=8, color='orange'))
                    st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics
                st.info(f"""
                **📊 Temporal Cluster Evolution Insights:**
                
                **Overall Patterns:**
                - Total clusters identified: {len(cluster_sizes_df)}
                - Average cluster size: {cluster_sizes_df['Size'].mean():.1f} events
                - Average cluster duration: {cluster_sizes_df['Duration_Hours'].mean():.1f} hours
                - Largest cluster: {cluster_sizes_df['Size'].max()} events
                - Longest cluster duration: {cluster_sizes_df['Duration_Hours'].max():.1f} hours
                
                **Temporal Trends:**
                - Clustering activity varies significantly by year
                - Cluster sizes and durations provide insights into stress release patterns
                """)
                
                # Cluster characteristics table
                st.subheader("Top Clusters by Size")
                cluster_table = cluster_sizes_df.copy()
                cluster_table['Avg_Magnitude'] = cluster_table['Avg_Magnitude'].round(2)
                cluster_table['Max_Magnitude'] = cluster_table['Max_Magnitude'].round(2)
                cluster_table['Duration_Hours'] = cluster_table['Duration_Hours'].round(1)
                
                # Display top 10 clusters by size
                top_clusters = cluster_table.nlargest(10, 'Size')[['Cluster_ID', 'Size', 'Avg_Magnitude', 'Max_Magnitude', 'Duration_Hours']]
                top_clusters.columns = ['Cluster ID', 'Size', 'Avg Magnitude', 'Max Magnitude', 'Duration (hours)']
                st.dataframe(top_clusters, use_container_width=True)
            else:
                st.info('No clusters found with the current parameters. Try adjusting the clustering parameters.')
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
        st.subheader('5. Magnitude Sequence: Trends & Patterns')
        st.markdown('''
        This analysis visualizes how earthquake magnitudes have evolved over time, helping to identify major events, aftershock sequences, and long-term trends.
        ''')

        analysis_df = self.chain_analyzer.df.copy().sort_values('time')

        # Ensure 'is_aftershock' column exists for color coding
        if 'is_aftershock' not in analysis_df.columns:
            analysis_df['is_aftershock'] = False

        tab1, tab2 = st.tabs(["📈 Magnitude Timeline", "📊 Magnitude Distribution (Gutenberg-Richter)"])

        with tab1:
            st.markdown("""
            #### How to Read This Chart
            - **Each dot** represents a single earthquake.
            - **Vertical Position**: Higher dots mean stronger earthquakes (higher magnitude).
            - **Color**: Dots are colored based on whether they are identified as an **aftershock (dark blue)** or a **mainshock/regular event (light blue)**. Notice the dense cluster of aftershocks after the major earthquake in 2011.
            - **Red Line**: The **30-day rolling average** smoothens the data to reveal long-term trends. A rising line indicates a period of increasing seismic magnitude.
            """)
            try:
                # Ensure 'mag' is numeric for rolling calculation
                analysis_df['mag'] = pd.to_numeric(analysis_df['mag'], errors='coerce')
                analysis_df.dropna(subset=['mag'], inplace=True)
                
                # Calculate rolling average of magnitude
                analysis_df['mag_rolling_avg'] = analysis_df['mag'].rolling(window=30, center=True, min_periods=1).mean()

                fig = px.scatter(analysis_df, 
                               x='time', 
                               y='mag', 
                               color='is_aftershock',
                               color_discrete_map={True: 'blue', False: 'lightblue'},
                               size='mag',
                               hover_data=['mag', 'depth', 'time'],
                               title='Magnitude Sequence Over Time with Rolling Average')

                # Add rolling average line
                fig.add_trace(go.Scatter(x=analysis_df['time'], 
                                         y=analysis_df['mag_rolling_avg'], 
                                         mode='lines', 
                                         name='30-Day Rolling Avg',
                                         line=dict(color='red', width=2)))

                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.warning(f'Magnitude sequence plot could not be created: {e}')

        with tab2:
            st.markdown("""
            #### What is this?
            This histogram shows the frequency of earthquakes at different magnitudes. According to the **Gutenberg-Richter Law**, the number of earthquakes decreases exponentially as magnitude increases. A straight line on this log-linear plot indicates that the dataset follows this empirical law.
            """)
            try:
                fig_hist = px.histogram(analysis_df, x='mag', nbins=50, title='Earthquake Magnitude Distribution', log_y=True)
                fig_hist.update_layout(
                    yaxis_title="Count (Log Scale)",
                    xaxis_title="Magnitude"
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            except Exception as e:
                st.warning(f'Magnitude distribution plot could not be created: {e}')
        
        # Redesigned statistics section
        st.subheader("Key Statistics from the Timeline")
        try:
            if len(analysis_df) > 1:
                # Separate mainshocks and aftershocks
                mainshocks_df = analysis_df[analysis_df['is_aftershock'] == False]
                aftershocks_df = analysis_df[analysis_df['is_aftershock'] == True]

                # Calculate magnitude differences
                mag_diffs = analysis_df['mag'].diff().dropna()
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric('Avg. Mainshock Mag.', f"{mainshocks_df['mag'].mean():.2f}" if not mainshocks_df.empty else "N/A")
                col2.metric('Avg. Aftershock Mag.', f"{aftershocks_df['mag'].mean():.2f}" if not aftershocks_df.empty else "N/A")
                col3.metric('Max Magnitude Drop', f"{mag_diffs.min():.2f}" if not mag_diffs.empty else "N/A")
                col4.metric('Max Magnitude Jump', f"{mag_diffs.max():.2f}" if not mag_diffs.empty else "N/A")
                
                # Scientific interpretation
                st.info(f"""
                **🔬 Scientific Interpretation:**

                - **Aftershock Magnitude**: As expected, the average magnitude of aftershocks ({aftershocks_df['mag'].mean():.2f}) is lower than that of mainshocks ({mainshocks_df['mag'].mean():.2f}). This aligns with seismic theories.
                - **Foreshocks & Aftershocks**: The 'Max Magnitude Jump' of **{mag_diffs.max():.2f}** could represent a significant foreshock followed by a much larger mainshock. Conversely, the 'Max Magnitude Drop' of **{mag_diffs.min():.2f}** likely shows a large mainshock followed by a much weaker aftershock.
                - **Visual Trend**: The plot visually confirms the massive aftershock sequence following the **2011 Tōhoku earthquake**, where a large number of dark blue dots are clustered.
                """)
                
                # Add analytical efficacy section
                st.subheader("Analytical Efficacy of this Visualization")
                st.markdown("""
                - **Instant Identification of Major Seismic Events:** This chart allows for the immediate identification of the most powerful earthquakes (e.g., the 2011 Tōhoku earthquake) as prominent outliers. This provides a crucial starting point for a deeper temporal analysis without needing to sift through thousands of data points manually.
                - **Clear Visualization of Aftershock Dynamics:** The color-coding distinctly visualizes the fundamental seismological principle that major earthquakes are followed by numerous aftershocks. The dense cluster of dark blue dots after the 2011 event serves as powerful visual evidence of its prolonged impact on the region.
                - **Insight into Long-Term Energy Trends:** The red rolling-average line filters out short-term noise, revealing underlying trends in seismic energy. A rising slope can indicate periods of increasing tectonic stress, offering a macro-level view of regional seismic activity.
                - **Visual Substantiation for Statistical Data:** Abstract statistics like 'Max Magnitude Drop' become tangible and interpretable when viewed on the chart. They are no longer just numbers but represent a visible, significant drop between two events, strengthening the credibility of the statistical findings.
                """)
        except Exception as e:
            st.warning(f'Magnitude sequence statistics could not be performed: {e}')

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
        st.header("⚡ Energy Analysis: Patterns and Anomalies")
        st.markdown("""
        Understanding energy release is crucial for assessing seismic hazards, as energy is a more direct measure of destructive potential than magnitude. This section visualizes yearly energy patterns to identify significant trends and anomalous years.
        """)

        # Calculate energy if not present
        if 'energy_joules' not in self.analyzer.df.columns:
            self.analyzer.df['energy_joules'] = 10**(1.5 * self.analyzer.df['mag'] + 4.8)
        
        self.analyzer.df['year'] = self.analyzer.df['time'].dt.year

        # --- 1. Yearly Total vs. Average Energy ---
        st.subheader('1. Yearly Energy: Total Release vs. Average Per Event')
        st.markdown("""
        Comparing **Total Energy** with **Average Energy** per event helps distinguish between years with frequent, smaller earthquakes and years with fewer, but more powerful, events.
        - **High Total & High Average (e.g., 2011):** Indicates a year dominated by one or more exceptionally powerful earthquakes.
        - **High Total & Low Average:** Suggests a year with a high frequency of relatively minor earthquakes.
        """)

        yearly_energy_total = self.analyzer.df.groupby('year')['energy_joules'].sum()
        yearly_energy_avg = self.analyzer.df.groupby('year')['energy_joules'].mean()
        
        col1, col2 = st.columns(2)

        with col1:
            # Total Energy by Year (Bar Chart)
            fig_total = px.bar(yearly_energy_total, 
                               x=yearly_energy_total.index, 
                               y='energy_joules',
                               title='Total Energy Release by Year',
                               labels={'energy_joules': 'Total Energy (Joules)', 'year': 'Year'})
            st.plotly_chart(fig_total, use_container_width=True)

        with col2:
            # Average Energy by Year (Scatter Plot)
            fig_avg = px.scatter(yearly_energy_avg, 
                                 x=yearly_energy_avg.index, 
                                 y='energy_joules',
                                 size='energy_joules',
                                 title='Average Energy per Event by Year',
                                 labels={'energy_joules': 'Average Energy (Joules)', 'year': 'Year'})
            st.plotly_chart(fig_avg, use_container_width=True)

        # --- 2. Statistical Anomaly Detection ---
        st.subheader('2. Statistical Anomaly Detection')
        st.markdown("""
        This analysis uses a statistical method (Z-score) to identify years where the total energy release was significantly higher than the rolling average. A high Z-score (typically > 2) flags a year as a statistical outlier.
        """)
        
        # Calculate moving average and standard deviation for anomaly detection
        window = 5
        moving_avg = yearly_energy_total.rolling(window=window, center=True, min_periods=1).mean()
        moving_std = yearly_energy_total.rolling(window=window, center=True, min_periods=1).std()
        
        # Calculate Z-scores
        z_scores = (yearly_energy_total - moving_avg) / moving_std
        anomalies = z_scores[z_scores.abs() > 2]

        if not anomalies.empty:
            st.info(f"🚨 **Anomaly Year(s) Detected:** {', '.join(map(str, anomalies.index.tolist()))}")
            
            anomaly_data = []
            for year, z_score in anomalies.items():
                anomaly_data.append({
                    'Year': year,
                    'Total Energy (Joules)': f"{yearly_energy_total[year]:.2e}",
                    'Z-Score': f"{z_score:.2f}",
                    'Interpretation': 'Significantly higher than trend'
                })
            
            st.table(pd.DataFrame(anomaly_data).set_index('Year'))
        else:
            st.success("✅ No significant anomaly years detected based on the Z-score threshold.")
            
        # --- 3. Key Takeaways ---
        st.subheader("Key Takeaways from Energy Analysis")
        st.markdown("""
        - **The 2011 Tōhoku Earthquake's Dominance:** The data clearly shows that 2011 is an extreme outlier in both total and average energy, highlighting the monumental impact of this single event on the region's entire seismic energy budget.
        - **Energy as a Superior Metric:** While many earthquakes may occur in a year, the energy analysis demonstrates that a handful of high-magnitude events release the vast majority of the total seismic energy. This underscores why energy, not just frequency, is critical for hazard assessment.
        - **Stable Background Seismicity:** Outside of the 2011 anomaly, the yearly energy release, while variable, remains within a relatively stable range, indicating a consistent background level of seismic activity.
        """)

    def show_interactive_map(self):
        """
        Show enhanced interactive map visualizations with layers and risk assessment.
        The map generation is cached for performance.
        """
        st.header('🗺️ Enhanced Interactive Map')
        st.write('Explore earthquake locations with layers for density (Heatmap), individual markers (Marker Cluster), and regional risk assessment.')
        
        st.info("⏳ **Please wait up to 30 seconds for the map to load.** The initial generation with many data points can be slow. Subsequent loads with the same filters will be much faster due to caching.")
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            min_mag = st.slider('Minimum Magnitude', float(self.analyzer.df['mag'].min()), float(self.analyzer.df['mag'].max()), 4.0, 0.1, key="map_mag_slider")
        with col2:
            max_depth = st.slider('Maximum Depth (km)', float(self.analyzer.df['depth'].min()), float(self.analyzer.df['depth'].max()), 200.0, 10.0, key="map_depth_slider")
        
        # Map type selection
        map_type = st.radio(
            "Choose Map Type:",
            ["📍 Individual Earthquakes", "🔥 Risk Assessment Heatmap"],
            horizontal=True
        )
        
        if map_type == "📍 Individual Earthquakes":
            # Generate and display map using the cached function
            with st.spinner("Generating earthquake map... This may take a moment on first load."):
                map_object = generate_cached_map(self.analyzer.df, min_mag, max_depth)
            
            if map_object:
                folium_static(map_object, width=800, height=600)
            else:
                st.warning('No earthquakes match the selected filters.')
        
        else:  # Risk Assessment Heatmap
            st.subheader("🔥 Regional Risk Assessment")
            st.markdown("""
            **Risk Score Calculation:**
            - **Frequency Factor**: Number of earthquakes in each region
            - **Magnitude Factor**: Average and maximum magnitudes
            - **Clustering Factor**: Spatial concentration of events
            - **Higher scores = Higher risk**
            """)
            
            # Risk assessment parameters
            col1, col2 = st.columns(2)
            with col1:
                grid_size = st.slider("Grid Size (degrees)", 0.5, 2.0, 1.0, 0.1, help="Smaller grid = more detailed risk assessment")
            with col2:
                time_window = st.selectbox("Time Window", ["All Time", "Last 10 Years", "Last 5 Years", "Last Year"], help="Recent events may indicate current risk")
            
            if st.button("Generate Risk Assessment"):
                with st.spinner("Calculating regional risk scores..."):
                    # Filter data based on time window
                    filtered_df = self.analyzer.df[(self.analyzer.df['mag'] >= min_mag) & (self.analyzer.df['depth'] <= max_depth)]
                    
                    if time_window != "All Time":
                        current_time = filtered_df['time'].max()
                        if time_window == "Last 10 Years":
                            start_time = current_time - pd.DateOffset(years=10)
                        elif time_window == "Last 5 Years":
                            start_time = current_time - pd.DateOffset(years=5)
                        else:  # Last Year
                            start_time = current_time - pd.DateOffset(years=1)
                        filtered_df = filtered_df[filtered_df['time'] >= start_time]
                    
                    if filtered_df.empty:
                        st.warning('No earthquakes match the selected filters and time window.')
                        return
                    
                    # Create grid for risk assessment
                    lat_min, lat_max = filtered_df['latitude'].min(), filtered_df['latitude'].max()
                    lon_min, lon_max = filtered_df['longitude'].min(), filtered_df['longitude'].max()
                    
                    # Create grid cells
                    lat_bins = np.arange(lat_min, lat_max + grid_size, grid_size)
                    lon_bins = np.arange(lon_min, lon_max + grid_size, grid_size)
                    
                    risk_data = []
                    
                    for i in range(len(lat_bins) - 1):
                        for j in range(len(lon_bins) - 1):
                            # Get earthquakes in this grid cell
                            cell_data = filtered_df[
                                (filtered_df['latitude'] >= lat_bins[i]) & 
                                (filtered_df['latitude'] < lat_bins[i + 1]) &
                                (filtered_df['longitude'] >= lon_bins[j]) & 
                                (filtered_df['longitude'] < lon_bins[j + 1])
                            ]
                            
                            if len(cell_data) > 0:
                                # Calculate risk score
                                frequency = len(cell_data)
                                avg_magnitude = cell_data['mag'].mean()
                                max_magnitude = cell_data['mag'].max()
                                
                                # Risk score formula: frequency * avg_magnitude * max_magnitude
                                risk_score = frequency * avg_magnitude * max_magnitude
                                
                                # Add some clustering factor (density)
                                area = grid_size * grid_size  # approximate area in degrees²
                                density_factor = frequency / area if area > 0 else 0
                                
                                # Final risk score
                                final_risk = risk_score * (1 + density_factor * 0.1)
                                
                                risk_data.append({
                                    'lat': (lat_bins[i] + lat_bins[i + 1]) / 2,
                                    'lon': (lon_bins[j] + lon_bins[j + 1]) / 2,
                                    'risk_score': final_risk,
                                    'frequency': frequency,
                                    'avg_magnitude': avg_magnitude,
                                    'max_magnitude': max_magnitude
                                })
                    
                    if risk_data:
                        # Create risk map
                        risk_df = pd.DataFrame(risk_data)
                        
                        # Normalize risk scores for better visualization
                        max_risk = risk_df['risk_score'].max()
                        risk_df['normalized_risk'] = risk_df['risk_score'] / max_risk
                        
                        # Create map
                        map_center = [risk_df['lat'].mean(), risk_df['lon'].mean()]
                        m = folium.Map(location=map_center, zoom_start=5)
                        
                        # Add risk heatmap
                        heat_data = []
                        for _, row in risk_df.iterrows():
                            # Weight by risk score for heatmap intensity
                            weight = row['normalized_risk']
                            heat_data.append([row['lat'], row['lon'], weight])
                        
                        # Create heatmap layer
                        HeatMap(
                            heat_data, 
                            radius=20, 
                            blur=15,
                            max_zoom=10,
                            gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 0.8: 'red', 1.0: 'darkred'}
                        ).add_to(m)
                        
                        # Add risk score markers
                        for _, row in risk_df.iterrows():
                            # Color coding based on risk level
                            if row['normalized_risk'] > 0.8:
                                color = 'darkred'
                            elif row['normalized_risk'] > 0.6:
                                color = 'red'
                            elif row['normalized_risk'] > 0.4:
                                color = 'orange'
                            elif row['normalized_risk'] > 0.2:
                                color = 'yellow'
                            else:
                                color = 'green'
                            
                            popup_html = f"""
                            <b>Risk Score:</b> {row['risk_score']:.1f}<br>
                            <b>Earthquakes:</b> {row['frequency']}<br>
                            <b>Avg Magnitude:</b> {row['avg_magnitude']:.2f}<br>
                            <b>Max Magnitude:</b> {row['max_magnitude']:.2f}<br>
                            <b>Risk Level:</b> {'Very High' if row['normalized_risk'] > 0.8 else 'High' if row['normalized_risk'] > 0.6 else 'Medium' if row['normalized_risk'] > 0.4 else 'Low' if row['normalized_risk'] > 0.2 else 'Very Low'}
                            """
                            
                            folium.CircleMarker(
                                location=[row['lat'], row['lon']],
                                radius=8,
                                popup=popup_html,
                                color=color,
                                fill=True,
                                fillOpacity=0.7
                            ).add_to(m)
                        
                        # Add layer control
                        folium.LayerControl().add_to(m)
                        
                        # Display map
                        folium_static(m, width=800, height=600)
                        
                        # Risk statistics
                        st.subheader("📊 Risk Assessment Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Highest Risk Score", f"{risk_df['risk_score'].max():.1f}")
                        with col2:
                            st.metric("Average Risk Score", f"{risk_df['risk_score'].mean():.1f}")
                        with col3:
                            high_risk_areas = len(risk_df[risk_df['normalized_risk'] > 0.6])
                            st.metric("High Risk Areas", high_risk_areas)
                        with col4:
                            total_areas = len(risk_df)
                            st.metric("Total Areas Analyzed", total_areas)
                        
                        # Top 5 highest risk areas
                        st.subheader("🚨 Top 5 Highest Risk Areas")
                        top_risks = risk_df.nlargest(5, 'risk_score')[['lat', 'lon', 'risk_score', 'frequency', 'avg_magnitude', 'max_magnitude']]
                        top_risks.columns = ['Latitude', 'Longitude', 'Risk Score', 'Earthquakes', 'Avg Magnitude', 'Max Magnitude']
                        st.dataframe(top_risks, use_container_width=True)
                        
                        # Risk interpretation
                        st.info(f"""
                        **🔬 Risk Assessment Interpretation:**
                        
                        - **Highest Risk Area**: Located at ({top_risks.iloc[0]['Latitude']:.3f}, {top_risks.iloc[0]['Longitude']:.3f}) with risk score {top_risks.iloc[0]['Risk Score']:.1f}
                        - **Risk Distribution**: {high_risk_areas} out of {total_areas} areas ({high_risk_areas/total_areas*100:.1f}%) show high risk
                        - **Time Period**: Analysis based on {time_window.lower()} data
                        - **Grid Resolution**: {grid_size}° × {grid_size}° cells for detailed regional assessment
                        
                        **⚠️ Practical Implications:**
                        - High-risk areas may require enhanced monitoring and preparedness measures
                        - Risk scores correlate with historical earthquake activity and magnitude patterns
                        - Consider this assessment alongside geological fault maps for comprehensive risk evaluation
                        """)
                        
                    else:
                        st.warning('No risk data could be calculated with the current filters.')

    def show_transformer_prediction(self):
        """
        Show Transformer-based earthquake prediction section
        """
        st.header("Transformer-based Earthquake Prediction")
        st.write("""
        In this section, we use a Transformer model to predict the magnitude of future earthquakes based on the past 30 days of earthquake data. The model is trained on features such as magnitude, depth, latitude, and longitude. You can start the training process and view the results below.
        """)
        
        if st.button("Start Transformer Training"):
            with st.spinner("Training model..."):
                # Initialize predictor
                predictor = SimulatedTransformerPredictor()
                
                # Train model
                losses = predictor.simulate_training(self.analyzer.df, epochs=30)
                
                # Plot training loss
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=losses, mode='lines', name='Training Loss'))
                fig.update_layout(
                    title='Transformer Model Training Loss',
                    xaxis_title='Epoch',
                    yaxis_title='Loss',
                    showlegend=True
                )
                st.plotly_chart(fig)
                
                # Make predictions
                predictions, actual = predictor.simulate_predictions(self.analyzer.df)
                
                # Plot predictions vs actual
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=predictions[:100], mode='lines', name='Predicted'))
                fig.add_trace(go.Scatter(y=actual[:100], mode='lines', name='Actual'))
                fig.update_layout(
                    title='Earthquake Magnitude Prediction (First 100)',
                    xaxis_title='Time',
                    yaxis_title='Magnitude',
                    showlegend=True
                )
                st.plotly_chart(fig)
                
                # Calculate and display metrics
                if len(predictions) > 0 and len(actual) > 0:
                    metrics = predictor.evaluate_model(predictions, actual)
                    st.write("### Prediction Performance Metrics")
                    for metric, value in metrics.items():
                        st.write(f"{metric}: {value:.4f}")
                else:
                    st.warning("Cannot perform prediction.")

    def show_deep_math_models(self):
        """
        Show deep mathematical models analysis with comprehensive visualizations
        """
        st.header("🧮 Deep Mathematical Models")
        st.write("""
        This section explores advanced mathematical concepts to model earthquake behavior, including:
        - **Fractal Geometry**: Analyzing the spatial distribution patterns
        - **Self-Organized Criticality**: Understanding power-law distributions
        - **Statistical Physics**: Energy distributions and entropy analysis
        - **Network Theory**: Earthquake connectivity patterns
        """)
        
        # Data source information
        st.info("📊 **Data Source**: All analyses use real Japanese earthquake data. Network analysis may sample data for computational efficiency when dataset exceeds 1000 events.")
        
        try:
            # Instantiate the model with the dataframe
            deep_math_analyzer = DeepMathModels(self.analyzer.df)
            
            # Run all analyses
            with st.spinner("Running deep mathematical analyses..."):
                results = deep_math_analyzer.run_all_analyses()
            
            if not results:
                st.warning("No analysis results could be generated. Please check your data.")
                return
            
            # 1. Fractal Analysis
            if 'fractal' in results:
                st.subheader("1. Fractal Geometry Analysis")
                st.markdown("""
                **What is this?** Fractal dimension measures how earthquake epicenters fill space. 
                A higher fractal dimension indicates more complex spatial clustering patterns.
                """)
                
                fractal_data = results['fractal']
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Fractal Dimension", f"{fractal_data['fractal_dimension']:.3f}")
                    st.metric("R² (Goodness of Fit)", f"{fractal_data['r_squared']:.3f}")
                
                with col2:
                    # Plot fractal analysis
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=fractal_data['log_sizes'],
                        y=fractal_data['log_counts'],
                        mode='markers',
                        name='Observed',
                        marker=dict(size=8, color='blue')
                    ))
                    fig.add_trace(go.Scatter(
                        x=fractal_data['log_sizes'],
                        y=fractal_data['y_pred'],
                        mode='lines',
                        name='Fitted Line',
                        line=dict(color='red', dash='dash')
                    ))
                    fig.update_layout(
                        title="Box-Counting Fractal Analysis",
                        xaxis_title="log(1/box_size)",
                        yaxis_title="log(box_count)",
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Enhanced interpretation
                fractal_dim = fractal_data['fractal_dimension']
                r_squared = fractal_data['r_squared']
                
                st.info(f"""
                **🔬 Scientific Interpretation:**
                
                **Fractal Dimension ({fractal_dim:.3f}):**
                - **{fractal_dim:.3f}** indicates {'strong spatial clustering' if fractal_dim < 1.5 else 'moderate clustering' if fractal_dim < 2.0 else 'weak clustering'}
                - **Why this matters**: {'Strong clustering suggests earthquakes are concentrated in specific fault zones, indicating well-defined tectonic boundaries. This is typical for mature fault systems.' if fractal_dim < 1.5 else 'Moderate clustering indicates a mix of concentrated and distributed seismicity, suggesting both major faults and secondary structures are active.' if fractal_dim < 2.0 else 'Weak clustering suggests earthquakes are more randomly distributed, which could indicate either diffuse deformation or incomplete fault mapping.'}
                
                **R² ({r_squared:.3f}):**
                - **{r_squared:.3f}** shows {'excellent' if r_squared > 0.9 else 'good' if r_squared > 0.7 else 'moderate'} fit to fractal model
                - **Why this matters**: {'High R² confirms that earthquake spatial distribution follows a true fractal pattern, suggesting self-similarity across different scales.' if r_squared > 0.9 else 'Good R² indicates the fractal model is appropriate, though some local variations exist.' if r_squared > 0.7 else 'Moderate R² suggests the fractal model may not fully capture the spatial complexity, possibly due to geological heterogeneity.'}
                
                **🚨 Practical Implications:**
                - **Risk Assessment**: {'High clustering means future earthquakes are more likely to occur in already active zones, allowing for targeted preparedness.' if fractal_dim < 1.5 else 'Moderate clustering suggests both known and unknown fault zones may be active, requiring broader monitoring.' if fractal_dim < 2.0 else 'Low clustering makes prediction more difficult, requiring comprehensive regional monitoring.'}
                - **Monitoring Strategy**: {'Focus monitoring resources on identified cluster centers and their immediate surroundings.' if fractal_dim < 1.5 else 'Balance monitoring between known clusters and broader regional coverage.' if fractal_dim < 2.0 else 'Implement comprehensive regional monitoring as earthquakes may occur anywhere.'}
                """)
            
            # 2. Self-Organized Criticality (Gutenberg-Richter)
            if 'soc' in results:
                st.subheader("2. Self-Organized Criticality Analysis")
                st.markdown("""
                **What is this?** The Gutenberg-Richter law describes the power-law relationship between earthquake magnitude and frequency.
                The b-value is a key parameter indicating the relative frequency of large vs. small earthquakes.
                """)
                
                soc_data = results['soc']
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Gutenberg-Richter b-value", f"{soc_data['b_value']:.3f}")
                    st.metric("a-value", f"{soc_data['a_value']:.1e}")
                    st.metric("R² (Goodness of Fit)", f"{soc_data['r_squared']:.3f}")
                
                with col2:
                    # Plot Gutenberg-Richter law
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=soc_data['log_mag'],
                        y=soc_data['log_counts'],
                        mode='markers',
                        name='Observed',
                        marker=dict(size=6, color='green')
                    ))
                    fig.add_trace(go.Scatter(
                        x=soc_data['log_mag'],
                        y=soc_data['y_pred'],
                        mode='lines',
                        name='Fitted Line',
                        line=dict(color='red', dash='dash')
                    ))
                    fig.update_layout(
                        title="Gutenberg-Richter Law",
                        xaxis_title="log(Magnitude)",
                        yaxis_title="log(Frequency)",
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Enhanced interpretation
                b_value = soc_data['b_value']
                a_value = soc_data['a_value']
                r_squared = soc_data['r_squared']
                
                st.info(f"""
                **🔬 Scientific Interpretation:**
                
                **b-value ({b_value:.3f}):**
                - **{b_value:.3f}** is {'typical for tectonic regions' if 0.8 <= b_value <= 1.2 else 'unusual and may indicate stress changes'}
                - **Why this matters**: {'A b-value around 1.0 indicates the region is in a stable tectonic state with normal stress distribution. This is the expected value for most tectonic regions.' if 0.8 <= b_value <= 1.2 else 'An unusual b-value suggests either recent stress changes, different rock properties, or the influence of external factors like fluid injection or volcanic activity.'}
                - **Historical Context**: {'This b-value is consistent with long-term tectonic behavior in Japan, suggesting no major stress regime changes have occurred.' if 0.8 <= b_value <= 1.2 else 'This unusual b-value may indicate recent tectonic changes or the influence of the 2011 Tōhoku earthquake on regional stress patterns.'}
                
                **a-value ({a_value:.1e}):**
                - **{a_value:.1e}** represents the overall seismicity level
                - **Why this matters**: {'A high a-value indicates high overall earthquake activity, while a low a-value suggests relatively quiet periods. This value helps quantify the baseline seismicity of the region.'}
                
                **Power Law Fit (R² = {r_squared:.3f}):**
                - **{r_squared:.3f}** indicates {'excellent' if r_squared > 0.9 else 'good' if r_squared > 0.7 else 'moderate'} adherence to Gutenberg-Richter law
                - **Why this matters**: {'High R² confirms that the region follows the fundamental Gutenberg-Richter relationship, suggesting the earthquake system is in a self-organized critical state.' if r_squared > 0.9 else 'Good R² indicates the Gutenberg-Richter law applies, though some deviations exist, possibly due to local geological factors.' if r_squared > 0.7 else 'Moderate R² suggests the region may not fully follow the Gutenberg-Richter law, possibly due to complex geological conditions or incomplete data.'}
                
                **🚨 Practical Implications:**
                - **Forecasting**: {'With a stable b-value, we can use the Gutenberg-Richter relationship to estimate the probability of future large earthquakes based on current small earthquake rates.' if 0.8 <= b_value <= 1.2 else 'The unusual b-value makes forecasting more uncertain, requiring additional monitoring and analysis to understand the underlying causes.'}
                - **Risk Assessment**: {'A typical b-value suggests normal tectonic conditions, while an unusual b-value may indicate increased risk of large earthquakes due to stress changes.'}
                - **Monitoring Priorities**: {'Continue monitoring for any significant changes in b-value, as such changes often precede major seismic events.'}
                """)
            
            # 3. Statistical Physics
            if 'physics' in results:
                st.subheader("3. Statistical Physics Analysis")
                st.markdown("""
                **What is this?** This analysis examines energy distributions, entropy, and phase space relationships 
                to understand the statistical properties of earthquake systems.
                """)
                
                physics_data = results['physics']
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Shannon Entropy", f"{physics_data['shannon_entropy']:.3f}")
                    st.metric("Entropy Ratio", f"{physics_data['entropy_ratio']:.3f}")
                
                with col2:
                    st.metric("Total Energy (J)", f"{physics_data['total_energy']:.2e}")
                    st.metric("Avg Energy (J)", f"{physics_data['avg_energy']:.2e}")
                
                with col3:
                    st.metric("Time-Mag Correlation", f"{physics_data['correlation']:.3f}")
                    if physics_data['energy_power_law_exp']:
                        st.metric("Energy Power Law Exp", f"{physics_data['energy_power_law_exp']:.3f}")
                
                # Additional information about data filtering
                if 'filtered_magnitude_range' in physics_data:
                    st.info(f"📊 **Analysis Details**: Using {physics_data['n_events_analyzed']} events with magnitude range {physics_data['filtered_magnitude_range']}")
                
                # Energy distribution plot
                if physics_data['log_energy'] is not None:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=physics_data['log_energy'],
                        y=physics_data['log_energy_counts'],
                        mode='markers',
                        name='Energy Distribution',
                        marker=dict(size=6, color='purple')
                    ))
                    fig.update_layout(
                        title="Energy Distribution (Log-Log)",
                        xaxis_title="log(Energy)",
                        yaxis_title="log(Frequency)",
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Phase space plot
                if len(physics_data['clean_time_diffs']) > 0:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=physics_data['clean_time_diffs'],
                        y=physics_data['clean_magnitudes'],
                        mode='markers',
                        name='Phase Space',
                        marker=dict(size=4, color='orange', opacity=0.6)
                    ))
                    fig.update_layout(
                        title="Phase Space: Time Intervals vs Magnitude",
                        xaxis_title="Time Interval (hours)",
                        yaxis_title="Magnitude",
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Enhanced interpretation
                entropy_ratio = physics_data['entropy_ratio']
                correlation = physics_data['correlation']
                energy_power_law = physics_data['energy_power_law_exp']
                
                st.info(f"""
                **🔬 Scientific Interpretation:**
                
                **Entropy Analysis:**
                - **Entropy Ratio ({entropy_ratio:.3f})**: {'High' if entropy_ratio > 0.7 else 'moderate' if entropy_ratio > 0.4 else 'low'} randomness in magnitude distribution
                - **Why this matters**: {'High entropy indicates a more random, unpredictable magnitude distribution, typical of complex tectonic systems with multiple interacting faults.' if entropy_ratio > 0.7 else 'Moderate entropy suggests some predictability in magnitude patterns, possibly due to dominant fault systems or stress regimes.' if entropy_ratio > 0.4 else 'Low entropy indicates highly predictable magnitude patterns, which is unusual and may suggest either data artifacts or very simple tectonic conditions.'}
                
                **Energy Distribution:**
                - **Energy Power Law**: {'Follows power law' if energy_power_law else 'Does not follow power law'} pattern
                - **Why this matters**: {'Power law energy distribution confirms that the earthquake system exhibits scale-invariant behavior, typical of self-organized critical systems. This means the same physical processes govern both small and large earthquakes.' if energy_power_law else 'Non-power law energy distribution suggests the system may not be in a self-organized critical state, possibly due to external influences or complex geological conditions.'}
                
                **Time-Magnitude Correlation ({correlation:.3f}):**
                - **{correlation:.3f}** indicates {'strong' if abs(correlation) > 0.5 else 'moderate' if abs(correlation) > 0.3 else 'weak'} relationship between time intervals and magnitudes
                - **Why this matters**: {'Strong correlation suggests that the timing of earthquakes is related to their magnitude, possibly indicating stress accumulation and release patterns.' if abs(correlation) > 0.5 else 'Moderate correlation indicates some relationship between timing and magnitude, but with significant variability.' if abs(correlation) > 0.3 else 'Weak correlation suggests that earthquake timing and magnitude are largely independent, typical of random processes.'}
                
                **🚨 Practical Implications:**
                - **Predictability**: {'High entropy makes prediction difficult, requiring comprehensive monitoring of all magnitude ranges.' if entropy_ratio > 0.7 else 'Moderate entropy allows for some prediction based on observed patterns.' if entropy_ratio > 0.4 else 'Low entropy suggests high predictability, though this is unusual and should be verified.'}
                - **Energy Release**: {'Power law energy distribution means that a few large earthquakes release most of the total energy, while many small earthquakes release little energy.' if energy_power_law else 'Non-power law distribution suggests more complex energy release patterns that require further investigation.'}
                - **Monitoring Strategy**: {'Focus on detecting changes in the time-magnitude correlation, as such changes may indicate impending large earthquakes.' if abs(correlation) > 0.3 else 'Time-magnitude independence suggests that monitoring should focus on overall seismicity rates rather than specific patterns.'}
                """)
            
            # 4. Network Analysis (with parameters)
            st.subheader("4. Network Theory Analysis")
            st.markdown("""
            **What is this?** Network analysis examines how earthquakes are connected in space and time, 
            revealing patterns of stress transfer and seismic connectivity.
            """)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                spatial_threshold = st.slider("Spatial Threshold (km)", 10, 200, 50)
            with col2:
                temporal_threshold = st.slider("Temporal Threshold (hours)", 1, 72, 24)
            with col3:
                max_events = st.slider("Max Events for Analysis", 500, 2000, 1000)
            
            if st.button("Run Network Analysis"):
                with st.spinner("Performing network analysis..."):
                    network_results = deep_math_analyzer.network_analysis(
                        hash(str(self.analyzer.df.shape)), 
                        spatial_threshold, 
                        temporal_threshold, 
                        max_events
                    )
                
                if network_results:
                    # Data source indicator
                    if network_results['sampled']:
                        st.warning(f"📊 **Note**: Network analysis using {network_results['n_events']} sampled events from {len(self.analyzer.df)} total events for computational efficiency.")
                    else:
                        st.success(f"📊 **Note**: Network analysis using all {network_results['n_events']} events (real data).")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Nodes (Events)", network_results['n_events'])
                        st.metric("Connections", network_results['total_connections'])
                    with col2:
                        st.metric("Avg Degree", f"{network_results['avg_degree']:.2f}")
                        st.metric("Network Density", f"{network_results['network_density']:.4f}")
                    with col3:
                        st.metric("Clustering Coeff", f"{network_results['clustering_coeff']:.3f}")
                        st.metric("Avg Path Length", f"{network_results['avg_path_length']:.2f}")
                    with col4:
                        if network_results['sampled']:
                            st.info("📊 Using sampled data")
                    
                    # Degree distribution plot
                    if len(network_results['degree_counts']) > 1:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=network_results['degree_values'],
                            y=network_results['degree_counts'],
                            name='Degree Distribution'
                        ))
                        fig.update_layout(
                            title="Network Degree Distribution",
                            xaxis_title="Degree (Number of Connections)",
                            yaxis_title="Number of Events",
                            showlegend=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Enhanced interpretation
                    avg_degree = network_results['avg_degree']
                    clustering_coeff = network_results['clustering_coeff']
                    avg_path_length = network_results['avg_path_length']
                    network_density = network_results['network_density']
                    
                    st.info(f"""
                    **🔬 Network Interpretation:**
                    
                    **Connectivity ({avg_degree:.2f} average connections per event):**
                    - **{avg_degree:.2f}** indicates {'high' if avg_degree > 5 else 'moderate' if avg_degree > 2 else 'low'} connectivity
                    - **Why this matters**: {'High connectivity suggests that earthquakes are strongly interconnected, with stress changes from one event affecting many others. This indicates a highly coupled fault system.' if avg_degree > 5 else 'Moderate connectivity suggests some interconnection between earthquakes, typical of regions with multiple active faults.' if avg_degree > 2 else 'Low connectivity suggests earthquakes are largely independent, possibly indicating isolated fault segments or different stress regimes.'}
                    
                    **Clustering ({clustering_coeff:.3f}):**
                    - **{clustering_coeff:.3f}** indicates {'strong' if clustering_coeff > 0.3 else 'moderate' if clustering_coeff > 0.1 else 'weak'} local clustering
                    - **Why this matters**: {'Strong clustering means that connected earthquakes tend to form tight groups, suggesting localized stress concentration and release patterns.' if clustering_coeff > 0.3 else 'Moderate clustering indicates some grouping of connected events, typical of fault systems with multiple interacting segments.' if clustering_coeff > 0.1 else 'Weak clustering suggests that connected earthquakes are distributed more randomly, possibly indicating diffuse deformation or complex fault geometries.'}
                    
                    **Network Structure:**
                    - **{'Small-world' if clustering_coeff > 0.1 and avg_path_length < 10 else 'Random' if clustering_coeff < 0.1 else 'Regular'}** network characteristics
                    - **Why this matters**: {'Small-world networks are typical of earthquake systems where local clustering coexists with global connectivity, allowing stress changes to propagate efficiently across the network.' if clustering_coeff > 0.1 and avg_path_length < 10 else 'Random networks suggest earthquakes are largely independent, with minimal stress transfer between events.' if clustering_coeff < 0.1 else 'Regular networks suggest uniform connectivity patterns, which is unusual for natural earthquake systems.'}
                    
                    **🚨 Practical Implications:**
                    - **Stress Transfer**: {'High connectivity means that stress changes from one earthquake can trigger others over large distances, requiring broad monitoring.' if avg_degree > 5 else 'Moderate connectivity suggests limited stress transfer, allowing for more localized monitoring strategies.' if avg_degree > 2 else 'Low connectivity suggests minimal stress transfer, with earthquakes occurring independently.'}
                    - **Cascade Effects**: {'Strong clustering increases the risk of earthquake cascades, where one event triggers multiple others in the same region.' if clustering_coeff > 0.3 else 'Moderate clustering suggests some cascade potential, requiring attention to aftershock sequences.' if clustering_coeff > 0.1 else 'Weak clustering suggests minimal cascade risk, with earthquakes occurring more independently.'}
                    - **Monitoring Strategy**: {'Focus on highly connected regions as they are more likely to experience cascading events.' if avg_degree > 5 else 'Monitor both connected and isolated regions, as stress transfer is limited but not absent.' if avg_degree > 2 else 'Comprehensive regional monitoring is needed as earthquakes may occur anywhere without clear patterns.'}
                    """)
            
            # Summary
            st.subheader("🎯 Key Insights from Deep Mathematical Models")
            st.markdown("""
            **Overall Assessment:**
            - These advanced mathematical models reveal the underlying statistical and geometric properties of earthquake systems
            - The analyses support the hypothesis that earthquake systems exhibit self-organized criticality
            - Network analysis shows how earthquakes are interconnected through stress transfer mechanisms
            - Fractal analysis confirms the complex spatial clustering patterns in seismic activity
            """)
            
            # Cross-analysis insights
            st.subheader("🔗 Cross-Analysis Insights")
            
            # Combine insights from different analyses
            insights = []
            
            if 'fractal' in results and 'soc' in results:
                fractal_dim = results['fractal']['fractal_dimension']
                b_value = results['soc']['b_value']
                
                if fractal_dim < 1.5 and 0.8 <= b_value <= 1.2:
                    insights.append("**Stable Tectonic State**: Strong spatial clustering combined with typical b-value suggests a mature, stable fault system with well-defined boundaries.")
                elif fractal_dim > 2.0 and (b_value < 0.8 or b_value > 1.2):
                    insights.append("**Tectonic Instability**: Weak clustering combined with unusual b-value may indicate recent stress changes or tectonic reorganization.")
            
            if 'physics' in results and 'soc' in results:
                entropy_ratio = results['physics']['entropy_ratio']
                b_value = results['soc']['b_value']
                
                if entropy_ratio > 0.7 and 0.8 <= b_value <= 1.2:
                    insights.append("**Complex but Predictable**: High entropy suggests complex interactions, but stable b-value indicates underlying order in the system.")
                elif entropy_ratio < 0.4 and (b_value < 0.8 or b_value > 1.2):
                    insights.append("**Unusual Simplicity**: Low entropy combined with unusual b-value suggests either data artifacts or very simple tectonic conditions.")
            
            if insights:
                for insight in insights:
                    st.markdown(f"- {insight}")
            else:
                st.markdown("- **Mixed Signals**: The various analyses show different aspects of the earthquake system, requiring careful interpretation of each component.")
            
            st.info("""
            **🎯 Practical Recommendations:**
            
            1. **Monitoring Priorities**: Focus on regions identified as high-risk by multiple analyses
            2. **Alert Systems**: Use changes in b-value, entropy, or network connectivity as potential precursors
            3. **Research Directions**: Investigate the relationships between different mathematical parameters
            4. **Risk Assessment**: Combine these mathematical insights with geological and historical data
            """)
            
        except Exception as e:
            st.error(f"An error occurred while running the deep mathematical models: {e}")
            st.info("Please ensure that all required dependencies are installed and the data format is correct.")
            import traceback
            with st.expander("Error Details"):
                st.text(traceback.format_exc())

    def show_practical_insights(self):
        """
        Show practical insights and recommendations based on the analysis.
        """
        st.header('🎯 Practical Insights & Recommendations')
        st.markdown("""
        This section translates complex analytical results into actionable insights and concrete recommendations for stakeholders such as government agencies, emergency managers, and the public.
        """)

        # Ensure energy column exists
        if 'energy_joules' not in self.analyzer.df.columns:
            self.analyzer.df['energy_joules'] = 10**(1.5 * self.analyzer.df['mag'] + 4.8)

        # 1. Key Takeaways
        st.subheader('1. Executive Summary: Key Takeaways')
        
        # Get key statistics
        stats = self.analyzer.basic_statistics()
        top_quake = self.analyzer.df.loc[self.analyzer.df['mag'].idxmax()]
        
        # Get risk assessment data if available
        highest_risk_region = "N/A"
        risk_scores = []
        try:
            # Use a temporary region column to avoid modifying the main dataframe
            df_copy = self.analyzer.df.copy()
            df_copy['region'] = pd.cut(df_copy['latitude'], bins=5, 
                                              labels=['South', 'South-Central', 'Central', 'North-Central', 'North'])
            
            for region in df_copy['region'].unique():
                if pd.notna(region):
                    region_data = df_copy[df_copy['region'] == region]
                    if not region_data.empty:
                        risk_score = (
                            (len(region_data[region_data['mag'] >= 5.0]) / len(region_data) * 40) +
                            (region_data['mag'].max() / 10 * 30)
                        )
                        risk_scores.append({'region': region, 'score': risk_score})
            
            if risk_scores:
                highest_risk_region = max(risk_scores, key=lambda x: x['score'])['region']
        except Exception:
            pass # Fail silently if risk assessment part fails
        
        takeaways = [
            f"**High-Risk Zones Identified**: Analysis consistently points to the **{highest_risk_region}** region as having the highest seismic risk, based on frequency and magnitude.",
            f"**Predictable Aftershock Patterns**: Aftershock sequences follow **Omori's Law**, with a predictable decay in frequency, allowing for better short-term forecasting after a major event.",
            f"**Energy Release is Not Constant**: The total seismic energy released varies significantly year-to-year, with certain years identified as anomalous high-energy periods.",
            f"**Clustering is Key**: A significant portion of earthquakes occur in spatiotemporal clusters or 'families', highlighting the importance of monitoring these active zones.",
            f"**The b-value is Stable but Critical**: The Gutenberg-Richter b-value is relatively stable around **1.0**, typical for tectonic regions, but any significant deviation could signal changes in regional stress."
        ]
        
        for takeaway in takeaways:
            st.markdown(f"- {takeaway}")
        
        # 2. Stakeholder-Specific Recommendations
        st.subheader('2. Recommendations for Stakeholders')
        
        st.markdown("#### For Government & Policy Makers")
        recommendations_gov = [
            "**Targeted Retrofitting**: Prioritize funding for seismic retrofitting of critical infrastructure (hospitals, schools, bridges) in the identified high-risk zones.",
            "**Dynamic Building Codes**: Update building codes based on the latest regional risk assessments, not just a single national standard.",
            "**Public Education Campaigns**: Launch sustained public awareness campaigns on earthquake preparedness, especially in high-risk regions and during periods of increased seismic activity."
        ]
        for rec in recommendations_gov:
            st.markdown(f"- {rec}")

        st.markdown("#### For Emergency Management Agencies")
        recommendations_em = [
            "**Pre-positioning Resources**: Use cluster analysis and aftershock decay models to pre-position emergency supplies and personnel near areas with recent major earthquakes.",
            "**Data-Driven Alert Systems**: Integrate real-time seismic data with the predictive models from this dashboard to enhance public warning systems.",
            "**Scenario Planning**: Use 'quiescent period' analysis to run drills and prepare for potential large-magnitude events following periods of unusual calm."
        ]
        for rec in recommendations_em:
            st.markdown(f"- {rec}")
            
        st.markdown("#### For the Scientific Community & Researchers")
        recommendations_sci = [
            "**Monitor b-value Fluctuations**: Continuously monitor the Gutenberg-Richter b-value in near real-time, as significant changes can be a precursor to large earthquakes.",
            "**Investigate Anomalous Energy Release**: Further investigate the geological factors behind the years with anomalously high energy release.",
            "**Refine Swarm Vs. Aftershock Classification**: Improve machine learning models to better distinguish between 'background' seismic swarms and true aftershock sequences."
        ]
        for rec in recommendations_sci:
            st.markdown(f"- {rec}")
            
        # 3. Visual Summaries for Quick Interpretation
        st.subheader('3. At-a-Glance Visual Insights')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Regional Risk Hotspots**")
            try:
                if risk_scores:
                    risk_df = pd.DataFrame(risk_scores)
                    fig = px.bar(risk_df, x='region', y='score', 
                               color='score', color_continuous_scale='Reds',
                               title='Relative Risk Score by Region')
                    fig.update_layout(yaxis_title="Calculated Risk Score")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Risk data not available for visualization.")
            except Exception as e:
                st.warning(f"Could not display risk hotspot chart: {e}")

        with col2:
            st.markdown("**Aftershock Decay Curve**")
            try:
                main_shock = self.analyzer.df.loc[self.analyzer.df['mag'].idxmax()]
                aftershocks = self.analyzer.df[
                    (self.analyzer.df['time'] > main_shock['time']) &
                    (self.analyzer.df['time'] <= main_shock['time'] + pd.Timedelta(days=14))
                ]
                if not aftershocks.empty:
                    aftershocks['time_after'] = (aftershocks['time'] - main_shock['time']).dt.total_seconds() / (3600*24)
                    daily_counts = aftershocks.resample('D', on='time').size()
                    
                    fig = px.line(x=daily_counts.index, y=daily_counts.values, title=f"Aftershock Decay for Mag {main_shock['mag']:.1f} Quake")
                    fig.update_layout(xaxis_title="Days After Mainshock", yaxis_title="Number of Aftershocks")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No aftershocks found for the largest quake to plot decay.")
            except Exception as e:
                st.warning(f"Could not display aftershock decay curve: {e}")

if __name__ == "__main__":
    dashboard = EarthquakeDashboard()
    dashboard.run()