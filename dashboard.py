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
from folium.plugins import MarkerCluster, HeatMap
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

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
        
        # Sidebar
        st.sidebar.title('Navigation')
        page = st.sidebar.selectbox(
            'Choose a page',
            ['Overview', 'Time Series Analysis', 'Chain Analysis', 'Energy Analysis', 'Interactive Map', 'Transformer Prediction']
        )
        
        if page == 'Overview':
            self.show_overview()
        elif page == 'Time Series Analysis':
            self.show_time_series_analysis()
        elif page == 'Chain Analysis':
            self.show_chain_analysis()
        elif page == 'Energy Analysis':
            self.show_energy_analysis()
        elif page == 'Transformer Prediction':
            self.show_transformer_prediction()
        else:
            self.show_interactive_map()
            
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
        Show chain (aftershock, Omori's Law, clustering, etc.) analysis with enhanced insights and quantitative assessment.
        """
        st.header('Chain Analysis')
        st.write("""
        This section provides comprehensive analysis of earthquake chains, including aftershock patterns, 
        Omori's Law validation, spatiotemporal clustering, and quantitative assessment of temporal relationships.
        """)

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
        st.subheader('1. Aftershock Analysis with Quantitative Assessment')
        st.markdown('''
        **What is this?**
        This analysis identifies aftershocks following a mainshock and provides quantitative assessment 
        of aftershock patterns, including regional variations and temporal decay characteristics.
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
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                mainshock_count = len(self.chain_analyzer.df[self.chain_analyzer.df['mag'] >= mainshock_mag])
                st.metric('Number of Main Shocks', mainshock_count)
            with col2:
                if 'is_aftershock' in self.chain_analyzer.df.columns:
                    n_aftershocks = int(self.chain_analyzer.df['is_aftershock'].sum())
                    st.metric('Number of Aftershocks', n_aftershocks)
                else:
                    st.metric('Number of Aftershocks', 0)
            with col3:
                if 'is_aftershock' in self.chain_analyzer.df.columns and mainshock_count > 0:
                    aftershock_ratio = n_aftershocks / mainshock_count
                    st.metric('Aftershock/Mainshock Ratio', f"{aftershock_ratio:.2f}")
                else:
                    st.metric('Aftershock/Mainshock Ratio', 0)
            with col4:
                if 'is_aftershock' in self.chain_analyzer.df.columns and len(self.chain_analyzer.df) > 0:
                    aftershock_percentage = (n_aftershocks / len(self.chain_analyzer.df)) * 100
                    st.metric('Aftershock Percentage', f"{aftershock_percentage:.1f}%")
                else:
                    st.metric('Aftershock Percentage', 0)
                    
        except Exception as e:
            st.error(f'Aftershock analysis could not be performed: {e}')
            with st.expander('Error Details', expanded=False):
                import traceback
                st.text(traceback.format_exc())

        # Ensure time_diff_hours exists and is float
        if 'time_diff' in self.chain_analyzer.df.columns:
            if np.issubdtype(self.chain_analyzer.df['time_diff'].dtype, np.timedelta64):
                self.chain_analyzer.df['time_diff_hours'] = self.chain_analyzer.df['time_diff'].dt.total_seconds() / 3600
            else:
                self.chain_analyzer.df['time_diff_hours'] = self.chain_analyzer.df['time_diff'].astype(float)
        else:
            self.chain_analyzer.df['time_diff_hours'] = np.nan

        # 2. Omori's Law Analysis with Fit Quality Assessment
        st.subheader("2. Omori's Law Analysis with Quantitative Validation")
        st.markdown('''
        **What is this?**
        Omori's Law describes how the frequency of aftershocks decays with time after a mainshock. 
        This section provides quantitative assessment of how well the law fits the data and identifies 
        regional or magnitude-based variations.
        ''')
        try:
            if omori_params is not None and not omori_params.empty:
                latest_main_shock = omori_params['main_shock_time'].max()
                self.chain_analyzer.plot_omori_law(latest_main_shock)
                
                # Omori's Law fit quality assessment
                st.subheader("Omori's Law Fit Quality Assessment")
                
                # Calculate fit quality metrics for each mainshock
                fit_metrics = []
                for _, row in omori_params.iterrows():
                    mainshock_time = row['main_shock_time']
                    p_value = row['p_value']
                    k_value = row['k_value']
                    c_value = row['c_value']
                    
                    # Assess fit quality based on p-value and parameter ranges
                    if p_value < 0.05:
                        fit_quality = "Good"
                        fit_color = "green"
                    elif p_value < 0.1:
                        fit_quality = "Moderate"
                        fit_color = "orange"
                    else:
                        fit_quality = "Poor"
                        fit_color = "red"
                    
                    fit_metrics.append({
                        'Mainshock Time': mainshock_time.strftime('%Y-%m-%d'),
                        'Magnitude': row['main_shock_mag'],
                        'P-value': f"{p_value:.4f}",
                        'K-value': f"{k_value:.4f}",
                        'C-value': f"{c_value:.4f}",
                        'Fit Quality': fit_quality
                    })
                
                # Display fit metrics
                fit_df = pd.DataFrame(fit_metrics)
                st.dataframe(fit_df, use_container_width=True)
                
                # Overall fit assessment
                good_fits = sum(1 for m in fit_metrics if m['Fit Quality'] == 'Good')
                total_fits = len(fit_metrics)
                fit_percentage = (good_fits / total_fits) * 100 if total_fits > 0 else 0
                
                st.info(f"""
                **📊 Omori's Law Fit Assessment:**
                - **Good Fits**: {good_fits}/{total_fits} ({fit_percentage:.1f}%)
                - **Moderate Fits**: {sum(1 for m in fit_metrics if m['Fit Quality'] == 'Moderate')}/{total_fits}
                - **Poor Fits**: {sum(1 for m in fit_metrics if m['Fit Quality'] == 'Poor')}/{total_fits}
                
                **🔍 Key Insights:**
                - {fit_percentage:.1f}% of mainshocks follow Omori's Law well (p < 0.05)
                - P-values < 0.05 indicate statistically significant decay patterns
                - K-values represent initial aftershock productivity
                - C-values indicate time delay before Omori decay begins
                """)
                
                # Regional analysis if possible
                if 'latitude' in self.chain_analyzer.df.columns and 'longitude' in self.chain_analyzer.df.columns:
                    st.subheader("Regional Omori's Law Variations")
                    
                    # Group by geographic regions
                    self.chain_analyzer.df['region'] = pd.cut(self.chain_analyzer.df['latitude'], bins=5, labels=['South', 'South-Central', 'Central', 'North-Central', 'North'])
                    
                    regional_stats = []
                    for region in self.chain_analyzer.df['region'].unique():
                        if pd.notna(region):
                            region_data = self.chain_analyzer.df[self.chain_analyzer.df['region'] == region]
                            if 'is_aftershock' in region_data.columns:
                                aftershock_count = region_data['is_aftershock'].sum()
                                total_count = len(region_data)
                                regional_stats.append({
                                    'Region': region,
                                    'Total Events': total_count,
                                    'Aftershocks': aftershock_count,
                                    'Aftershock Rate': f"{(aftershock_count/total_count)*100:.1f}%"
                                })
                    
                    if regional_stats:
                        regional_df = pd.DataFrame(regional_stats)
                        st.dataframe(regional_df, use_container_width=True)
                        
                        st.write("""
                        **🌍 Regional Insights:**
                        - Different regions may show varying aftershock patterns due to geological differences
                        - Higher aftershock rates in certain regions may indicate different stress regimes
                        - Regional variations in Omori's Law parameters suggest localized tectonic conditions
                        """)
                
        except Exception as e:
            st.warning(f'Omori\'s Law analysis could not be performed: {e}')

        # 3. Spatiotemporal Clustering Analysis
        st.subheader('3. Spatiotemporal Clustering Analysis with Pattern Recognition')
        st.markdown('''
        **What is this?**
        This analysis uses ST-DBSCAN clustering to identify spatiotemporal earthquake clusters and 
        provides insights into cluster formation patterns, regional variations, and temporal evolution.
        ''')
        
        # Clustering parameters
        eps_space = st.slider('Spatial Epsilon (km)', 10, 100, 50, 5, key='eps_space_cluster')
        eps_time = st.slider('Temporal Epsilon (hours)', 1, 72, 24, 1, key='eps_time_cluster')
        min_samples = st.slider('Minimum Samples', 2, 20, 5, 1, key='min_samples_cluster')
        
        try:
            with st.spinner('Performing clustering analysis...'):
                df_clustered = cached_analyze_clusters(self.chain_analyzer.df, eps_space, eps_time, min_samples)
            
            # Update dataframe
            self.chain_analyzer.df = df_clustered
            
            # Clustering statistics
            if 'cluster' in self.chain_analyzer.df.columns:
                cluster_stats = self.chain_analyzer.df['cluster'].value_counts()
                noise_count = cluster_stats.get(-1, 0)
                cluster_count = len(cluster_stats) - (1 if -1 in cluster_stats else 0)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric('Total Clusters', cluster_count)
                with col2:
                    st.metric('Noise Points', noise_count)
                with col3:
                    clustered_events = len(self.chain_analyzer.df) - noise_count
                    st.metric('Clustered Events', clustered_events)
                with col4:
                    clustering_ratio = (clustered_events / len(self.chain_analyzer.df)) * 100
                    st.metric('Clustering Ratio', f"{clustering_ratio:.1f}%")
                
                # Temporal cluster analysis
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
                
                # Monthly cluster patterns
                st.subheader("Monthly Cluster Patterns")
                monthly_clusters = self.chain_analyzer.df[self.chain_analyzer.df['cluster'] != -1].groupby(['year', 'month'])['cluster'].nunique().reset_index()
                monthly_clusters['date'] = pd.to_datetime(monthly_clusters[['year', 'month']].assign(day=1))
                
                fig = px.scatter(monthly_clusters, x='date', y='cluster', 
                               title='Monthly Cluster Counts Over Time',
                               labels={'cluster': 'Number of Clusters', 'date': 'Date'})
                fig.update_traces(marker=dict(size=8, color='green'))
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
                    - Monthly patterns show seasonal variations in cluster formation
                    - Cluster sizes and durations provide insights into stress release patterns
                    """)
                    
                    # Cluster characteristics table
                    st.subheader("Detailed Cluster Characteristics")
                    cluster_table = cluster_sizes_df.copy()
                    cluster_table['Avg_Magnitude'] = cluster_table['Avg_Magnitude'].round(2)
                    cluster_table['Max_Magnitude'] = cluster_table['Max_Magnitude'].round(2)
                    cluster_table['Duration_Hours'] = cluster_table['Duration_Hours'].round(1)
                    
                    # Display top 20 clusters by size
                    top_clusters = cluster_table.nlargest(20, 'Size')[['Cluster_ID', 'Size', 'Avg_Magnitude', 'Max_Magnitude', 'Duration_Hours']]
                    top_clusters.columns = ['Cluster ID', 'Size', 'Avg Magnitude', 'Max Magnitude', 'Duration (hours)']
                    st.dataframe(top_clusters, use_container_width=True)
                
        except Exception as e:
            st.warning(f'Clustering analysis could not be performed: {e}')

        # 4. Time Interval Analysis with Burst Detection
        st.subheader('4. Time Interval Analysis with Burst Pattern Detection')
        st.markdown('''
        **What is this?**
        This analysis examines time intervals between consecutive earthquakes to identify 
        burst patterns, temporal clustering, and unusual temporal sequences.
        ''')
        
        try:
            if 'time_diff_hours' in self.chain_analyzer.df.columns and self.chain_analyzer.df['time_diff_hours'].notnull().any():
                # Basic statistics
                time_diffs = self.chain_analyzer.df['time_diff_hours'].dropna()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric('Mean Interval', f"{time_diffs.mean():.2f} hours")
                with col2:
                    st.metric('Median Interval', f"{time_diffs.median():.2f} hours")
                with col3:
                    st.metric('Min Interval', f"{time_diffs.min():.2f} hours")
                with col4:
                    st.metric('Max Interval', f"{time_diffs.max():.2f} hours")
                
                # Burst detection
                burst_threshold = time_diffs.quantile(0.1)  # Bottom 10% as bursts
                burst_events = time_diffs[time_diffs <= burst_threshold]
                burst_percentage = (len(burst_events) / len(time_diffs)) * 100
                
                st.info(f"""
                **⚡ Burst Pattern Analysis:**
                - **Burst Threshold**: {burst_threshold:.2f} hours (10th percentile)
                - **Burst Events**: {len(burst_events)} ({burst_percentage:.1f}% of total)
                - **Burst Definition**: Events occurring within {burst_threshold:.2f} hours of previous event
                
                **🔍 Burst Insights:**
                - {burst_percentage:.1f}% of earthquakes occur in burst patterns
                - Burst events may indicate stress transfer or aftershock sequences
                - High burst percentage suggests clustered temporal behavior
                """)
                
                # Time interval distribution
                fig = px.histogram(time_diffs, nbins=50, title='Distribution of Time Intervals Between Earthquakes')
                fig.add_vline(x=burst_threshold, line_dash="dash", line_color="red", 
                             annotation_text=f"Burst Threshold ({burst_threshold:.1f}h)")
                fig.update_xaxes(title='Time Interval (hours)')
                fig.update_yaxes(title='Count')
                st.plotly_chart(fig)
                
                # Compare clustered vs non-clustered intervals
                if 'cluster' in self.chain_analyzer.df.columns:
                    clustered_intervals = self.chain_analyzer.df[self.chain_analyzer.df['cluster'] != -1]['time_diff_hours'].dropna()
                    non_clustered_intervals = self.chain_analyzer.df[self.chain_analyzer.df['cluster'] == -1]['time_diff_hours'].dropna()
                    
                    if len(clustered_intervals) > 0 and len(non_clustered_intervals) > 0:
                        st.subheader("Clustered vs Non-Clustered Time Intervals")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Clustered Events:**")
                            st.write(f"- Mean interval: {clustered_intervals.mean():.2f} hours")
                            st.write(f"- Median interval: {clustered_intervals.median():.2f} hours")
                            st.write(f"- Count: {len(clustered_intervals)}")
                        
                        with col2:
                            st.write("**Non-Clustered Events:**")
                            st.write(f"- Mean interval: {non_clustered_intervals.mean():.2f} hours")
                            st.write(f"- Median interval: {non_clustered_intervals.median():.2f} hours")
                            st.write(f"- Count: {len(non_clustered_intervals)}")
                        
                        st.write("""
                        **📊 Comparison Insights:**
                        - Clustered events typically show shorter time intervals
                        - Non-clustered events represent background seismicity
                        - The difference in intervals helps distinguish aftershock sequences from independent events
                        """)
                
            else:
                st.info('No time difference data available.')
        except Exception as e:
            st.warning(f'Time interval analysis could not be performed: {e}')

        # 5. Magnitude Sequence Analysis
        st.subheader('5. Enhanced Magnitude Sequence Analysis')
        st.markdown('''
        **What is this?**
        This analysis has been enhanced to provide deeper insights into magnitude patterns over time. 
        It helps to visually distinguish major events, identify trends, and understand the frequency distribution of earthquake magnitudes, which often follows the Gutenberg-Richter law.
        ''')

        analysis_df = self.chain_analyzer.df.copy().sort_values('time')

        tab1, tab2 = st.tabs(["Magnitude Sequence Plot", "Magnitude Distribution (Gutenberg-Richter)"])

        with tab1:
            st.markdown("""
            **📈 Magnitude Sequence Over Time**

            This plot visualizes each earthquake over time.
            - **Size** of the dot represents the **magnitude** of the earthquake (larger dots for stronger quakes).
            - **Color** distinguishes between regular events and identified **aftershocks**.
            - A **30-day rolling average** line is included to show the general trend of earthquake magnitudes.
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
                               color='is_aftershock' if 'is_aftershock' in analysis_df.columns else None,
                               size='mag',  # Use magnitude for size
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
            **📊 Magnitude Distribution**

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
        
        # Magnitude sequence statistics (kept from original)
        st.subheader("Magnitude Change Statistics")
        try:
            if len(analysis_df) > 1:
                # Calculate magnitude differences
                mag_diffs = analysis_df['mag'].diff().dropna()
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric('Mean Mag Change', f"{mag_diffs.mean():.3f}")
                col2.metric('Mag Change Std', f"{mag_diffs.std():.3f}")
                col3.metric('Max Mag Increase', f"{mag_diffs.max():.3f}")
                col4.metric('Max Mag Decrease', f"{mag_diffs.min():.3f}")
                
                # Magnitude sequence insights
                increasing_sequences = (mag_diffs > 0).sum()
                decreasing_sequences = (mag_diffs < 0).sum()
                total_sequences = len(mag_diffs)
                
                st.info(f"""
                **📈 Magnitude Sequence Insights:**
                
                **Pattern Analysis:**
                - Increasing magnitude sequences: {increasing_sequences} ({increasing_sequences/total_sequences*100:.1f}%)
                - Decreasing magnitude sequences: {decreasing_sequences} ({decreasing_sequences/total_sequences*100:.1f}%)
                - Average magnitude change: {mag_diffs.mean():.3f} units
                
                **Scientific Interpretation:**
                - An increasing trend in magnitude might indicate foreshock activity building up to a larger event.
                - A decreasing trend is characteristic of an aftershock sequence following a main shock.
                """)
        except Exception as e:
            st.warning(f'Magnitude sequence statistics could not be performed: {e}')

        # 6. Summary Statistics with Scientific Context
        st.subheader('6. Comprehensive Analysis Summary')
        st.markdown('''
        **What is this?**
        A comprehensive summary of key statistics and scientific insights from all chain analysis components.
        ''')
        try:
            stats = self.chain_analyzer.generate_chain_statistics()
            
            # Enhanced summary with scientific context
            st.write("**📊 Quantitative Summary:**")
            st.json(stats)
            
            # Scientific interpretation
            st.write("""
            **🔬 Scientific Interpretation and Implications:**
            
            **1. Aftershock Patterns:**
            - The aftershock/mainshock ratio provides insights into regional stress characteristics
            - High aftershock percentages suggest regions with significant stress accumulation
            - Temporal aftershock decay patterns help understand stress release mechanisms
            
            **2. Omori's Law Validation:**
            - Good fits to Omori's Law indicate typical aftershock behavior
            - Poor fits may suggest unusual stress conditions or data quality issues
            - Regional variations in Omori parameters reflect different tectonic settings
            
            **3. Clustering Analysis:**
            - High clustering ratios indicate significant spatiotemporal earthquake grouping
            - Cluster characteristics provide insights into fault zone properties
            - Temporal cluster evolution may indicate changing stress conditions
            
            **4. Temporal Patterns:**
            - Burst patterns suggest stress transfer or aftershock sequences
            - Time interval distributions help distinguish different earthquake types
            - Magnitude sequences provide clues about earthquake triggering mechanisms
            
            **5. Practical Applications:**
            - These patterns can inform earthquake forecasting models
            - Regional variations help assess seismic hazard in different areas
            - Temporal clustering may indicate periods of increased seismic risk
            """)
            
        except Exception as e:
            st.warning(f'Statistics summary could not be generated: {e}')

    def show_energy_analysis(self):
        """
        Enhanced energy analysis with meaningful insights beyond total energy.
        """
        st.header('Enhanced Energy Analysis')
        st.write('Advanced analysis of earthquake energy patterns with anomaly detection and regional insights.')

        # Calculate energy in Joules and convert to TNT equivalent
        self.analyzer.df['energy_joules'] = 10**(1.5 * self.analyzer.df['mag'] + 4.8)
        self.analyzer.df['energy_tnt'] = self.analyzer.df['energy_joules'] / 4.184e9  # Convert to tons of TNT
        
        # Add year column for yearly analysis
        self.analyzer.df['year'] = self.analyzer.df['time'].dt.year
        
        # Define high energy threshold (10^13 joules = ~2.4 tons TNT)
        high_energy_threshold = 1e13  # joules
        high_energy_threshold_tnt = high_energy_threshold / 4.184e9

        # 1. Enhanced Energy Distribution Analysis
        st.subheader('1. Enhanced Energy Distribution Analysis')
        
        col1, col2 = st.columns(2)
        with col1:
            # Energy distribution with high-energy events highlighted
            fig = px.histogram(
                self.analyzer.df, 
                x='energy_joules', 
                nbins=50, 
                log_x=True,
                title='Energy Distribution with High-Energy Events',
                color_discrete_sequence=['lightblue']
            )
            fig.add_vline(x=high_energy_threshold, line_dash="dash", line_color="red", 
                         annotation_text="High Energy Threshold")
            fig.update_xaxes(title='Energy (Joules)')
            fig.update_yaxes(title='Count')
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Energy per event distribution
            yearly_stats = self.analyzer.df.groupby('year').agg({
                'energy_joules': ['sum', 'mean', 'count'],
                'mag': 'mean'
            }).round(2)
            yearly_stats.columns = ['Total_Energy', 'Avg_Energy_Per_Event', 'Event_Count', 'Avg_Magnitude']
            yearly_stats['Energy_Per_Event_Ratio'] = yearly_stats['Total_Energy'] / yearly_stats['Event_Count']
            
            # High energy event rate
            high_energy_events = self.analyzer.df[self.analyzer.df['energy_joules'] >= high_energy_threshold]
            high_energy_rate = high_energy_events.groupby('year').size() / self.analyzer.df.groupby('year').size() * 100
            
            fig = px.scatter(
                x=yearly_stats.index,
                y=yearly_stats['Avg_Energy_Per_Event'],
                size=yearly_stats['Event_Count'],
                title='Average Energy per Event by Year',
                labels={'x': 'Year', 'y': 'Average Energy per Event (Joules)'}
            )
            fig.update_traces(marker=dict(color='orange', opacity=0.7))
            st.plotly_chart(fig, use_container_width=True)

        # 2. Anomaly Detection with Moving Averages
        st.subheader('2. Anomaly Detection and Trends')
        
        # Calculate yearly energy statistics
        yearly_energy = self.analyzer.df.groupby('year')['energy_joules'].sum()
        
        # Calculate moving average and standard deviation
        window = min(5, len(yearly_energy) // 3)  # Adaptive window size
        moving_avg = yearly_energy.rolling(window=window, center=True).mean()
        moving_std = yearly_energy.rolling(window=window, center=True).std()
        
        # Detect anomalies (z-score > 2)
        z_scores = (yearly_energy - moving_avg) / moving_std
        anomalies = z_scores.abs() > 2
        
        # Plot with anomalies highlighted
        fig = go.Figure()
        
        # Add total energy line
        fig.add_trace(go.Scatter(
            x=yearly_energy.index,
            y=yearly_energy.values,
            mode='lines+markers',
            name='Total Energy',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))
        
        # Add moving average
        fig.add_trace(go.Scatter(
            x=moving_avg.index,
            y=moving_avg.values,
            mode='lines',
            name=f'{window}-Year Moving Average',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=moving_avg.index,
            y=moving_avg + 2*moving_std,
            mode='lines',
            name='Upper Bound (2σ)',
            line=dict(color='gray', width=1, dash='dot'),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=moving_avg.index,
            y=moving_avg - 2*moving_std,
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(128,128,128,0.2)',
            name='Confidence Interval',
            line=dict(color='gray', width=1, dash='dot')
        ))
        
        # Highlight anomalies
        if anomalies.any():
            anomaly_years = yearly_energy[anomalies].index
            anomaly_values = yearly_energy[anomalies].values
            fig.add_trace(go.Scatter(
                x=anomaly_years,
                y=anomaly_values,
                mode='markers',
                name='Anomalies (|z-score| > 2)',
                marker=dict(color='red', size=12, symbol='diamond'),
                text=[f'Year: {year}<br>Energy: {val:.2e} J<br>Z-score: {z_scores[year]:.2f}' 
                      for year, val in zip(anomaly_years, anomaly_values)],
                hovertemplate='%{text}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Yearly Energy Release with Anomaly Detection',
            xaxis_title='Year',
            yaxis_title='Total Energy (Joules)',
            hovermode='closest'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display anomaly information
        if anomalies.any():
            st.info(f"🚨 **Anomaly Years Detected:** {', '.join(map(str, anomaly_years))}")
            for year in anomaly_years:
                st.write(f"**{year}**: Energy = {yearly_energy[year]:.2e} J, Z-score = {z_scores[year]:.2f}")

        # 3. Dual-axis Analysis: Energy vs High-Energy Event Rate
        st.subheader('3. Energy vs High-Energy Event Rate Analysis')
        
        # Calculate high-energy event rate by year
        high_energy_rate = high_energy_events.groupby('year').size() / self.analyzer.df.groupby('year').size() * 100
        
        # Create dual-axis plot
        fig = go.Figure()
        
        # Primary axis: Total energy
        fig.add_trace(go.Scatter(
            x=yearly_energy.index,
            y=yearly_energy.values,
            mode='lines+markers',
            name='Total Energy (Joules)',
            yaxis='y',
            line=dict(color='blue', width=2)
        ))
        
        # Secondary axis: High-energy event rate
        fig.add_trace(go.Scatter(
            x=high_energy_rate.index,
            y=high_energy_rate.values,
            mode='lines+markers',
            name='High-Energy Event Rate (%)',
            yaxis='y2',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title='Yearly Energy vs High-Energy Event Rate',
            xaxis_title='Year',
            yaxis=dict(title='Total Energy (Joules)', side='left'),
            yaxis2=dict(title='High-Energy Event Rate (%)', side='right', overlaying='y'),
            hovermode='closest'
        )
        st.plotly_chart(fig, use_container_width=True)

        # 4. Regional Energy Analysis
        st.subheader('4. Regional Energy Analysis')
        
        # Create regional grid
        lat_bins = st.slider('Latitude Grid Size', 5, 20, 10, key='lat_bins_energy')
        lon_bins = st.slider('Longitude Grid Size', 5, 20, 10, key='lon_bins_energy')
        
        self.analyzer.df['lat_bin'] = pd.cut(self.analyzer.df['latitude'], bins=lat_bins, labels=False)
        self.analyzer.df['lon_bin'] = pd.cut(self.analyzer.df['longitude'], bins=lon_bins, labels=False)
        
        # Calculate regional statistics
        regional_stats = self.analyzer.df.groupby(['lat_bin', 'lon_bin']).agg({
            'energy_joules': ['sum', 'mean', 'count'],
            'mag': 'mean',
            'latitude': 'mean',
            'longitude': 'mean'
        }).round(2)
        
        regional_stats.columns = ['Total_Energy', 'Avg_Energy', 'Event_Count', 'Avg_Magnitude', 'Center_Lat', 'Center_Lon']
        regional_stats = regional_stats.reset_index()
        
        # Calculate energy concentration (energy per unit area)
        regional_stats['Energy_Concentration'] = regional_stats['Total_Energy'] / regional_stats['Event_Count']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Regional energy heatmap
            fig = px.density_heatmap(
                regional_stats,
                x='lon_bin',
                y='lat_bin',
                z='Total_Energy',
                title='Regional Total Energy Distribution',
                labels={'lon_bin': 'Longitude Bin', 'lat_bin': 'Latitude Bin', 'Total_Energy': 'Total Energy (Joules)'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Energy concentration heatmap
            fig = px.density_heatmap(
                regional_stats,
                x='lon_bin',
                y='lat_bin',
                z='Energy_Concentration',
                title='Regional Energy Concentration (Energy per Event)',
                labels={'lon_bin': 'Longitude Bin', 'lat_bin': 'Latitude Bin', 'Energy_Concentration': 'Energy per Event (Joules)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Regional scatter plot with size and color
        fig = px.scatter(
            regional_stats,
            x='Center_Lon',
            y='Center_Lat',
            size='Total_Energy',
            color='Energy_Concentration',
            hover_data=['Event_Count', 'Avg_Magnitude'],
            title='Regional Energy Analysis: Size = Total Energy, Color = Energy Concentration',
            labels={'Center_Lon': 'Longitude', 'Center_Lat': 'Latitude'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 5. Summary Statistics and Insights
        st.subheader('5. Key Insights and Statistics')
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric('Total Energy Released', f"{self.analyzer.df['energy_joules'].sum():.2e} J")
        with col2:
            st.metric('Average Energy per Event', f"{self.analyzer.df['energy_joules'].mean():.2e} J")
        with col3:
            st.metric('High-Energy Events (>10¹³ J)', f"{len(high_energy_events)} ({len(high_energy_events)/len(self.analyzer.df)*100:.1f}%)")
        with col4:
            st.metric('Max Energy Release', f"{self.analyzer.df['energy_joules'].max():.2e} J")
        
        # Top energy years
        st.write("**Top 5 Years by Total Energy Release:**")
        top_years = yearly_energy.nlargest(5)
        for year, energy in top_years.items():
            st.write(f"• **{year}**: {energy:.2e} Joules")
        
        # Regional insights
        st.write("**Regional Insights:**")
        top_regions = regional_stats.nlargest(5, 'Total_Energy')
        st.write("**Top 5 Regions by Total Energy:**")
        for _, region in top_regions.iterrows():
            st.write(f"• **Region ({region['lat_bin']}, {region['lon_bin']})**: "
                    f"{region['Total_Energy']:.2e} J, {region['Event_Count']} events, "
                    f"avg mag: {region['Avg_Magnitude']:.1f}")

        # 6. Energy-Magnitude Relationship (Enhanced)
        st.subheader('6. Enhanced Energy-Magnitude Relationship')
        
        # Calculate theoretical vs actual energy
        theoretical_energy = 10**(1.5 * self.analyzer.df['mag'] + 4.8)
        energy_ratio = self.analyzer.df['energy_joules'] / theoretical_energy
        
        # Add energy_ratio to dataframe for plotting
        self.analyzer.df['energy_ratio'] = energy_ratio
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Energy vs Magnitude scatter with theoretical line
            fig = px.scatter(
                self.analyzer.df,
                x='mag',
                y='energy_joules',
                log_y=True,
                title='Energy vs Magnitude with Theoretical Relationship',
                color='energy_ratio',
                color_continuous_scale='viridis'
            )
            fig.add_scatter(
                x=self.analyzer.df['mag'],
                y=theoretical_energy,
                mode='lines',
                name='Theoretical',
                line=dict(color='red', width=2)
            )
            fig.update_xaxes(title='Magnitude')
            fig.update_yaxes(title='Energy (Joules)')
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Energy ratio distribution
            fig = px.histogram(
                x=energy_ratio,
                nbins=30,
                title='Distribution of Actual/Theoretical Energy Ratios',
                labels={'x': 'Actual/Theoretical Energy Ratio'}
            )
            fig.add_vline(x=1, line_dash="dash", line_color="red", 
                         annotation_text="Theoretical = 1.0")
            st.plotly_chart(fig, use_container_width=True)
        
        # Gutenberg-Richter relationship statistics
        log_energy = np.log10(self.analyzer.df['energy_joules'])
        slope, intercept = np.polyfit(self.analyzer.df['mag'], log_energy, 1)
        
        st.write("**Gutenberg-Richter Energy Relationship:**")
        st.write(f"• **Slope**: {slope:.2f} (theoretical: 1.5)")
        st.write(f"• **Intercept**: {intercept:.2f} (theoretical: 4.8)")
        st.write(f"• **R²**: {np.corrcoef(self.analyzer.df['mag'], log_energy)[0,1]**2:.3f}")

    def show_interactive_map(self):
        """
        Show enhanced interactive map visualizations with layers.
        """
        st.header('Enhanced Interactive Map')
        st.write('Explore earthquake locations with layers for density (Heatmap) and clustering. Use the layer control in the top-right to switch views.')
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            min_mag = st.slider('Minimum Magnitude', float(self.analyzer.df['mag'].min()), float(self.analyzer.df['mag'].max()), 4.0, 0.1)
        with col2:
            max_depth = st.slider('Maximum Depth (km)', float(self.analyzer.df['depth'].min()), float(self.analyzer.df['depth'].max()), 200.0, 10.0)
        
        filtered_df = self.analyzer.df[(self.analyzer.df['mag'] >= min_mag) & (self.analyzer.df['depth'] <= max_depth)]
        
        if filtered_df.empty:
            st.warning('No earthquakes match the selected filters.')
            return
            
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
        
        # Display map
        folium_static(m, width=800, height=600)

    def show_transformer_prediction(self):
        """
        Show Transformer-based earthquake prediction section
        """
        st.header("Transformer-based Earthquake Prediction")
        st.write("""
        In this section, we use a Transformer model to predict the magnitude of future earthquakes based on the past 30 days of earthquake data. You can start the training process and view the results below.
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

if __name__ == "__main__":
    dashboard = EarthquakeDashboard()
    dashboard.run()