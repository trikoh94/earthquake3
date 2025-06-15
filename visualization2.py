# pip install streamlit pandas matplotlib seaborn plotly folium streamlit-folium

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import folium
from streamlit_folium import st_folium

# Set font for Korean (optional, safe to remove for English)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Load dataset
df = pd.read_csv("japanearthquake.csv")
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

# App Title
st.title("🇯🇵 Japan Earthquake Data Visualization")
st.markdown("This dashboard visualizes various aspects of earthquake activity in Japan using interactive charts and maps.")

# -----------------------
# Section 1: Magnitude Distribution
st.subheader("1. Magnitude Distribution")
fig1, ax1 = plt.subplots()
sns.histplot(df['mag'], bins=30, kde=True, ax=ax1, color='crimson')
ax1.set_title("Magnitude Histogram")
st.pyplot(fig1)
st.markdown("Most earthquakes have a magnitude below 5.0, with stronger ones occurring less frequently.")

# -----------------------
# Section 2: Monthly Earthquake Frequency
st.subheader("2. Earthquake Frequency Over Time")
st.line_chart(df['mag'].resample('M').count())
st.markdown("Earthquakes occur consistently throughout the year, with some periods showing spikes.")

# -----------------------
# Section 3: Depth vs Magnitude
st.subheader("3. Depth vs Magnitude")
fig2, ax2 = plt.subplots()
sns.scatterplot(x='depth', y='mag', hue='magType', data=df.reset_index(), ax=ax2)
ax2.set_title("Depth vs Magnitude")
st.pyplot(fig2)
st.markdown("Shallow earthquakes can still have high magnitudes, posing significant risk.")

# -----------------------
# Section 4: Correlation Heatmap
st.subheader("4. Correlation Heatmap")
fig3, ax3 = plt.subplots()
sns.heatmap(df[['mag', 'depth', 'gap', 'rms']].corr(), annot=True, cmap='coolwarm', ax=ax3)
ax3.set_title("Variable Correlation")
st.pyplot(fig3)
st.markdown("There is limited correlation between depth and magnitude. Gap (location uncertainty) may play a key role.")

# -----------------------
# Section 5: Earthquake Map
st.subheader("5. Earthquake Location Map")
m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=5)
for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=max(row['mag'], 1.5),
        popup=f"M{row['mag']}, {row['place']}",
        color='red',
        fill=True,
        fill_opacity=0.6
    ).add_to(m)
st_data = st_folium(m, width=700, height=500)

# -----------------------
# Section 6: Animated Scatter Plot (Depth vs Magnitude over Time)
st.subheader("6. Earthquake Animation Over Time")
fig = px.scatter(
    df.reset_index().sort_values('time'),
    x="depth",
    y="mag",
    animation_frame=df.reset_index()['time'].dt.strftime('%Y-%m-%d'),
    size="rms",
    color="magType",
    title="Depth vs Magnitude Over Time"
)
st.plotly_chart(fig)

# -----------------------
# Footer
st.markdown("---")
st.markdown("🔍 For more interactive data science tools and templates, visit [GPTOnline.ai/ko](https://gptonline.ai/ko/)")
