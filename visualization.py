# 필수 라이브러리 설치
# pip install pandas matplotlib seaborn plotly folium
import matplotlib.pyplot as plt
import matplotlib as mpl

# 한글 폰트 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
mpl.rcParams['axes.unicode_minus'] = False
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# 데이터 로딩 및 전처리
df = pd.read_csv("japanearthquake.csv")
df['time'] = pd.to_datetime(df['time'])

# -------------------
# 시각화 1: 마그니튜드 히스토그램
plt.figure(figsize=(10, 4))
sns.histplot(df['mag'], bins=30, kde=True, color='crimson')
plt.title('① 마그니튜드 분포')
plt.xlabel('Magnitude')
plt.ylabel('Count')
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------
# 시각화 2: 시간에 따른 지진 발생 수
df.set_index('time', inplace=True)
plt.figure(figsize=(12, 4))
df['mag'].resample('M').count().plot(color='teal')
plt.title('② 월별 지진 발생 수')
plt.ylabel('Count')
plt.xlabel('Time')
plt.grid(True)
plt.show()

# -------------------
# 시각화 3: 다변량 관계 Pairplot
sns.pairplot(df.reset_index()[['mag', 'depth', 'gap', 'rms']])
plt.suptitle('③ 지진 변수 간 관계 분석', y=1.02)
plt.show()

# -------------------
# 시각화 4: 심도 vs 마그니튜드 산점도
plt.figure(figsize=(10, 5))
sns.scatterplot(x='depth', y='mag', hue='magType', data=df, alpha=0.7)
plt.title('④ 심도와 마그니튜드의 관계')
plt.xlabel('Depth (km)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------
# 시각화 5: Heatmap
plt.figure(figsize=(8, 6))
corr = df[['mag', 'depth', 'gap', 'rms', 'dmin']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('⑤ 변수 상관관계 Heatmap')
plt.tight_layout()
plt.show()

# -------------------
# 시각화 6: Plotly 멀티서브플롯
fig = make_subplots(rows=1, cols=2, subplot_titles=("⑥ 마그니튜드 분포", "⑦ 심도 분포"))

fig.add_trace(go.Histogram(x=df['mag'], name='Magnitude', marker_color='indianred'), row=1, col=1)
fig.add_trace(go.Histogram(x=df['depth'], name='Depth', marker_color='steelblue'), row=1, col=2)

fig.update_layout(title_text="⑥-⑦ Plotly 서브플롯", showlegend=False)
fig.show()

# -------------------
# 시각화 7: Plotly 인터랙티브 산점도
fig = px.scatter(df.reset_index(), x="depth", y="mag", color="magType",
                 size="rms", hover_data=["place"],
                 title="⑧ 심도와 마그니튜드의 인터랙티브 산점도")
fig.show()

# -------------------
# 시각화 8: 지진 지도 시각화 (Folium)
earthquake_map = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=5)
for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=max(row['mag'], 1.5),
        popup=f"M {row['mag']}, {row['place']}",
        color='darkred',
        fill=True,
        fill_opacity=0.6
    ).add_to(earthquake_map)
earthquake_map.save("earthquake_map.html")
print("⑨ 지도 시각화: 'earthquake_map.html' 파일로 저장되었습니다.")

# -------------------
# 시각화 9: 시간 애니메이션 (Plotly)
fig = px.scatter(df.reset_index().sort_values('time'), x="depth", y="mag",
                 animation_frame=df.reset_index()['time'].dt.strftime('%Y-%m-%d'),
                 size="rms", color="magType", range_y=[df['mag'].min(), df['mag'].max()],
                 title="⑩ 시간에 따른 지진 심도 & 마그니튜드 변화")
fig.show()

