import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1) CSV 읽기 ─ 경로는 상황에 맞게 수정하세요
df = pd.read_csv("japanearthquake.csv")

# 2) 전처리(필요 최소한)
df = df[['time', 'depth', 'mag']].copy()
df = df.dropna(subset=['mag', 'depth'])
df = df[(df['mag'] > 0) & (df['depth'] >= 0)]

# ------------------------------------------------------------------
# 🎨 ① 지진 규모 분포 히스토그램
plt.figure(figsize=(8, 5))
plt.hist(df['mag'], bins=30, edgecolor='k')
plt.title("Magnitude Distribution")
plt.xlabel("Magnitude (Mw)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 🎨 ② 규모별 건수 막대그래프 (0.5 단위 binning)
bin_edges  = np.arange(0, df['mag'].max() + 0.5, 0.5)
bin_labels = [f"{b:.1f}–{b+0.5:.1f}" for b in bin_edges[:-1]]
df['mag_bin'] = pd.cut(df['mag'], bins=bin_edges, labels=bin_labels, right=False)

count_by_bin = df['mag_bin'].value_counts().sort_index()

plt.figure(figsize=(10, 4))
count_by_bin.plot(kind='bar', width=0.8)
plt.title("Number of Earthquakes per 0.5-Magnitude Bin")
plt.xlabel("Magnitude Range")
plt.ylabel("Count")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 🎨 ③ 규모 vs. 깊이 산점도 (색: 규모)
plt.figure(figsize=(7, 6))
scatter = plt.scatter(df['depth'], df['mag'], c=df['mag'],
                      alpha=0.7, cmap='viridis', s=15)
plt.gca().invert_xaxis()              # 깊이가 0에 가까울수록 왼쪽
plt.colorbar(scatter, label='Magnitude')
plt.title("Magnitude vs. Depth")
plt.xlabel("Depth (km)")
plt.ylabel("Magnitude (Mw)")
plt.tight_layout()
plt.show()
