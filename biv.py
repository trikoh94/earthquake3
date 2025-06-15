import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# CSV 불러오기
df = pd.read_csv("japanearthquake.csv")

# 결측치 제거 (컬럼명 수정 반영)
df = df.dropna(subset=['depth', 'mag'])

# 산점도 + 회귀선
plt.figure(figsize=(10, 6))
sns.regplot(x='depth', y='mag', data=df, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title("Earthquake Depth vs Magnitude with Regression Line")
plt.xlabel("Depth (km)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# Pearson 상관계수
corr, pval = pearsonr(df['depth'], df['mag'])
print(f"📉 Pearson correlation (depth vs mag): {corr:.3f} (p-value: {pval:.3f})")
