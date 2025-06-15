import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # 시각화 향상을 위해

# 1. CSV 파일 불러오기
df = pd.read_csv("japanearthquake.csv")

# 2. 기본 정보 확인
print(df.head())              # 상위 5개 행 미리보기
print(df.info())              # 열 타입과 결측치 정보
print(df.describe())          # 숫자형 변수 통계 요약
print(df.columns)             # 열 이름 확인

# 3. 결측치 확인
print(df.isnull().sum())

# 4. 지진 강도(Magnitude) 분포 시각화
if 'Magnitude' in df.columns:
    plt.hist(df['Magnitude'], bins=30, color='skyblue', edgecolor='black')
    plt.title("Earthquake Magnitude Distribution")
    plt.xlabel("Magnitude")
    plt.ylabel("Frequency")
    plt.show()

# 5. 시간대별 발생 건수 분석 (날짜 컬럼이 있다면)
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day

    df['Year'].value_counts().sort_index().plot(kind='bar', figsize=(10,4))
    plt.title("Number of Earthquakes per Year")
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.show()

# 6. 위치(위도, 경도) 시각화 (지도에 점 찍기)
if {'Latitude', 'Longitude'}.issubset(df.columns):
    plt.figure(figsize=(8, 6))
    plt.scatter(df['Longitude'], df['Latitude'], alpha=0.5, c=df.get('Magnitude', 1), cmap='Reds')
    plt.colorbar(label='Magnitude')
    plt.title("Earthquake Locations in Japan")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.show()
