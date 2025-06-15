import pandas as pd

# CSV 파일 읽기
df = pd.read_csv("japanearthquake.csv")

# 데이터 크기와 미리보기
print(df.shape)
df.head()


import pandas as pd

# CSV 파일 불러오기
df = pd.read_csv("japanearthquake.csv")

# 1. 열 이름 확인
print("🔎 Columns:", df.columns.tolist())

# 2. 결측치 확인
print("🧼 Missing values:\n", df.isnull().sum())

# 3. 필요없는 열 제거 (예: 'nst', 'net', 'id', 'status', 'updated', 'magSource' 등 분석에 불필요한 열)
columns_to_drop = ['nst', 'gap', 'dmin', 'rms', 'net', 'id', 'updated',
                   'horizontalError', 'depthError', 'magError', 'magNst',
                   'status', 'locationSource', 'magSource']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# 4. 결측치 제거 또는 보간
df = df.dropna(subset=['time', 'latitude', 'longitude', 'depth', 'mag'])  # 주요 값 없는 행 제거

# 5. 시간 데이터 변환
df['time'] = pd.to_datetime(df['time'])

# 6. 이상치 제거 (예: mag 음수, depth 음수)
df = df[(df['mag'] > 0) & (df['depth'] >= 0)]

# 7. 인덱스 재설정
df.reset_index(drop=True, inplace=True)

# 8. 정제된 데이터 확인
print("🧹 Cleaned DataFrame:")
print(df.info())
print(df.head())

# 전처리 완료된 DataFrame을 CSV 파일로 저장하기
df.to_csv("japanearthquake_cleaned.csv", index=False)