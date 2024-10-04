#%%
import numpy as np
import pandas as pd

# 例として時系列データを生成
time_series = np.random.randn(1000)

# pandas DataFrameに変換
data = pd.DataFrame(time_series, columns=['value'])

# 特徴抽出
# ここでは簡単なローリングウィンドウの平均と分散を使用
window_size = 10
data['mean'] = data['value'].rolling(window=window_size).mean()
data['std'] = data['value'].rolling(window=window_size).std()

# ローリングによって作成されたNaN値を削除
data.dropna(inplace=True)

# 特徴を正規化
data['mean'] = (data['mean'] - data['mean'].mean()) / data['mean'].std()
data['std'] = (data['std'] - data['std'].mean()) / data['std'].std()


from sklearn.svm import OneClassSVM

# 訓練用の特徴
X_train = data[['mean', 'std']]

print(X_train)
# OCSVMの訓練
ocsvm = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.1)
ocsvm.fit(X_train)

# 異常を予測
data['anomaly'] = ocsvm.predict(X_train)

# 異常は-1とラベル付けされる
anomalies = data[data['anomaly'] == -1]

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(data.index, data['value'], label='Time Series')
plt.scatter(anomalies.index, anomalies['value'], color='red', label='Anomalies')
plt.legend()
plt.show()

#%%

from hashlib import sha256, sha384

print(sha256.__doc__)
print(sha256(""))
# %%
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import numpy as np

# サンプルの二次元データを作成
data = np.array([[1, 2],
                 [2, 4],
                 [3, 6],
                 [4, 8]])

# 標準化を行う
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
print(data_scaled)

# OCSVMを学習させる
ocsvm = OneClassSVM(kernel='rbf')
ocsvm.fit(data_scaled)

# 予測を行う
predictions = ocsvm.predict(data_scaled)

print("予測結果:", predictions)

# %%
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# 各住宅の電力消費データ（5つの住宅、600秒間）
power_data = np.random.normal(loc=100, scale=5, size=(5, 600))  # 5つの住宅のデータ

# 各住宅ごとの特徴量（平均値、標準偏差など）を抽出
features = []
for house_data in power_data:
    mean = np.mean(house_data)      # 平均
    std = np.std(house_data)        # 標準偏差
    max_value = np.max(house_data)  # 最大値
    min_value = np.min(house_data)  # 最小値
    features.append([mean, std, max_value, min_value])

features = np.array(features)

# スケーリング
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# OCSVMで異常検知
ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)
ocsvm.fit(scaled_features)

# 各住宅ごとの異常検知結果
predictions = ocsvm.predict(scaled_features)

# -1 が異常、1 が正常
for i, result in enumerate(predictions):
    if result == -1:
        print(f"住宅 {i+1} は異常です")
    else:
        print(f"住宅 {i+1} は正常です")


# %%
