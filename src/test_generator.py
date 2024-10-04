#%%
import numpy as np 
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, AutoDateLocator
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import numpy as np
from scipy.interpolate import CubicSpline
from datetime import datetime, timedelta
from collections import deque


def base_pattern():
    DATA_PATH = Path("../data/AEP_hourly.csv")
    df = pl.read_csv(DATA_PATH)
    oneday_data = df.select(pl.col('AEP_MW')).to_numpy().flatten()[1200:1224]
    oneday_data *= (1 / oneday_data[0]) / 3600
    return oneday_data

def home_worker_pattern(): 
    DATA_PATH = Path("../data/AEP_hourly.csv")
    df = pl.read_csv(DATA_PATH)
    oneday_data = df.select(pl.col('AEP_MW')).to_numpy().flatten()[1200:1224]
    oneday_data *= (1 / oneday_data[0]) / 3600
    _temp = oneday_data[7]
    for _ in range(7, 18):
        oneday_data[_] = np.random.normal(loc=_temp, scale=0.0000001) 
    return oneday_data

def midnight_worker_pattern():
    DATA_PATH = Path("../data/AEP_hourly.csv")
    df = pl.read_csv(DATA_PATH)
    oneday_data = df.select(pl.col('AEP_MW')).to_numpy().flatten()[1200:1224]
    oneday_data *= (1 / oneday_data[0]) / 3600
    dq = deque(oneday_data)
    dq.rotate(12)
    oneday_data = list(dq)
    return oneday_data

def empty_house_pattern():
    DATA_PATH = Path("../data/AEP_hourly.csv")
    df = pl.read_csv(DATA_PATH)
    oneday_data = df.select(pl.col('AEP_MW')).to_numpy().flatten()[1200:1224]
    oneday_data *= (1 / oneday_data[0]) / 3600
    for _ in range(len(oneday_data)):
            oneday_data[_] = 0
    return oneday_data

def defiy_amount(object_list: list):
    min_k = 6 / 52000
    max_k = 15 / 52000
    object_list = np.array(object_list)
    object_list = object_list * np.random.uniform(min_k, max_k)
    return object_list 

def interpolation(pattern: str):
    data = pattern_saver[pattern]
    hours = np.arange(0, 24)   
    power_consumption = np.array(data)  

    # スプライン補間
    cs = CubicSpline(hours, power_consumption)
    seconds = np.arange(0, 24*3600)
    start_time = datetime(2024, 1, 1)
    date_index = [start_time + timedelta(seconds=int(i)) for i in seconds]

    # 1秒ごとの電力消費値を計算
    second_values = cs(seconds / 3600)  # 時間を秒単位から時間単位に変換
    # Normalize second_values to [0, 1]
    if np.max(second_values) == np.min(second_values):
        normalized_values = np.zeros_like(second_values)
    else:
        normalized_values = (second_values - np.min(second_values)) / (np.max(second_values) - np.min(second_values))
    
    return date_index, normalized_values

def add_noise(data, noise_level):
    return data + np.random.normal(0, noise_level, data.shape)


def generate_data(pattern: str):
    date_index, normalized_values = interpolation(pattern)
    values = defiy_amount(normalized_values)
    add_noise(values, values.mean() * 0.1)
    return date_index, values


#生のコード
#整理必要

# pre_Data = base_pattern()
# time, data = generate_data("base")
# num_of_house = 2
# df = pl.DataFrame({
#     "Time": time,
#     "House1_P": data
# })
# for _ in range(num_of_house):
#     df = df.with_columns([pl.Series(f"House{_ + 2}_P", generate_data("base")[1])])

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import pandas as pd 
import copy 
import itertools
import check
from tqdm import tqdm
import importlib
importlib.reload(check)


num_of_house = 6

# データの読み込み
df = check.reshape_data()

# newdf = pd.DataFrame()
# stacked = np.empty((0, 12), dtype=float)
# for _ in range(df.shape[0]):
#     voltage = np.random.normal(100, 5, 5)
#     one_row_power = df.iloc[_, :len(df.columns) - 1].to_numpy()
#     current = one_row_power / voltage
#     I0 = - sum(current)
#     V0 = 100
#     updated_voltage = np.append(voltage, V0)
#     updated_current = np.append(current, I0)
#     combined = np.empty((updated_current.size + updated_voltage.size,), dtype=updated_current.dtype)
#     combined[0::2] = updated_current
#     combined[1::2] = updated_voltage

#     stacked = pd.DataFrame(np.vstack((stacked, combined)))
# stacked.to_csv("/Users/lawliet/study/theft_detection_simulator/data/combined_data.csv")

stacked = pd.read_csv("/Users/lawliet/study/theft_detection_simulator/data/combined_data.csv")

MinMax = MinMaxScaler()
stacked = MinMax.fit_transform(stacked)

train_data, test_data = train_test_split(stacked, test_size=0.9, random_state=42)
np.nan_to_num(train_data, 0)
np.nan_to_num(test_data, 0)
anomalies = test_data.copy()
anomalies[:, 0] *= 2


# gammas = np.arange(0.1, 0.9, 0.1)
# nus = np.arange(0.1, 0.9, 0.1)
ocsvm = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.1)
ocsvm.fit(train_data)
predictions = ocsvm.predict(test_data)
anomalies_predictions = ocsvm.predict(anomalies)

distances = ocsvm.decision_function(anomalies)
fpr, tpr, thresholds = roc_curve(anomalies_predictions, distances)
roc_auc = auc(fpr, tpr)





#%%

train_data = copy.deepcopy(df)
train_data = train_data.iloc[0:1000, :] 
train_data = train_data.fillna(0)
train_data = train_data.to_numpy()
test_data = copy.deepcopy(df)
test_data = test_data.iloc[1000:1020, :]
test_data = test_data.fillna(0)
test_data = test_data.to_numpy()
test_data[:10, 0] *= 1.001


# gammas = np.arange(0.1, 0.9, 0.01)
# nus = np.arange(0.1, 0.9, 0.01)
gammas = [0.1]
nus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9]
TP = 0 
FP = 0
TN = 0
FN = 0
FPRS = []
TPRS = []

for i in gammas:
    for j in nus:
        ocsvm = OneClassSVM(kernel='rbf', gamma=i, nu=j)
        ocsvm.fit(train_data)
        predictions = ocsvm.predict(test_data)
        for idx, _ in enumerate(predictions):
            if idx < 10:
                if _ == -1:
                    TP += 1
                else:
                    FN += 1
            else:
                if _ == -1:
                    FP += 1
                else:
                    TN += 1
        print(predictions)
        if FP + TN == 0:
            FPRS.append(0)
        elif TP + FN == 0:
            TPRS.append(0)
        else:
            FPRS.append(FP / (FP + TN))
            TPRS.append(TP / (TP + FN))
        TP = 0
        FP = 0
        TN = 0
        FN = 0

plt.xlabel("FPR")
plt.ylabel("TPR")
plt.scatter(FPRS, TPRS, marker="o", color="red")


#%%



# データの正規化
# scaler = MinMaxScaler()
# scaled_train_data = scaler.fit_transform(train_data)
# scaled_test_data = scaler.transform(test_data)

gammas = [0.1]
nus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9]
FPRS = []
TPRS = []

for _i in gammas:
    for _j in nus:
        start_times = [np.random.randint(0, 24 * 3600) for _ in range(num_of_house - 1)]
        start_times[0] = 0


        point_counter = [0 for _ in range(num_of_house)]
        ocsvm = OneClassSVM(kernel='rbf', gamma=_i, nu=_j)
        ocsvm.fit(train_data)
        predictions = ocsvm.predict(test_data)
        counter_history = [0 for _ in range(len(predictions))]
        for idx, _ in enumerate(predictions):
            if idx == 0:
                if _ == -1:
                    counter_history[idx] = 1
                    for idx2, __ in enumerate(start_times):
                        if __ <= idx and idx <= __ + 600:
                            point_counter[idx2] += 1
                            point_counter[5] += 1
                else:
                    counter_history[idx] = 0
            else:
                if _ == -1:
                    counter_history[idx] = counter_history[idx - 1] + 1
                    for idx2, __ in enumerate(start_times):
                        if __ <= idx and idx <= __ + 600:
                            point_counter[idx2] += 1
                            point_counter[5] += 1
                else:
                    counter_history[idx] = counter_history[idx - 1]


        #judge 
        TP = 0
        FP = 0
        TN = 0
        FN = 0 
        line = np.mean(point_counter[:4])

        for _ in range(4):
            if _ == 0:
                if point_counter[_] >= line:
                    TP += 1
                else:
                    FN += 1
            else:
                if point_counter[_] >= line:
                    FP += 1
                else:
                    TN += 1
        
        FPRS.append(FP / (FP + TN))
        TPRS.append(TP / (TP + FN))

plt.plot(FPRS, TPRS, marker="o", color="red", alpha=0.5)


#%%

for _i in gammas:
    for _j in nus:
        ocsvm = OneClassSVM(kernel='rbf', gamma=_i, nu=_j)
        ocsvm.fit(train_data)

        FP_cnt = 0
        TP_cnt = 0
        FN_cnt = 0
        TN_cnt = 0

        predictions = ocsvm.predict(test_data)
        TN_cnt = np.count_nonzero(predictions == 1)
        FP_cnt = np.count_nonzero(predictions == -1)

        anormaly_predictions = ocsvm.predict(anormaly_data)
        TP_cnt = np.count_nonzero(anormaly_predictions == -1)
        FN_cnt = np.count_nonzero(anormaly_predictions == 1)
        FPRS.append(FP_cnt / (FP_cnt + TN_cnt))
        TPRS.append(TP_cnt / (TP_cnt + FN_cnt))









#%%
point_counter = [0 for _ in range(num_of_house)]
for idx, _ in enumerate(predictions):
    if _ == -1 and idx <= 100:
        point_counter[0] += 1
        point_counter[1] += 1
    elif _ == -1:
        point_counter[0] += 1
        point_counter[np.random.randint(0,10)] += 1

average = sum(point_counter[1:]) / num_of_house

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(point_counter[1:], marker="o", color="red", alpha=0.5)
ax.grid()
# plt.plot(point_counter[1:], marker="o", color="red", alpha=0.5)
# p = plt.hlines([average], 0, 9, "green", linestyles='dashed')
ax.hlines([average + 3 * np.std(point_counter[1:])], 0, 9, "blue", linestyles='dashed')
ax.hlines([average - 3 * np.std(point_counter[1:])], 0, 9, "blue", linestyles='dashed')
ax.hlines([average -  np.std(point_counter[1:])], 0, 9, "blue", linestyles='dashed')
ax.hlines([average + np.std(point_counter[1:])], 0, 9, "blue", linestyles='dashed')
ax.set_xlabel("House")
ax.set_ylabel("Penalty Count")


print("--------------------------------------------")
# print(f"TPR: {point_counter[0] / 100:.2f}")
# print(f"FPR: {average / 100:.2f}")
print("--------------------------------------------")
# plt.boxplot(point_counter[1:])


# 結果の表示
# anomalies = np.where(predictions == -1)
# counter = np.count_nonzero(predictions == -1)
# print(f"異常検知されたインデックス: {}, 異常検知された回数: {counter}回")
# print(f"偽陽性率: {counter / len(predictions) * 100:.2f}%")
# print(predictions)



# # 異常検知されたデータポイントの可視化
# fig = plt.figure()
# ax = fig.add_subplot(211)
# ax1 = fig.add_subplot(212)
# # plt.plot(data_scaled[0], label='odd point')
# for _ in range(num_of_house):
#     ax.plot(data_scaled[_], label=f'House{_ + 1}')
#     ax1.plot(test_data[_], label=f'House{_ + 1}')
# ax.set_xlabel("Time[sec]")
# ax.set_ylabel("Power Consumption[kWh]")
# ax.grid()
# ax1.set_xlabel("Time[sec]")
# ax1.set_ylabel("Power Consumption[kWh]")
# ax1.grid()
# ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
# plt.show()


def generate_pattern_data(house_count: int, time_count: int, step: int=1):
    data_label = [f"I{i//2 + 1}" if i%2 == 0 else f"V{i//2 + 1}" for i in range(house_count * 2)]

    datas = []
    for time in range(time_count):
        index = (time + 1)//3600
        k = oneday_data[index]
        #まずIを求める
        I = [np.random.normal(0.5 * k, 0.007) for _ in range(house_count)]
        I[0] -= sum(I)

        #次にVを求める
        V = [0 for _ in range(house_count)]
        V[0] = 100
        now_current = sum(I[1:])
        for i in range(1, house_count):
            V[i] = V[i - 1] - now_current
            now_current -= I[i]
        
        row = []
        for _ in range(len(I)):
            row.append(I[_])
            row.append(V[_])
        
        datas.append(row)
    
    df = pl.DataFrame(data=datas, schema=data_label)
    y = df.select(pl.col('I3')).to_numpy().flatten()
    yy = df.select(pl.col('I5')).to_numpy().flatten()
    x = np.array([i + 1 for i in range(len(y))])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y, marker="o")
    ax.scatter(x, yy, marker="x", alpha=0.1)
    ax.grid()
    ax.set_xlabel('Time')
    ax.set_ylabel('I [A]')
    plt.show()

    
# %%
