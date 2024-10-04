#%%
from pathlib import Path
import numpy as np
import polars as pl
import matplotlib.pyplot as plt 
from data import generate_data
from datetime import datetime

# data_instance = generate_data(100, 100, 0.2, 10)
# y = data_instance.data.select(pl.col('I1')).to_numpy().flatten()
# x = np.array([i + 1 for i in range(len(y))])


# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(x, y, marker="o")
# ax.grid()
# ax.set_xlabel('Time')
# ax.set_ylabel('I [A]')
#%%
fig = plt.figure()
ax = fig.add_subplot(111)

data_path = Path("../datasets/AEP_hourly.csv")
df = pl.read_csv(data_path)
y = df.select(pl.col('AEP_MW')).to_numpy().flatten()[:24]
x = np.array([i + 1 for i in range(len(y))])
ax.grid()
ax.set_xlabel('Total Energy Consumption [MWh]')
ax.set_ylabel('Time [hour]')
ax.plot(x, y, marker="o", color="#ff7f0e")

# %%
