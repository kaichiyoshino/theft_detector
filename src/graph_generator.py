#%%
from data import DataReader
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt 

class GraphGenerator:
    def __init__(self, path):
        self.reader = DataReader(path)
        self.df = self.reader.get_data()

    def plot(self, house_number):
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1)
        x = np.array(self.df[house_number, 1: ])
        y = np.arange(0, len(x[0]))
        ax.plot(y, x[0], c='g', label='Electricity consumption', marker='o')
        ax.set_xlabel('Day')
        ax.set_ylabel('Electricity consumption[kWh]')
        ax.grid()

ins = GraphGenerator('../datasets/electricity_theft_data.csv')  
ins.plot(100)
