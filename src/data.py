#%%
import polars as pl 
import numpy as np 
import matplotlib.pyplot as plt 


class DataReader:
    def __init__(self, path):
        self.path = path
        self.df = pl.read_csv(self.path, index_col=0)
    
    def get_data(self):
        return self.df


