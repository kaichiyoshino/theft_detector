#%%
import numpy as np 
import pandas as pd 
from pathlib import Path

DATA_PATH = Path("../data/ota_city")
files = [file.name for file in DATA_PATH.rglob('*') if file.is_file()]


def generate_data():
   #df作成
    energy_volumes = []
    for file_name in files:

        file_path = DATA_PATH / file_name

        _df = pd.read_csv(file_path)
        _sup = _df.iloc[:,0]
        _cons = _df.iloc[:,2]
        total_volume = _sup + _cons

        energy_volumes.append(total_volume)

    energy_volumes_df = pd.DataFrame(np.array(energy_volumes)).transpose()
    

    return energy_volumes_df


def reshape_data():
    df = generate_data()
    #dfの補完
    df['grid'] = -df.sum(axis=1)
    return df

# %%

