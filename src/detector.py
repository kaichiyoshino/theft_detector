
#%%
import polars as pl
from data import DataReader
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score
)
from sklearn.ensemble import GradientBoostingClassifier as GBC

dataset = DataReader(Path("../datasets/Electricity_Theft_Data.csv")).get_data()
dataset = dataset.with_row_index()
dataset.head()
# dataset.with_columns(pl.all().fill_null(pl.all().median()))
dataset = dataset.fill_null(0)


TARGET = 'CHK_STATE'
FEATURE = dataset.drop(['CHK_STATE']).clone()

# dataset['CHK_STATE'] = dataset['CHK_STATE'].cast('float32')

y = np.array(dataset[TARGET][1:])
X = np.array(FEATURE[1:, :])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 549)
clf = GBC(n_estimators=1000, learning_rate=0.5, max_depth=1,random_state=1).fit(X_train,y_train)
pred = clf.predict(X_test)
pred_prob = clf.predict_proba(X_test)[:,1]
acc_score = accuracy_score(y_test, pred)
auc_score = roc_auc_score(y_test, pred_prob)
print(acc_score)
print(auc_score)