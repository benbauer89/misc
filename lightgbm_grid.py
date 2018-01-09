import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid

train = pd.read_csv('train.csv.zip', compression='zip')

y = train['target']
train.drop(['target'], axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.33, random_state=42)

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)


### kind of constant stuff
params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'num_class': 9,
            'learning_rate': 0.01, # I prefer to set it very small and dont treat as a hyperparameter
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'verbose': 0
        }

# stuff to iterate over 

param_grid = {
              'num_leaves': [40, 60, 75], ## usw
              'feature_fraction': [0,7, 0.8, 0.9, 1] # usw
             }

# paramtergrid stuff (high numer of iterations + low learning rate is kind of fixed)
for param_grid in ParameterGrid(param_grid):
    param_grid.update(params) # update is doing the magic 
    print param_grid # during debugging
    gbm = lgb.train(param_grid, 
    				lgb_train, 
    				valid_sets=lgb_eval, # reference for early stopping
    				num_boost_round=2000,  # set this high - early stopping will help you out
    				early_stopping_rounds=10
    				)

### whats missing? ###

# return params for best score and the number of best iterations based on early_stopping_rounds