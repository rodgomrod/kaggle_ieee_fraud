#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import gc
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.metrics import auc, precision_recall_curve
import datetime


from utils.schemas import *
from utils.functions import *


# In[2]:


data_folder = 'input'


# In[3]:


train = pd.read_csv(data_folder+'/train_ft_eng_1.zip', dtype = schema_ft_eng_1)
test = pd.read_csv(data_folder+'/test_ft_eng_1.zip', dtype = schema_ft_eng_1)


# In[17]:


# mini_train = train.sample(100000, random_state = 42).fillna(X.median())


# In[66]:


X_cols = [x for x in train.columns if x not in ['isFraud', 'TransactionDT', 'TransactionID']]


# In[109]:


X = train.sort_values('TransactionDT[X_cols].fillna(train.median())
y = train.sort_values('TransactionDT.isFraud


# In[110]:


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X.iloc[:int(X.shape[0]*0.8), :]
X_test = X.iloc[int(X.shape[0]*0.8):, :]
y_train = y[:int(X.shape[0]*0.8)]
y_test = y[int(X.shape[0]*0.8):]


# In[111]:


# rf = RandomForestClassifier(n_estimators=500, oob_score=True, n_jobs=-1, random_state=42)
# rf.fit(X_train, y_train)


# In[112]:


gc.collect()


# In[25]:


# perm = PermutationImportance(rf).fit(X_test, y_test)


# In[26]:


# eli5.show_weights(perm)


# In[27]:



# In[113]:


params = {
    'max_depth': -1,
    'num_leaves': 256,
    'metric': ['AUC'],
    'first_metric_only': True,
    'n_estimators': 20000,
    'learning_rate': 0.05,
    'colsample_bytree': 0.8,
    'objective': 'xentropy',
    'n_jobs': -1,
    'bagging_fraction': 0.8,
    'bagging_freq': 7,
    'lambda_l1': 0,
    'lambda_l2': 0,
    'bagging_seed': 42,
    'seed': 42,
    'feature_fraction_seed': 42,
    'drop_seed': 42,
    'data_random_seed': 42,
    'scale_pos_weight': 3,
}


# In[114]:


def custom_loss(y_pred, y_true):
    precision, recall, thresholds = precision_recall_curve(np.where(y_true >= 0.5, 1, 0), y_pred)
    AUC = auc(recall, precision)
    if AUC != AUC:
        AUC = 0
    return 'PR_AUC', AUC, True


# In[115]:


lgb_model = lgb.LGBMClassifier(**params)


# In[116]:


y.value_counts()


# In[118]:

print('Training lgb model')

lgb_model.fit(X_train,
                   y_train,
                   eval_set=[(X_test, y_test)],
                   verbose=50,
                   early_stopping_rounds=50,
#                    eval_metric=custom_loss
                  )


# In[86]:


lgb_imp = lgb_model.feature_importances_/lgb_model.feature_importances_.max()



# In[60]:


cat_ft_id = list()
n = 0
for c in X_cols:
    if c in cat_ft:
        cat_ft_id.append(n)
    n += 1


# In[70]:


params = {'depth':13,
          'iterations':20000,
          'eval_metric':'AUC',
          'random_seed':42,
          'logging_level':'Verbose',
          'allow_writing_files':False,
          'early_stopping_rounds':20,
          'learning_rate':0.07,
          'thread_count':8,
          'boosting_type':'Plain',
          'bootstrap_type':'Bernoulli',
          'rsm':0.3}


# In[71]:


model_cb = CatBoostClassifier(**params)


# In[72]:

print('Training CatBoost model')

model_cb.fit(X_train,
             y_train,
             cat_features=cat_ft_id,
             eval_set=(X_test, y_test),
             verbose=100
             )


# In[85]:


cb_imp = model_cb.feature_importances_/model_cb.feature_importances_.max()


# In[89]:

print('Training Rf model')
model_rf = RandomForestClassifier(n_estimators=500, oob_score=True, n_jobs=-1, random_state=42)
model_rf.fit(X_train, y_train)


# In[91]:


rf_imp = model_rf.feature_importances_/model_rf.feature_importances_.max()


# In[ ]:





# In[ ]:





# In[92]:


df_imp = pd.DataFrame({'feature': X_train.columns, 'importance': lgb_imp+cb_imp+rf_imp}).sort_values('importance', ascending = False)


# In[108]:


df_imp.head(10)



# In[105]:


today = datetime.date.today()
D = today.strftime('%Y%m%d


# In[107]:

print('Saving data')
df_imp.to_csv('docs/ft_importances_{}.csv'.format(D), index=None, header=True)


# In[ ]:




