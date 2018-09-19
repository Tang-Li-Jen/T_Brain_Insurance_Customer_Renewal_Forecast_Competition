import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV,StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import KFold
from sklearn import preprocessing
from lightgbm import LGBMRegressor
from lightgbm import LGBMClassifier
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier


# load date

tr = pd.read_csv('tr11.csv')
te = pd.read_csv('te11.csv')


# filter data

df_duplicate = pd.read_csv('duplicate_policy.csv', names = ["Policy_Number"])
df_wrong_dbirth = pd.read_csv('wrong_dbirth.csv', names = ["Policy_Number"])
df_del = pd.concat([df_duplicate, df_wrong_dbirth])

tr = pd.merge(tr, df_del, on='Policy_Number',how="outer",indicator=True)
tr = tr[tr['_merge'] == 'left_only']

tr = tr[tr.sum_Premium < 130000]


# feature importance

classifier_features = [ col for col in tr.columns.values if col not in [
     "Policy_Number", "Next_Premium",'_merge','premium_div_insured_amt'
]]

train_X = tr[classifier_features]
train_y = np.where(tr.Next_Premium <= 500, 0, 1)
#切割訓練、驗證資料
Xtr, Xv, ytr, yv = train_test_split(train_X, train_y, test_size=0.3, random_state=927)

model_1 = LGBMClassifier(
    boosting_type='gbdt', num_leaves=85, max_depth= 15, learning_rate=0.001, 
    n_estimators= 9840, subsample_for_bin=400000, objective="binary",
    min_split_gain=0.0, min_child_weight=0.01, min_child_samples=50, subsample=0.8, 
    subsample_freq=1, colsample_bytree=0.7, reg_alpha=5.0, reg_lambda=0.0,
    silent=True
)
model_1.fit(train_X, train_y)

classifier_feature_importance = pd.DataFrame({'feature_name':model_1.booster_.feature_name(),'importance':model_1.booster_.feature_importance()} )
classifier_feature_importance = classifier_feature_importance.sort_values(by = 'importance', ascending= False)


# feature number selection

classifier_all =  [col for col in classifier_feature_importance.feature_name ]

tmp = classifier_feature_importance[classifier_feature_importance.importance != 0]
classifier_0 = [col for col in tmp.feature_name ]

tmp = classifier_feature_importance[classifier_feature_importance.importance >= 10]
classifier_50 = [col for col in tmp.feature_name ]

tmp = classifier_feature_importance[classifier_feature_importance.importance >= 300]
classifier_100 = [col for col in tmp.feature_name ]

tmp = classifier_feature_importance[classifier_feature_importance.importance >= 800]
classifier_500 = [col for col in tmp.feature_name ]

tmp = classifier_feature_importance[classifier_feature_importance.importance >= 1500]
classifier_1000 = [col for col in tmp.feature_name ]


# stack

# classifier_all

ls = [classifier_all, classifier_0, classifier_50, classifier_100, classifier_500, classifier_1000]
best_n_ls = []

j = 0
for i in ls:
    train_X = tr[i]
    train_y = np.where(tr.Next_Premium <= 500, 0, 1)
    #切割訓練、驗證資料
    Xtr, Xv, ytr, yv = train_test_split(train_X, train_y, test_size=0.3, random_state=927)
    model_1 = LGBMClassifier(
        boosting_type='gbdt', num_leaves=95, max_depth= 18, learning_rate=0.001, 
        n_estimators= 30000, subsample_for_bin=400000, objective="binary",
        min_split_gain=0.0, min_child_weight=0.01, min_child_samples=50, subsample=0.8, 
        subsample_freq=1, colsample_bytree=0.7, reg_alpha=10.0, reg_lambda=0.0,
        silent=True
    )
    model_1.fit(Xtr,ytr, eval_set=[(Xtr, ytr), (Xv, yv)], eval_metric='auc',
               early_stopping_rounds=500, verbose=1000)
    best_n_ls.append(int(model_1.best_iteration_))
    j += 1

tr['buy'] = np.where(tr.Next_Premium <= 500, 0, 1)

ntrain = tr.shape[0]
ntest = te.shape[0]
SEED = 927 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
#kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)
kf = KFold(n_splits=NFOLDS)

def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(tr)):
        print(i)
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.fit(x_tr, y_tr)

        oof_train[test_index] = clf.predict_proba(x_te)[:,1]
        oof_test_skf[i, :] = clf.predict_proba(x_test)[:,1]

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

# best_n_ls = [9895, 10089, 10626, 8732, 10000, 10568]

i = 0
for features in ls:
    model = LGBMClassifier(
    boosting_type='gbdt', num_leaves=95, max_depth= 18, learning_rate=0.001, 
    n_estimators= best_n_ls[i], subsample_for_bin=400000, objective="binary",
    min_split_gain=0.0, min_child_weight=0.01, min_child_samples=50, subsample=0.8, 
    subsample_freq=1, colsample_bytree=0.7, reg_alpha=10.0, reg_lambda=0.0,
    silent=True
    )
    model_oof_train, model_oof_test = get_oof(model, 
                                tr[features].values, tr["buy"].values, te[features].values)
    np.savetxt("meta_features_train_0912_"+str(i+1)+'_2.csv', model_oof_train, delimiter=',')
    np.savetxt("meta_features_test_0912_"+str(i+1)+'_2.csv', model_oof_test, delimiter=',')
    i += 1

# stack classifier

# test
meta_features_test1 = pd.read_csv("meta_features_test_0912_1.csv", names = ['meta_features_1'])
meta_features_test2 = pd.read_csv("meta_features_test_0912_2.csv", names = ['meta_features_2'])
meta_features_test3 = pd.read_csv("meta_features_test_0912_3.csv", names = ['meta_features_3'])
meta_features_test4 = pd.read_csv("meta_features_test_0912_4.csv", names = ['meta_features_4'])
meta_features_test5 = pd.read_csv("meta_features_test_0912_5.csv", names = ['meta_features_5'])
meta_features_test6 = pd.read_csv("meta_features_test_0912_6.csv", names = ['meta_features_6'])


meta_features_test1_2 = pd.read_csv("meta_features_test_0912_1_2.csv", names = ['meta_features_1_2'])
meta_features_test2_2 = pd.read_csv("meta_features_test_0912_2_2.csv", names = ['meta_features_2_2'])
meta_features_test3_2 = pd.read_csv("meta_features_test_0912_3_2.csv", names = ['meta_features_3_2'])
meta_features_test4_2 = pd.read_csv("meta_features_test_0912_4_2.csv", names = ['meta_features_4_2'])
meta_features_test5_2 = pd.read_csv("meta_features_test_0912_5_2.csv", names = ['meta_features_5_2'])
meta_features_test6_2 = pd.read_csv("meta_features_test_0912_6_2.csv", names = ['meta_features_6_2'])


meta_features_test1_3 = pd.read_csv("meta_features_test_0912_1_3.csv", names = ['meta_features_1_3'])
meta_features_test2_3 = pd.read_csv("meta_features_test_0912_2_3.csv", names = ['meta_features_2_3'])
meta_features_test3_3 = pd.read_csv("meta_features_test_0912_3_3.csv", names = ['meta_features_3_3'])
meta_features_test4_3 = pd.read_csv("meta_features_test_0912_4_3.csv", names = ['meta_features_4_3'])
meta_features_test5_3 = pd.read_csv("meta_features_test_0912_5_3.csv", names = ['meta_features_5_3'])
meta_features_test6_3 = pd.read_csv("meta_features_test_0912_6_3.csv", names = ['meta_features_6_3'])


# train
meta_features_train1 = pd.read_csv("meta_features_train_0912_1.csv", names = ['meta_features_1'])
meta_features_train2 = pd.read_csv("meta_features_train_0912_2.csv", names = ['meta_features_2'])
meta_features_train3 = pd.read_csv("meta_features_train_0912_3.csv", names = ['meta_features_3'])
meta_features_train4 = pd.read_csv("meta_features_train_0912_4.csv", names = ['meta_features_4'])
meta_features_train5 = pd.read_csv("meta_features_train_0912_5.csv", names = ['meta_features_5'])
meta_features_train6 = pd.read_csv("meta_features_train_0912_6.csv", names = ['meta_features_6'])


meta_features_train1_2 = pd.read_csv("meta_features_train_0912_1_2.csv", names = ['meta_features_1_2'])
meta_features_train2_2 = pd.read_csv("meta_features_train_0912_2_2.csv", names = ['meta_features_2_2'])
meta_features_train3_2 = pd.read_csv("meta_features_train_0912_3_2.csv", names = ['meta_features_3_2'])
meta_features_train4_2 = pd.read_csv("meta_features_train_0912_4_2.csv", names = ['meta_features_4_2'])
meta_features_train5_2 = pd.read_csv("meta_features_train_0912_5_2.csv", names = ['meta_features_5_2'])
meta_features_train6_2 = pd.read_csv("meta_features_train_0912_6_2.csv", names = ['meta_features_6_2'])


meta_features_train1_3 = pd.read_csv("meta_features_train_0912_1_3.csv", names = ['meta_features_1_3'])
meta_features_train2_3 = pd.read_csv("meta_features_train_0912_2_3.csv", names = ['meta_features_2_3'])
meta_features_train3_3 = pd.read_csv("meta_features_train_0912_3_3.csv", names = ['meta_features_3_3'])
meta_features_train4_3 = pd.read_csv("meta_features_train_0912_4_3.csv", names = ['meta_features_4_3'])
meta_features_train5_3 = pd.read_csv("meta_features_train_0912_5_3.csv", names = ['meta_features_5_3'])
meta_features_train6_3 = pd.read_csv("meta_features_train_0912_6_3.csv", names = ['meta_features_6_3'])

test_meta_features = pd.concat([meta_features_test1,
                                meta_features_test2,
                                meta_features_test3,
                                meta_features_test4,
                                meta_features_test5,
                                meta_features_test6,
                                meta_features_test1_2,
                                meta_features_test2_2,
                                meta_features_test3_2,
                                meta_features_test4_2,
                                meta_features_test5_2,
                                meta_features_test6_2,
                                meta_features_test1_3,
                                meta_features_test2_3,
                                meta_features_test3_3,
                                meta_features_test4_3,
                                meta_features_test5_3,
                                meta_features_test6_3], axis=1)

train_meta_features = pd.concat([meta_features_train1,
                                 meta_features_train2,
                                 meta_features_train3,
                                 meta_features_train4,
                                 meta_features_train5,
                                 meta_features_train6,
                                 meta_features_train1_2,
                                 meta_features_train2_2,
                                 meta_features_train3_2,
                                 meta_features_train4_2,
                                 meta_features_train5_2,
                                 meta_features_train6_2,
                                 meta_features_train1_3,
                                 meta_features_train2_3,
                                 meta_features_train3_3,
                                 meta_features_train4_3,
                                 meta_features_train5_3,
                                 meta_features_train6_3], axis=1)

tr = tr.reset_index(drop=True)
te = te.reset_index(drop=True)

tr = pd.concat([tr, train_meta_features], axis=1)
te = pd.concat([te, test_meta_features], axis=1)

# Fine tune meta classifier
# all features

classifier_features = [ col for col in tr.columns.values if col not in [
     "Policy_Number", "Next_Premium",'_merge','premium_div_insured_amt'
]]

train_X = tr[classifier_features]
train_y = np.where(tr.Next_Premium <= 500, 0, 1)

#切割訓練、驗證資料
Xtr, Xv, ytr, yv = train_test_split(train_X, train_y, test_size=0.3, random_state=927)

# 3 stack classifier model

model_1 = LGBMClassifier(
    boosting_type='gbdt', num_leaves=85, max_depth= 15, learning_rate=0.001, 
    n_estimators= 30000, subsample_for_bin=400000, objective="binary",
    min_split_gain=0.0, min_child_weight=0.01, min_child_samples=50, subsample=0.8, 
    subsample_freq=1, colsample_bytree=0.7, reg_alpha=5.0, reg_lambda=0.0,
    silent=True
)

model_1.fit(Xtr,ytr, eval_set=[(Xtr, ytr), (Xv, yv)], eval_metric='auc',
           early_stopping_rounds=500, verbose=1000)

# 0913
model_1 = LGBMClassifier(
    boosting_type='gbdt', num_leaves=85, max_depth= 15, learning_rate=0.001, 
    n_estimators= int(model_1.best_iteration_), subsample_for_bin=400000, objective="binary",
    min_split_gain=0.0, min_child_weight=0.01, min_child_samples=50, subsample=0.8, 
    subsample_freq=1, colsample_bytree=0.7, reg_alpha=5.0, reg_lambda=0.0,
    silent=True
)
model_1.fit(train_X, train_y)

tr["tr_buy"] = model_1.predict(tr[classifier_features])
te["te_buy"] = model_1.predict(te[classifier_features])

tr_buy_0913 = tr.tr_buy
te_buy_0913 = te.te_buy

tr_buy_0913.to_csv('tr_buy_0913_1.csv')
te_buy_0913.to_csv('te_buy_0913_1.csv')


# Tune Regressor
regressor_features = [ col for col in tr_tmp.columns.values if col not in [
    "Policy_Number", "Next_Premium",'tr_buy','te_buy','buy_prob','_merge','premium_div_insured_amt'
]]

train_X = tr_tmp[regressor_features]
train_y = tr_tmp['Next_Premium']

#切割訓練、驗證資料
Xtr, Xv, ytr, yv = train_test_split(train_X, train_y, test_size=0.3, random_state=927)

# 0913 3stack

model_reg = LGBMRegressor(
boosting_type= 'gbdt',
colsample_bytree = 0.7,
learning_rate = 0.001,
max_depth = 18,
min_child_samples = 50,
min_child_weight = 0.01,
min_split_gain = 1.0,
n_estimators = 30000,
num_leaves = 85,
objective = "regression_l1",
reg_alpha = 5,
reg_lambda = 0,
silent = True,
subsample = 0.8,
subsample_for_bin = 400000,
subsample_freq =  1
)
model_reg.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xv, yv)], early_stopping_rounds=100,
              eval_metric="'l1'",verbose=500)

# official regressor

regressor_features = [ col for col in tr.columns.values if col not in [
    "Policy_Number", "Next_Premium",'tr_buy','te_buy','buy_prob','_merge','premium_div_insured_amt'
]]
train_X = tr_tmp[regressor_features]
train_y = tr_tmp['Next_Premium']

model = LGBMRegressor(
boosting_type= 'gbdt',
colsample_bytree = 0.7,
learning_rate = 0.001,
max_depth = 18,
min_child_samples = 50,
min_child_weight = 0.01,
min_split_gain = 1.0,
n_estimators = 25628,
num_leaves = 85,
objective = "regression_l1",
reg_alpha = 5,
reg_lambda = 0,
silent = True,
subsample = 0.8,
subsample_for_bin = 400000,
subsample_freq =  1
)

model.fit(train_X, train_y)

submit_0 = pd.DataFrame({"Policy_Number":te[te.te_buy == 0]['Policy_Number'], "Next_Premium": 0})
submit_1 = pd.DataFrame({"Policy_Number":te[te.te_buy == 1]["Policy_Number"], 
                         "Next_Premium":model_reg.predict(te[te.te_buy == 1][regressor_features])})

submit = pd.concat([submit_0, submit_1])

submit.loc[submit.Next_Premium < 0, 'Next_Premium'] = 0
submit.to_csv('submit_0913_2.csv', index=False)


# method2 average

f1 = pd.read_csv("submit_0914_1.csv")
f2 = pd.read_csv("submit_0913_1.csv")
f3 = pd.read_csv("submit_0912_2.csv")
f4 = pd.read_csv("submit_0910_1.csv")
f5 = pd.read_csv("submit_0909_1.csv")
f6 = pd.read_csv("submit_0908_2.csv")
f7 = pd.read_csv("submit_0907_1.csv")
f8 = pd.read_csv("submit_0906_3.csv")
f9 = pd.read_csv("submit_0906_2.csv")
f10 = pd.read_csv("submit_0905_1_1.csv")
f11 = pd.read_csv("submit_0905_1.csv")

test = pd.merge(f1, f2, how='inner', on="Policy_Number")
test = pd.merge(test, f3, how='inner', on="Policy_Number")
test = pd.merge(test, f4, how='inner', on="Policy_Number")
test = pd.merge(test, f5, how='inner', on="Policy_Number")
test = pd.merge(test, f6, how='inner', on="Policy_Number")
test = pd.merge(test, f7, how='inner', on="Policy_Number")
test = pd.merge(test, f8, how='inner', on="Policy_Number")
test = pd.merge(test, f9, how='inner', on="Policy_Number")
test = pd.merge(test, f10, how='inner', on="Policy_Number")
test = pd.merge(test, f11, how='inner', on="Policy_Number")
test = pd.merge(test, f1, how='inner', on="Policy_Number")

test_np = test.values
test_np1 = test_np[:,2:]

ans = np.zeros(shape=(test_np1.shape[0],2))
ans_pd = pd.DataFrame({'Next_Premium':ans[:,0],'Policy_Number':ans[:,1]})
ans_pd.iloc[:,1:2] = test.iloc[:,1:2]

for i in range(test_np1.shape[0]):
    if list(test_np1[i,:]).count(0) > 6:
        continue
    else:
        sum1 = 0
        count1 = 0
        for j in range(test_np1[i,:].shape[0]):
            if test_np1[i,j] != 0:
                sum1 += test_np1[i,j]
                count1 += 1
        mean1 = sum1 / count1
        ans_pd.iloc[i,0:1] = mean1

ans_pd.to_csv('submit_0914_2.csv', index=False)
