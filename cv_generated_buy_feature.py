import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV,StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import KFold
from sklearn import preprocessing
from lightgbm import LGBMClassifier
from sklearn import metrics
from mlxtend.feature_selection import ColumnSelector
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# set data path
path = '/Users/charlie/Desktop/insurance/'

# import data
tr = pd.read_csv(path+ 'tr2.csv')
te = pd.read_csv(path +'te2.csv')

# features to train classifier
train_features = ['mean_Premium', 'std_Premium',
       'median_Premium', 'sum_Premium', 'min_Premium', 'max_Premium',
       #'Insured_Amount1', 'Insured_Amount2', 'Insured_Amount3',
       'mean_Insured_Amount1', 'std_Insured_Amount1',
       'median_Insured_Amount1', 'sum_Insured_Amount1',
       'min_Insured_Amount1', 'max_Insured_Amount1',
       'mean_Insured_Amount2', 'std_Insured_Amount2',
       'median_Insured_Amount2', 'sum_Insured_Amount2',
       'min_Insured_Amount2', 'max_Insured_Amount2',
       'mean_Insured_Amount3', 'std_Insured_Amount3',
       'median_Insured_Amount3', 'sum_Insured_Amount3',
       'min_Insured_Amount3', 'max_Insured_Amount3', 'qpt',
       'Replacement_cost_of_insured_vehicle',
       'Engine_Displacement_(Cubic_Centimeter)', 'age',
       'Multiple_Products_with_TmNewa_(Yes_or_No?)', 'lia_class',
       'plia_acc', 'pdmg_acc', 'mean_Deductible', 'std_Deductible',
       'median_Deductible', 'sum_Deductible', 'min_Deductible',
       'max_Deductible', 'Coverage_Group0', 'Coverage_Group1',
       'Coverage_Group2', 'Coverage_Group0_YN', 'Coverage_Group1_YN',
       'Coverage_Group2_YN', 'fassured1', 'fassured2', 'fassured3',
       'fassured6', 'fmarriage0', 'fmarriage1', 'fmarriage2', 'fsex0',
       'fsex1', 'fsex2', 'Manafactured_Year_and_Month', 'Cancellation',
       'lia', 'acc', 'Policy_showup_Numbers', 'Claim_Number', 'mean_Loss',
       'std_Loss', 'median_Loss', 'sum_Loss', 'min_Loss', 'max_Loss',
       'mean_Expenses', 'std_Expenses', 'median_Expenses', 'sum_Expenses',
       'min_Expenses', 'max_Expenses', 'mean_total_paid', 'std_total_paid',
       'median_total_paid', 'sum_total_paid', 'min_total_paid',
       'max_total_paid', 'mean_claimants', 'std_claimants',
       'median_claimants', 'sum_claimants', 'min_claimants',
       'max_claimants', 'Salvage_or_Subrogation?',
       'mean_Salvage_or_Subrogation', 'mean_claim_Deductible',
       'sum_claim_Deductible', 'mean_Fault', 'sum_Fault', 'Imported_cars', 'zip_cluster_2',
                  'iply_haser1', 'iply_haser2',
       'iply_haser3', 'iply_haser4', 'iply_haser5',
                  '00I', '01A', '01J',
       '02K', '03L', '04M', '05E', '05N', '06F', '07P', '08H', '09@',
       '09I', '10A', '12L', '14E', '14N', '15F', '15O', '16G', '16P',
       '18@', '18I', '20B', '20K', '25G', '26H', '27I', '29B', '29K',
       '32N', '33F', '33O', '34P', '35H', '36I', '37J', '40M', '41E',
       '41N', '42F', '45@', '46A', '47B', '51O', '55J', '56B', '56K',
       '57C', '57L', '65K', '66C', '66L', '67D', '68E', '68N', '70G',
       '70P', '71H', '72@',
        'Model1',
                  
                  'Channel',
'mean_premium_age',
       'sum_premium_age', 'std_premium_age', 'median_premium_age',
       'min_premium_age', 'max_premium_age', 'mean_premium_d_age',
       'sum_premium_d_age', 'Manafactured_age', 'Manafactured_d_age',
                  #
                  'mean_coverage_premium_0',
       'mean_coverage_premium_1', 'mean_coverage_premium_2',
       'sum_coverage_premium_0', 'sum_coverage_premium_1',
       'sum_coverage_premium_2',
                  #
                  'mean_coverage_insured_amount_0',
       'mean_insured_amount_1', 'mean_insured_amount_2',
       'sum_insured_amount_0', 'sum_insured_amount_1',
       'sum_insured_amount_2',
                  #
                  "neg_repl_car_age",
                  #
                  'Model',
                  #
                  'Age_d_age','Engine_repl','Engine_Age','mean_Engine_Premium', 'sum_Engine_Premium',
                  #
                  'd_age','d_age_class','Age_add_d_age','qpt_engine', 'qpt_repl',
       'qpt_car_age',
                  #
'coverage_pca_0',
       'coverage_pca_1', 'coverage_pca_2', 'coverage_pca_3',
       'coverage_pca_4', 'coverage_pca_5', 'coverage_pca_6',
                  'sum_plia_acc_Premium','sum_lia_class_Premium',
                  'mean_car_premium', 'sum_car_premium','engine_plia', 'engine_lia_class',
                  'mean_iply_premium',
       'sum_iply_premium','is_motor_car_truck','fequipment1',
       'fequipment2', 'fequipment3', 'fequipment4', 'fequipment5',
       'fequipment6', 'fequipment9', 'zero_Insured_Amount','mean_At_Fault_Premium', 'sum_At_Fault_Premium',
                  'no_future','mean_engine_lia_class_premium', 'sum_engine_lia_class_premium',
       'mean_engine_plia_premium', 'sum_engine_plia_premium',
       'mean_engine_pdmg_premium', 'sum_engine_pdmg_premium',
                  'Claim_Number_Premium', 'Claim_Number_Insured_Amount1',
       'Claim_Number_Insured_Amount2', 'Claim_Number_Insured_Amount3',
                  'sum_claimants_Premium', 'sum_claimants_Insured_Amount1',
       'sum_claimants_Insured_Amount2', 'sum_claimants_Insured_Amount3',
                  'legal_man','sum_claimants_repl', 'sum_claimants_repl_premium',
       'sum_claimants_qpt', 'sum_claimants_qpt_premium',
       'sum_claimants_engine', 'sum_claimants_engine_premium',
       'Claim_Number_repl', 'Claim_Number_repl_premium',
       'Claim_Number_qpt', 'Claim_Number_qpt_premium',
                  #
                  'channel_premium','Nature_of_the_claim_1', 'Nature_of_the_claim_2',
                  'driver_relation_1', 'driver_relation_2', 'driver_relation_3',
       'driver_relation_4', 'driver_relation_5', 'driver_relation_6',
       'driver_relation_7','sum_Premium_div_age','channel_cluster_8',
                  #
                  'model1_cluster_6_y',
                  #
    'zip_cluster_6',
       'iply_cluster_6'
                 ]

# train and test data
train_X = tr[train_features]
train_y = np.where(tr['Next_Premium'] <= 500 , 0 , 1)
test_X = te[train_features]

# CV way to generate prob of buy
oof_train = np.zeros((len(tr),))
oof_test_skf = np.empty((5, len(te)))
oof_test = np.zeros((len(te),))
model_1 = LGBMClassifier(
    boosting_type='gbdt', num_leaves=85, max_depth= 15, learning_rate=0.003, 
    n_estimators= 3677, subsample_for_bin=400000, objective="binary",
    min_split_gain=0.0, min_child_weight=0.01, min_child_samples=50, subsample=0.8, 
    subsample_freq=1, colsample_bytree=0.7, reg_alpha=5.0, reg_lambda=0.0,
    silent=True
)

kf = KFold(n_splits=5)
for n_fold, (train_index, test_index) in enumerate(kf.split(train_X)):
    print n_fold
    X_train, X_test = train_X.iloc[train_index], train_X.iloc[test_index]
    y_train, y_test = train_y[train_index], train_y[test_index]
    model_1.fit(X_train, y_train)
    #prediction = model_1.predict_proba(X_test)
    #train_score.append(prediction[:,1])
    oof_train[test_index] = model_1.predict_proba(X_test)[:,1]
    oof_test_skf[n_fold, :] = model_1.predict_proba(test_X)[:,1]

oof_test[:] = oof_test_skf.mean(axis=0)
te['buy'] = oof_test
tr['buy'] = oof_train
