# packages

# standard
import numpy as np
import pandas as pd
import time

# plots
import matplotlib.pyplot as plt
import seaborn as sns

# machine learning tools
import h2o
from h2o.estimators import H2OGeneralizedLinearEstimator, H2ORandomForestEstimator, H2OGradientBoostingEstimator
from sklearn.feature_selection import mutual_info_classif # mutual information
# load data + first glance
t1 = time.time()
df_train = pd.read_csv('../input/tabular-playground-series-may-2022/train.csv')
df_test = pd.read_csv('../input/tabular-playground-series-may-2022/test.csv')
df_sub = pd.read_csv('../input/tabular-playground-series-may-2022/sample_submission.csv')
t2 = time.time()
print('Elapsed time:', np.round(t2-t1,4))
# first glance (training data)
df_train.head()
# dimensions
print('Train Set:', df_train.shape)
print('Test Set :', df_test.shape)
# structure / missing values
df_train.info(verbose=True, show_counts=True)
# same for test set
df_test.info(verbose=True, show_counts=True)
# f27 is special...
df_train.f_27.value_counts()
# aux function
def extract_char(i_string, i_k):
    return i_string[i_k]
# decompose f_27 in character features
for k in range(10):
    feature_name = 'f_27_' + str(k)
    print(feature_name)
    df_train[feature_name] = list(map(lambda x: extract_char(x,k), df_train.f_27))
    df_test[feature_name] = list(map(lambda x: extract_char(x,k), df_test.f_27))
df_train['unique_chars'] = df_train.f_27.apply(lambda s: len(set(s)))
df_test['unique_chars'] = df_test.f_27.apply(lambda s: len(set(s)))
features_num = ['f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05',
                'f_06', 'f_07', 'f_08', 'f_09', 'f_10', 'f_11',
                'f_12', 'f_13', 'f_14', 'f_15', 'f_16', 'f_17',
                'f_18', 'f_19', 'f_20', 'f_21', 'f_22', 'f_23',
                'f_24', 'f_25', 'f_26', 'f_28', 'f_29', 'f_30']
features_char = ['f_27_0', 'f_27_1', 'f_27_2', 'f_27_3', 'f_27_4',
                 'f_27_5', 'f_27_6', 'f_27_7', 'f_27_8', 'f_27_9',
                 'unique_chars']
# numerical features
df_train[features_num].describe()
df_train[features_char].describe(include='all')
# distribution of each character feature
for f in features_char:
    plt.figure(figsize=(10,3))
    df_train[f].value_counts().sort_index().plot(kind='bar')
    plt.title(f + ' - Train')
    plt.grid()
    plt.show()

    corr_pearson = df_train[features_num + ['unique_chars']].corr(method='pearson')

    plt.figure(figsize=(9, 8))
    sns.heatmap(corr_pearson, annot=False, cmap='RdYlGn', vmin=-1, vmax=+1)
    plt.title('Pearson Correlation')
    plt.show()
# target - basic stats
print(df_train.target.value_counts())
df_train.target.value_counts().plot(kind='bar')
plt.title('Target')
plt.grid()
plt.show()
# mutual information
# (see https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html)
features_num_plus = features_num + ['unique_chars']

t1 = time.time()
x = df_train[features_num_plus]
y = df_train.target
mi = mutual_info_classif(x, y)
t2 = time.time()
print('Elapsed time [s]: ', np.round(t2-t1,2))
# plot mutual informations
plt.figure(figsize=(16,4))
plt.bar(height=mi,x=features_num_plus)
plt.xticks(rotation=90)
plt.title('Mutual Information - Target vs Feature [numerical]')
plt.grid()
plt.show()
# for categorical features convert to int first
features_char_minus = features_char.copy()
features_char_minus.remove('unique_chars')
df_train_char_num = df_train[features_char_minus].apply(lambda col : col.astype('category').cat.codes)

t1 = time.time()
x = df_train_char_num[features_char_minus]
y = df_train.target
mi_char = mutual_info_classif(x, y)
t2 = time.time()
print('Elapsed time [s]: ', np.round(t2-t1,2))
# plot mutual informations
plt.figure(figsize=(16,4))
plt.bar(height=mi_char,x=features_char_minus)
plt.xticks(rotation=90)
plt.title('Mutual Information - Target vs Feature [char]')
plt.grid()
plt.show()
# plot each numerical feature split by target=0/1
for f in features_num:
    plt.figure(figsize=(10,3))
    sns.violinplot(data=df_train, y='target', x=f, orient='h')
    plt.title(f + ' - Train')
    plt.grid()
    plt.show()
for f in features_char:
    ctab = pd.crosstab(df_train[f], df_train.target)
    ctab_norm = ctab.transpose() / (ctab.sum(axis=1))
    plt.figure(figsize=(16,3))
    sns.heatmap(ctab_norm, annot=True,
                cmap='Blues',
                linecolor='black',
                linewidths=0.1)
    plt.title(f)
    plt.show()
# select predictors
predictors = features_num + features_char
print('Number of predictors: ', len(predictors))
print(predictors)
# start H2O
h2o.init(max_mem_size='12G', nthreads=4) # Use maximum of 12 GB RAM and 4 cores
# upload train/test set to H2O environment
t1 = time.time()
train_hex = h2o.H2OFrame(df_train)
test_hex = h2o.H2OFrame(df_test)
t2 = time.time()
print('Elapsed time [s]: ', np.round(t2-t1,2))

# force categorical target
train_hex['target'] = train_hex['target'].asfactor()
#  fit Gradient Boosting model
n_cv = 5

fit_GBM = H2OGradientBoostingEstimator(ntrees=750,
                                       max_depth=15,
                                       min_rows=10,
                                       learn_rate=0.1, # default: 0.1
                                       sample_rate=1,
                                       col_sample_rate=0.5,
                                       nfolds=n_cv,
                                       score_each_iteration=True,
                                       stopping_metric='auc',
                                       stopping_rounds=5,
                                       stopping_tolerance=0.00001,
                                       seed=999)
# train model
t1 = time.time()
fit_GBM.train(x=predictors,
              y='target',
              training_frame=train_hex)
t2 = time.time()
print('Elapsed time [s]: ', np.round(t2-t1,2))
# show cross validation metrics
fit_GBM.cross_validation_metrics_summary()
# show scoring history - training vs cross validations
for i in range(n_cv):
    cv_model_temp = fit_GBM.cross_validation_models()[i]
    df_cv_score_history = cv_model_temp.score_history()
    my_title = 'CV ' + str(1+i) + ' - Scoring History [AUC]'
    plt.scatter(df_cv_score_history.number_of_trees,
                y=df_cv_score_history.training_auc,
                c='blue', label='training')
    plt.scatter(df_cv_score_history.number_of_trees,
                y=df_cv_score_history.validation_auc,
                c='darkorange', label='validation')
    plt.title(my_title)
    plt.xlabel('Number of Trees')
    plt.ylabel('AUC')
    plt.ylim(0.7,1.0)
    plt.legend()
    plt.grid()
    plt.show()
# variable importance
fit_GBM.varimp_plot(len(predictors))
plt.show()
# training performance
perf_train = fit_GBM.model_performance(train=True)
perf_train.plot()
plt.show()
# cross validation performance
perf_cv = fit_GBM.model_performance(xval=True)
perf_cv.plot()
plt.show()
# predict on train set (extract probabilities only)
pred_train_GBM = fit_GBM.predict(train_hex)['p1']
pred_train_GBM = pred_train_GBM.as_data_frame().p1

# plot train set predictions (probabilities)
plt.figure(figsize=(8,4))
plt.hist(pred_train_GBM, bins=100)
plt.title('Predictions on Train Set - GBM')
plt.grid()
plt.show()
# check calibration
n_actual = sum(df_train.target)
n_pred_GBM = sum(pred_train_GBM)

print('Actual Frequency    :', n_actual)
print('Predicted Frequency :', n_pred_GBM)
print('Calibration Ratio   :', n_pred_GBM / n_actual)
# predict on test set (extract probabilities only)
pred_test_GBM = fit_GBM.predict(test_hex)['p1']
pred_test_GBM = pred_test_GBM.as_data_frame().p1
# plot test set predictions (probabilities)
plt.figure(figsize=(8,4))
plt.hist(pred_test_GBM, bins=100)
plt.title('Predictions on Test Set - GBM')
plt.grid()
plt.show()
# GBM submission
df_sub_GBM = df_sub.copy()
df_sub_GBM.target = pred_test_GBM
display(df_sub_GBM.head())
# save to file
df_sub_GBM.to_csv('submission_GBM.csv', index=False)