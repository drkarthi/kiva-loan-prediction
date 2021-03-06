'''
Reusing some code from https://github.com/drkarthi/driven-data-predicting-poverty/blob/master/predict.py
'''

import pandas as pd
import numpy as np
import sklearn.linear_model as sklm
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import sklearn.metrics as skm
import sklearn.ensemble as ske
import sklearn.preprocessing as skp
from sklearn.feature_selection import VarianceThreshold
from fancyimpute import SimpleFill, KNN, SoftImpute
# from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import matplotlib.pyplot as plt
import logging
import pickle
import os
import datetime
import pandas_profiling
import pdb

def lr_train(X_train, y_train, penalty, reg_const):
    model = sklm.LogisticRegression(penalty = penalty, C = reg_const)
    model.fit(X_train, y_train)
    return model

def sklearn_predict(model, X_test):
    pred = model.predict(X_test)
    probs = model.predict_proba(X_test)
    pos_probs = probs[:, 1]
    return pred, pos_probs

def drop_high_missing_features(df, threshold=0.7):
    # drop columns missing over threshold
    s_missing_fraction = pd.isnull(df).sum()/df.shape[0]
    high_missing_features = s_missing_fraction[s_missing_fraction > threshold].index
    # print("Number of features removed: ", len(high_missing_features))
    df_low_miss = df.drop(high_missing_features, axis = 1)
    return df_low_miss

def fill_missing_values(df):
    df = drop_high_missing_features(df)
    is_missing = pd.isnull(df).sum().sum()
    pdb.set_trace()
    if is_missing:
    	arr_complete = SimpleFill().fit_transform(df)
    	df = pd.DataFrame(arr_complete, columns = df.columns)
    # df.fillna(df.mean(), inplace = True)	
    return df

def error_analysis(X_train, labels, preds):
    # store the true positives, false negative, false positives and tru negatives for error analysis
    tp = X_train.iloc[np.where([label == 1 and pred == 1 for (label, pred) in zip(labels, preds)])]
    tp['label'] = 1
    tp['pred'] = 1

    fn = X_train.iloc[np.where([label == 1 and pred == 0 for (label, pred) in zip(labels, preds)])]
    fn['label'] = 1
    fn['pred'] = 0

    fp = X_train.iloc[np.where([label == 0 and pred == 1 for (label, pred) in zip(labels, preds)])]
    fp['label'] = 0
    fp['pred'] = 1

    tn = X_train.iloc[np.where([label == 0 and pred == 0 for (label, pred) in zip(labels, preds)])]
    tn['label'] = 0
    tn['pred'] = 0

    logging.info("Writing data for error analysis")

    tp.to_csv("error_analysis/true_positives.csv")
    fn.to_csv("error_analysis/false negatives.csv")
    fp.to_csv("error_analysis/false_positives.csv")
    tn.to_csv("error_analysis/true_negatives.csv")

    return tp, fn, fp, tn

# TODO: verify this is working correctly
def precision(preds, train_data):
    '''
    Self-defined eval metric for lightGBM cross-validation
    f(preds: array, train_data: Dataset) -> name: string, eval_result: float, is_higher_better: bool
    '''
    global FEVAL_COUNT
    labels = train_data.get_label()
    # pdb.set_trace()
    preds = 1. / (1. + np.exp(-preds)) # lgb return log odds ratio?
    tp = [(label == 1) and (pred >= 0.5) for label, pred in zip(labels, preds)]
    fp = [(label == 0) and (pred >= 0.5) for label, pred in zip(labels, preds)]
    precision = tp.count(True)/(tp.count(True)+fp.count(True))
    return 'precision', precision, True

# TODO: verify this is working correctly
def recall(preds, train_data):
    '''
    Self-defined eval metric for lightGBM cross-validation
    f(preds: array, train_data: Dataset) -> name: string, eval_result: float, is_higher_better: bool
    '''
    labels = train_data.get_label()
    # pdb.set_trace()
    preds = 1. / (1. + np.exp(-preds)) # lgb return log odds ratio?
    tp = [(label == 1) and (pred >= 0.5) for label, pred in zip(labels, preds)]
    fn = [(label == 1) and (pred < 0.5) for label, pred in zip(labels, preds)]
    recall = tp.count(True)/(tp.count(True)+fn.count(True))
    return 'recall', recall, True

def main():
    logging.basicConfig(level=logging.DEBUG)
    # read data
    df_loans = pd.read_csv("data/loans_no_descriptions.csv")
    print("Dataset read with {} loans".format(len(df_loans)))

    df_loans.info()
    # pdb.set_trace()

    # subset to funded and expired loans, leaving out currently fundraising loans and refunded (problematic) loans
    df_loans = df_loans[df_loans['STATUS'].isin(['funded', 'expired'])]
    df_loans['IS_UNFUNDED'] = np.where(df_loans['STATUS'] == "expired", 1, 0)

    # define X and y
    drop_columns = ['LOAN_ID', 'LOAN_NAME', 'FUNDED_AMOUNT', 'STATUS', 'COUNTRY_CODE', 
                    'DISBURSE_TIME', 'RAISED_TIME', 'NUM_LENDERS_TOTAL', 'NUM_JOURNAL_ENTRIES', 
                    'NUM_BULK_ENTRIES', 'BORROWER_NAMES', 'IS_UNFUNDED', 'PARTNER_ID']
    text_fields = ['LOAN_USE', 'TAGS']
    alternative_data_fields = ['IMAGE_ID', 'VIDEO_ID', 'TOWN_NAME']
    date_fields = ['POSTED_TIME', 'PLANNED_EXPIRATION_TIME']
    y = df_loans['IS_UNFUNDED'].astype('float')
    df_loans.drop(drop_columns + text_fields + alternative_data_fields, axis = 1, inplace = True)

    df_loans.info()
    # pdb.set_trace()

    # preprocess
    print("Preprocessing..")

    # handle multiple borrowers for BORROWER_GENDERS and BORROWER_PICTURED and add a NUM_BORROWERS field
    df_loans['NUM_BORROWERS'] = df_loans['BORROWER_GENDERS'].str.split(', ').str.len()
    df_loans['PERCENT_FEMALE'] = df_loans['BORROWER_GENDERS'].str.count("female") / df_loans['NUM_BORROWERS']
    df_loans['BORROWER_PICTURED'] = df_loans['BORROWER_PICTURED'].str.lower().str.split(', ').str.count("true") > 0

    # handle date fields and add a PLANNED_DURATION field
    df_loans['POSTED_TIME'] = pd.to_datetime(df_loans['POSTED_TIME'])
    df_loans['PLANNED_EXPIRATION_TIME'] = pd.to_datetime(df_loans['PLANNED_EXPIRATION_TIME'])
    df_loans['PLANNED_DURATION'] = df_loans['PLANNED_EXPIRATION_TIME'] - df_loans['POSTED_TIME']

    df_loans['POSTED_DAY'] = df_loans['POSTED_TIME'].dt.day
    df_loans['POSTED_MONTH'] = df_loans['POSTED_TIME'].dt.month
    df_loans['POSTED_YEAR'] = df_loans['POSTED_TIME'].dt.year
    df_loans['PLANNED_DURATION'] = df_loans['PLANNED_DURATION'].dt.days

    df_loans.drop(['BORROWER_GENDERS', 'POSTED_TIME', 'PLANNED_EXPIRATION_TIME'], axis = 1, inplace = True)

    logging.info("Converting categorical variables into dummy variables..")
    for col in ["ORIGINAL_LANGUAGE", "ACTIVITY_NAME", "SECTOR_NAME", "COUNTRY_NAME", "CURRENCY_POLICY", "CURRENCY", "REPAYMENT_INTERVAL", "DISTRIBUTION_MODEL"]:
        df_loans[col] = df_loans[col].astype('category')
    # df_loans = pd.get_dummies(df_loans, sparse = True)
    # df_loans = pd.get_dummies(df_loans)

    # pdb.set_trace()
    # df_loans = df_loans[['LOAN_AMOUNT', 'LENDER_TERM', 'NUM_BORROWERS', 'PERCENT_FEMALE', 'PLANNED_DURATION', 
                        # 'CURRENCY_POLICY', 'REPAYMENT_INTERVAL', 'DISTRIBUTION_MODEL']]
    df_loans = df_loans.drop(['ACTIVITY_NAME', 'ORIGINAL_LANGUAGE', 'BORROWER_PICTURED'], axis = 1)

    df_loans.info()
    # pdb.set_trace()

    X_train, X_test, y_train, y_test = train_test_split(df_loans, y, test_size=0.2, random_state=0)
    logging.info("Train test split successful")
    # X_train = fill_missing_values(X_train)
    # X_train.fillna(X_train.mean(), inplace = True)
    # X_test = fill_missing_values(X_test)
    # X_test.fillna(X_test.mean(), inplace = True)
    # logging.info("Filled missing values")

    df_loans.info()
    X_train.info()

    # del df_loans
    # logging.info("Deleted the original dataframe")

    train_data = lgb.Dataset(X_train, y_train, free_raw_data=True)
    test_data = lgb.Dataset(X_test, y_test, free_raw_data=True)

    logging.info("Created lgb datasets")
	
    # if os.path.isfile('model.pkl'):
    #     # TODO: if pickled model does not have the same columns as the training dataset, then train a new model
    #     logging.info("Trying to load existing model")
    #     try:
    #         model_lgbm = pickle.load(open('model.pkl', 'rb'))
    #     except FileNotFoundError:
    #         print("Could not load file")
    # else:
    logging.info("Training a new model since a model does not already exist or could not be loaded")
    # TODO: what do these parameters mean?
    param = {'num_leaves':31, 'num_trees':10, 'objective':'binary', 'metric':'binary_logloss'}
    num_round = 10
    
    # train native lightGBM model
    bst = lgb.train(param, train_data, num_round)
    bst.save_model('performant_model.txt')

    # train lightGBM model using sklean API
    model_lgbm = lgb.LGBMClassifier(num_leaves = param['num_leaves'],
                                    n_estimators = param['num_trees'],
                                    objective = param['objective'])
    model_lgbm.fit(X_train, y_train)
    feature_importance_scores = model_lgbm.booster_.feature_importance()
    feature_names = model_lgbm.booster_.feature_name()
    feature_importance = dict(zip(feature_names, feature_importance_scores))
    feature_importance = {k:v for k,v in sorted(feature_importance.items(), key = lambda item: item[1], reverse = True)}
    print(feature_importance)
    pickle.dump(model_lgbm, open('performant_model.pkl', 'wb'))
    cv_metrics = lgb.cv(param, 
                        train_data, 
                        num_round, 
                        nfold=5, 
                        feval=lambda preds, train_data: [precision(preds, train_data), 
                                                         recall(preds, train_data)])
    pdb.set_trace()

    print("Model trained and saved")

    y_pred = model_lgbm.predict(X_train)
    lgbm_recall = skm.recall_score(y_train, y_pred)
    lgbm_precision = skm.precision_score(y_train, y_pred)

    logging.info("Precision of sklearn wrapper model: " + str(lgbm_precision))
    logging.info("Recall of sklearn wrapper model: " + str(lgbm_recall))

    tp, fn, fp, tn = error_analysis(X_train, y_train, y_pred)

    pdb.set_trace()

    y_prob_native = bst.predict(X_train)
    y_pred_native = [1 if prob >= 0.5 else 0 for prob in y_prob_native]
    # TODO: convert continuous output to binary output befoire computing metrics
    bst_recall = skm.recall_score(y_train, y_pred_native)
    bst_precision = skm.precision_score(y_train, y_pred_native)

    logging.info("Precision of native model: " + str(bst_precision))
    logging.info("Recall of native model: " + str(bst_recall))

    # train a logistic regression model and predict
    # model = lr_train(X_train, y_train, penalty = 'l2', reg_const = 1)
    # pred, probs = sklearn_predict(model, X_train)

    pdb.set_trace()

    # accuracy_cv = cross_val_score(model_lgbm, X_train, y_train, scoring = "accuracy", cv = 5)
    # precision_cv = cross_val_score(model_lgbm, X_train, y_train, scoring = "precision", cv = 5)
    # recall_cv = cross_val_score(model_lgbm, X_train, y_train, scoring = "recall", cv = 5)
    # print("Cross validation accuracy: ", sum(accuracy_cv)/len(accuracy_cv))
    # print("Cross validation precision: ", sum(precision_cv)/len(precision_cv))
    # print("Cross validation recall: ", sum(recall_cv)/len(recall_cv))

t1 = datetime.datetime.now()
main()
t2 = datetime.datetime.now()
print("Time taken: ", t2 - t1)
pdb.set_trace()