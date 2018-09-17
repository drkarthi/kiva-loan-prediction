'''
Reusing some code from https://github.com/drkarthi/driven-data-predicting-poverty/blob/master/predict.py
'''

import pandas as pd
import numpy as np
import sklearn.linear_model as sklm
from sklearn.model_selection import train_test_split, cross_val_score
import sklearn.metrics as skm
import sklearn.ensemble as ske
import sklearn.preprocessing as skp
from sklearn.feature_selection import VarianceThreshold
from fancyimpute import SimpleFill, KNN, MICE, SoftImpute
# from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
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
	if is_missing:
		arr_complete = SimpleFill().complete(df)
		df = pd.DataFrame(arr_complete, columns = df.columns)	
	return df

def main():
	# read data
	df_loans = pd.read_csv("data/loans_no_descriptions.csv")
	print("Dataset read with {} loans".format(len(df_loans)))
	pdb.set_trace()

	# subset to funded and expired loans, leaving out currently fundraising loans and refunded (problematic) loans
	df_loans = df_loans[df_loans['STATUS'].isin(['funded', 'expired'])]
	df_loans['IS_UNFUNDED'] = np.where(df_loans['STATUS'] == "expired", 1, 0)

	# define X and y
	drop_columns = ['LOAN_ID', 'LOAN_NAME', 'FUNDED_AMOUNT', 'STATUS', 'COUNTRY_CODE', 'DISBURSE_TIME', 'RAISED_TIME', 'NUM_LENDERS_TOTAL', 
					'NUM_JOURNAL_ENTRIES', 'NUM_BULK_ENTRIES', 'BORROWER_NAMES', 'IS_UNFUNDED', 'PARTNER_ID']
	text_fields = ['LOAN_USE', 'TAGS']
	alternative_data_fields = ['IMAGE_ID', 'VIDEO_ID', 'TOWN_NAME']
	date_fields = ['POSTED_TIME', 'PLANNED_EXPIRATION_TIME']
	X = df_loans.drop(drop_columns + text_fields + alternative_data_fields, axis = 1)
	y = df_loans['IS_UNFUNDED']

	pdb.set_trace()

	# preprocess
	print("Preprocessing..")

	# handle multiple borrowers for BORROWER_GENDERS and BORROWER_PICTURED and add a NUM_BORROWERS field
	X['NUM_BORROWERS'] = X['BORROWER_GENDERS'].str.split(', ').str.len()
	X['PERCENT_FEMALE'] = X['BORROWER_GENDERS'].str.split(', ').str.count("female") / X['NUM_BORROWERS']
	X['BORROWER_PICTURED'] = X['BORROWER_PICTURED'].str.lower().str.split(', ').str.count("true") > 0

	# handle date fields and add a PLANNED_DURATION field
	X['POSTED_TIME'] = pd.to_datetime(X['POSTED_TIME'])
	X['PLANNED_EXPIRATION_TIME'] = pd.to_datetime(X['PLANNED_EXPIRATION_TIME'])
	X['PLANNED_DURATION'] = X['PLANNED_EXPIRATION_TIME'] - X['POSTED_TIME']

	X['POSTED_DAY'] = X['POSTED_TIME'].dt.day
	X['POSTED_MONTH'] = X['POSTED_TIME'].dt.month
	X['POSTED_YEAR'] = X['POSTED_TIME'].dt.year
	X['PLANNED_DURATION'] = X['PLANNED_DURATION'].dt.days

	X = X.drop(['BORROWER_GENDERS', 'POSTED_TIME', 'PLANNED_EXPIRATION_TIME'], axis = 1)

	pdb.set_trace()

	print("Converting categorical variables into dummy variables..")
	X = pd.get_dummies(X)

	pdb.set_trace()

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
	X_train = fill_missing_values(X_train)
	X_test = fill_missing_values(X_test)

	# train the model and predict
	model = lr_train(X_train, y_train, penalty = 'l2', reg_const = 1)
	pred, probs = sklearn_predict(model, X_test)

	pdb.set_trace()

	accuracy_cv = cross_val_score(model, X_train, y_train, scoring = "accuracy", cv = 5)
	precision_cv = cross_val_score(model, X_train, y_train, scoring = "precision", cv = 5)
	recall_cv = cross_val_score(model, X_train, y_train, scoring = "recall", cv = 5)
	print("Cross validation accuracy: ", sum(accuracy_cv)/len(accuracy_cv))
	print("Cross validation precision: ", sum(precision_cv)/len(precision_cv))
	print("Cross validation recall: ", sum(recall_cv)/len(recall_cv))

	pdb.set_trace()

main()