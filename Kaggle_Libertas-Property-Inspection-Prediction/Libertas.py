import pandas as pd
import numpy as np 
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
import xgboost as xgb
	
def xgboost_pred(train,labels,test):
	params = {}
	params["objective"] = "count:poisson"
	params["eta"] = 0.005
	params["min_child_weight"] = 6
	params["subsample"] = 0.7
	params["colsample_bytree"] = 0.7
	params["scale_pos_weight"] = 1
	params["silent"] = 1
	params["max_depth"] = 9
    
    
	plst = list(params.items())

	#Using 4000 rows for early stopping. 
	offset = 4000
	num_rounds = 10000
	xgtest = xgb.DMatrix(test)

	#create a train and validation dmatrices 
	xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
	xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

	#train using early stopping and predict
	watchlist = [(xgtrain, 'train'),(xgval, 'val')]
	model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)
	preds1 = model.predict(xgtest,ntree_limit=model.best_iteration)


	#reverse train and labels and use different 4k for early stopping. 
	train = train[::-1,:]
	labels = np.log(labels[::-1])

	xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
	xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

	watchlist = [(xgtrain, 'train'),(xgval, 'val')]
	model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)
	preds2 = model.predict(xgtest,ntree_limit=model.best_iteration)


	#combine predictions
	#since the metric only cares about relative rank we don't need to average
	print "preds1 -> ", preds1 * 0.5
	print "preds2 -> ", np.exp(preds2)*0.5, "\n"
	preds = (preds1)*0.5 + np.exp(preds2)*0.5
	return preds

def xgboost_Label(train,test,labels):
	print("Train & Test xgboost_Label")
	train.drop('T2_V10', axis=1, inplace=True)
	train.drop('T2_V7', axis=1, inplace=True)
	train.drop('T1_V13', axis=1, inplace=True)
	train.drop('T1_V10', axis=1, inplace=True)

	test.drop('T2_V10', axis=1, inplace=True)
	test.drop('T2_V7', axis=1, inplace=True)
	test.drop('T1_V13', axis=1, inplace=True)
	test.drop('T1_V10', axis=1, inplace=True)

	train = np.array(train)
	test = np.array(test)

	for i in range(train.shape[1]):
	    lbl = preprocessing.LabelEncoder()
	    lbl.fit(list(train[:,i]) + list(test[:,i]))
	    train[:,i] = lbl.transform(train[:,i])
	    test[:,i] = lbl.transform(test[:,i])

	train = train.astype(float)
	test = test.astype(float)


	preds = xgboost_pred(train,labels,test)
	return preds

def xgboost_Vect(train,test,labels):
	print("Train & Test xgboost_Vect")
	test = test.T.to_dict().values()
	train = train.T.to_dict().values()
	vec = DictVectorizer()
	train = vec.fit_transform(train)
	test = vec.transform(test)

	preds = xgboost_pred(train,labels,test)
	return preds

def xgboost_Dummies(train,test,labels):
	print("Train & Test xgboost_Dummies")
	train = pd.get_dummies(train)
	test = pd.get_dummies(test)
	preds = xgboost_pred(train.values,labels,test.values)
	return preds

if __name__ == '__main__':

	#load train and test 
	print("Chargement des donnees")
	train  = pd.read_csv('train.csv', index_col=0)
	test  = pd.read_csv('test.csv', index_col=0)

	labels = train.Hazard
	train.drop('Hazard', axis=1, inplace=True)


	############ RF LABEL #################
	train_rf = train.copy()
	test_rf = test.copy()

	train_rf.drop('T2_V10', axis=1, inplace=True)
	train_rf.drop('T2_V7', axis=1, inplace=True)
	train_rf.drop('T1_V13', axis=1, inplace=True)
	train_rf.drop('T1_V10', axis=1, inplace=True)

	test_rf.drop('T2_V10', axis=1, inplace=True)
	test_rf.drop('T2_V7', axis=1, inplace=True)
	test_rf.drop('T1_V13', axis=1, inplace=True)
	test_rf.drop('T1_V10', axis=1, inplace=True)

	train_rf = np.array(train_rf)
	test_rf = np.array(test_rf)
	for i in range(train_rf.shape[1]):
	    lbl = LabelEncoder()
	    lbl.fit(list(train_rf[:,i]))
	    train_rf[:,i] = lbl.transform(train_rf[:,i])

	for i in range(test_rf.shape[1]):
	    lbl = LabelEncoder()
	    lbl.fit(list(test_rf[:,i]))
	    test_rf[:,i] = lbl.transform(test_rf[:,i])

	train_rf = train_rf.astype(float)
	test_rf = test_rf.astype(float)

	############# RF DUMMIES ##############
	train_du = train.copy()
	test_du = test.copy()

	train_du.drop('T2_V10', axis=1, inplace=True)
	train_du.drop('T2_V7', axis=1, inplace=True)
	train_du.drop('T1_V13', axis=1, inplace=True)
	train_du.drop('T1_V10', axis=1, inplace=True)

	test_du.drop('T2_V10', axis=1, inplace=True)
	test_du.drop('T2_V7', axis=1, inplace=True)
	test_du.drop('T1_V13', axis=1, inplace=True)
	test_du.drop('T1_V10', axis=1, inplace=True)
	train_du = pd.get_dummies(train_du)
	test_du = pd.get_dummies(test_du)


	algorithms = [
		[RandomForestRegressor(n_estimators=2048, max_depth=18, min_samples_split=4, min_samples_leaf=20), "labels"],
		[RandomForestRegressor(n_estimators=2048, max_depth=22, min_samples_split=4, min_samples_leaf=20), "dummies"],
		["xgboost_Dummies", ""],
		["xgboost_Label", ""],
		["xgboost_Vect", ""]
	]
	full_predictions = []
	for alg, predictors in algorithms:
		if(alg == "xgboost_Label"):
			full_predictions.append(xgboost_Label(train, test, labels))
		elif(alg == "xgboost_Vect"):
			full_predictions.append(xgboost_Vect(train, test, labels))
		elif(alg == "xgboost_Dummies"):
			full_predictions.append(xgboost_Dummies(train, test, labels))
		else :
			if(predictors == "dummies"):
				print("Train ", alg.__class__.__name__ , " dummies Model ")
				alg=BaggingRegressor(alg)
				alg.fit(train_du, labels)
				print 'Prediction :' , alg.__class__.__name__ , ' dummies Model '
				prediction = alg.predict(test_du)
				full_predictions.append(prediction)
			else :
				print("Train ", alg.__class__.__name__ , " Label Model ")
				alg=BaggingRegressor(alg)
				alg.fit(train_rf, labels)
				print 'Prediction :' , alg.__class__.__name__ , ' Label Model '
				prediction = alg.predict(test_rf)
				full_predictions.append(prediction)


	# Ensemble models
	RF_label_pred = full_predictions[0]
	RF_dummies_pred = full_predictions[1]
	pred_xgb_dummies = full_predictions[2]
	pred_xgb_Label = full_predictions[3]
	pred_xgb_Vect = full_predictions[4]

	pred_RF = RF_dummies_pred * 0.50 + RF_label_pred * 0.50 
	pred_xgb = pred_xgb_dummies * 0.45 + pred_xgb_Label * 0.28 + pred_xgb_Vect * 0.27 
	preds = pred_RF * 0.15 + pred_xgb * 0.85 

	#generate solution
	print("Construction csv")
	preds = pd.DataFrame({"Id": test.index, "Hazard": preds})
	preds = preds.set_index('Id')
	preds.to_csv('prediction_PqiNNN_Sad.csv')

