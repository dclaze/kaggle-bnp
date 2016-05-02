import kaggle_santander, os

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score

import numpy as np
import xgboost as xgb

def beep():
    print "\a";

def getFullFilePath(fileName):
	return os.path.join(os.path.dirname(__file__), '../data', fileName);


def allItemsExceptIndex(arr, index):
    if index == 0:
        return arr[1:];
    elif index == len(arr) - 1:
        return arr[0:index];
    else:
        return arr[0:index] + arr[index+1:]

def predictionToOutput(test_data, prediction):
    output = [];
    for idx, val in enumerate(prediction):
        output.append([test_data[idx][0], 1 - val[0]])
    return np.array(output);

def run_random_forest():
    dataset = np.genfromtxt(open(getFullFilePath('train.csv'),'r'), delimiter=',', dtype='f8')[1:] 
    train = [x[0:370] for x in dataset]
    target = [x[370] for x in dataset]
    test = np.genfromtxt(open(getFullFilePath('test.csv'),'r'), delimiter=',', dtype='f8')[1:]

    train_exp, train_test, target_exp, target_test = train_test_split(train, target, test_size=0.20, random_state=1337)

    rf_classifier = RandomForestClassifier(n_estimators=300, n_jobs=3)
    rf_classifier.fit(train_exp, target_exp)

    prediction = rf_classifier.predict_proba(test)

    print("Random Forest Score: ", roc_auc_score(target_test, rf_classifier.predict_proba(train_test)[:,1],average='macro'))

    np.savetxt('data/final_submission_random_forests.csv', predictionToOutput(test, prediction), delimiter=',', fmt='%d,%f', header='ID, TARGET', comments='')

def run_xg_boost():
    dataset = np.genfromtxt(open(getFullFilePath('train.csv'),'r'), delimiter=',', dtype='f8')[1:]
    train = [x[0:370] for x in dataset]
    target = [x[370] for x in dataset]
    test = np.genfromtxt(open(getFullFilePath('test.csv'),'r'), delimiter=',', dtype='f8')[1:]
    
    train_exp, train_test, target_exp, target_test = train_test_split(train, target, test_size=0.20, random_state=1337)
    
    xgb_classifier = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
    xgb_classifier.fit(train_exp, target_exp)
    
    prediction = xgb_classifier.predict_proba(test)

    print("XGBoost Score: ", roc_auc_score(target_test, xgb_classifier.predict_proba(train_test)[:,1],average='macro'))

    np.savetxt('data/final_submission_xg_boost.csv', predictionToOutput(test, prediction), delimiter=',', fmt='%d,%f', header='ID, TARGET', comments='')

def main():
    run_xg_boost();
    run_random_forest();
    beep();

if __name__=="__main__":
    main()