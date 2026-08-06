import sys
sys.path.append("..")
from project_Info import *
from LatexTable import *
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ASMOTE import ASMOTE, NoSMOTE
import numpy as np
import scipy.io as sio
import time
import argparse
from utils import cal_metrics

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier                                              # <-- Added KNN
from sklearn.neural_network import MLPClassifier                                                # <-- Added MLP
from sklearn.svm import SVC                                                                     # <-- Added SVM
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier  # <-- Added ET, ADA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier

from sklearn.base import clone

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--technique', choices=['ADASYN', 'ASMOTE', 'NoSMOTE', 'SMOTE', 'RUS', 'TL', 'ENN'], default='ASMOTE', help='Data augmentation technique to use')
    
    parser.add_argument('--classifier', type=str, choices=['LightGBM', 'DecisionTree', 'LogisticRegression', 'RF', 'XGB', 'SVM', 'KNN', 'ET', 'ADA', 'MLP', 'NaiveBayes', 'LDA', 'Ridge'], default='LightGBM')
    parser.add_argument('--label_column_name', type=str, default='CommentsAssociatedLabel')
    parser.add_argument('--random_state', type=int, default=1)

    return parser



def create_classifier(name : str, random_state : int):
    classifiers = {
        'LightGBM': LGBMClassifier(
            random_state=random_state,
            n_jobs=12,
            n_estimators=500,
            learning_rate=0.05,
            verbosity=-1
        ),
        'NaiveBayes': GaussianNB(),
        'LDA': LinearDiscriminantAnalysis(),
        'LogisticRegression': LogisticRegression(
            random_state=random_state,
            max_iter=1000,
        ),
        'Ridge': RidgeClassifier(random_state=random_state),
        'RF': RandomForestClassifier(
            random_state=random_state,
            n_jobs=12,
            n_estimators=500,
        ),
        'XGB': XGBClassifier(
            random_state=random_state,
            n_jobs=12,
            n_estimators=500,
            learning_rate=0.05,
            eval_metric='logloss',
        ),
        'SVM': SVC(
            random_state=random_state,
            probability=True,
        ),
        'KNN': KNeighborsClassifier(n_jobs=12),
        'ET': ExtraTreesClassifier(
            random_state=random_state,
            n_jobs=12,
            n_estimators=500,
        ),
        'ADA': AdaBoostClassifier(
            random_state=random_state,
            n_estimators=200,
        ),
        'MLP': MLPClassifier(
            random_state=random_state,
            max_iter=1000,
        ),
        'DecisionTree': DecisionTreeClassifier(
            random_state=random_state,
        ),
    }

    return clone(classifiers.get(name, classifiers['DecisionTree']))

def create_asmote_classifier(name, random_state):
    classifiers = {
            'LightGBM': LGBMClassifier(random_state=random_state, verbosity=-1),
            'NaiveBayes': GaussianNB(),
            'LDA': LinearDiscriminantAnalysis(),
            'LogisticRegression': LogisticRegression(
                random_state=random_state,
                max_iter=2000,
            ),
            'Ridge': RidgeClassifier(random_state=random_state),
            'RF': RandomForestClassifier(random_state=random_state),
            'XGB': XGBClassifier(
                random_state=random_state, 
                eval_metric='logloss'
            ),
            'SVM': SVC(
                random_state=random_state,
                probability=True,
            ),
            'KNN': KNeighborsClassifier(),
            'ET': ExtraTreesClassifier(
                random_state=random_state
            ),
            'ADA': AdaBoostClassifier(
                random_state=random_state
            ),
            'MLP': MLPClassifier(
                random_state=random_state,
                max_iter=1000,
            ),
            'DecisionTree': DecisionTreeClassifier(
                random_state=random_state,
            ),
        }
    
    return clone(classifiers.get(name, classifiers['DecisionTree']))

# ten folds cross validation
def ten_folds(file_name, level, k_fold=10):
    # load dataset
    data = pd.read_csv(file_name)
    data.columns = data.columns.str.lower()

    # define features and labels
    if level == 'file':
        X = data[file_feature_names_lowercase]
    elif level == 'class':
        X = data[class_feature_names_lowercase]
    elif level == 'method':
        X = data[method_feature_names_lowercase]
    else:
        X = data[block_feature_names_lowercase]
    y = data[args.label_column_name.lower()]

    # create StratifiedKFold object
    skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=args.random_state)

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    start_time = time.time()
    feature_importances = []
    AUCs = []
    MCCs = []
    X = X.fillna(-1)
    for train_index, test_index in skf.split(X, y):
        # label evenly distributed
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Prediction Classifier
        clf = create_classifier(name=args.classifier, random_state=args.random_state)

        # Resample technique
        if args.technique == 'ASMOTE':
            tech = ASMOTE(random_state=args.random_state, clf=create_asmote_classifier(name = args.classifier, random_state=args.random_state))
        elif args.technique == 'ADASYN':
            tech = ADASYN(random_state=args.random_state)
        elif args.technique == 'SMOTE':
            tech = SMOTE(random_state=args.random_state)
        elif args.technique == 'NoSMOTE':
            tech = NoSMOTE()
        elif args.technique == 'RUS':
            tech = RandomUnderSampler(random_state=args.random_state)
        elif args.technique == 'TL':
            tech = TomekLinks()
        elif args.technique == 'ENN':
            tech = EditedNearestNeighbours()


        # Scale features only if using LogisticRegression, SVM, KNN, MLP, or ASMOTE (all distance-sensitive)
        distance_sensitive_models = ['LogisticRegression', 'SVM', 'KNN', 'MLP', 'LDA', 'Ridge']
        if args.classifier in distance_sensitive_models:
            scaler = StandardScaler()
            X_train_processed = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
            X_test_processed = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        else:
            X_train_processed = X_train.copy()
            X_test_processed = X_test.copy()

        # removing duplicates
        mask = ~X_test_processed.duplicated()

        X_test_processed = X_test_processed[mask]
        y_test = y_test[mask]

        # Apply the resampler on the processed training data
        X_train_resample, y_train_resample = tech.fit_resample(X_train_processed, y_train)

        # 4. Train the model on resampled data
        clf.fit(X_train_resample, y_train_resample)

        # 5. Predict using the correctly processed test set
        y_pred = clf.predict(X_test_processed)
        if hasattr(clf, "predict_proba"):
            y_pred_prob = clf.predict_proba(X_test_processed)[:, 1]
        else:
            # Use decision function for RidgeClassifier as a proxy for AUC calculation
            y_pred_prob = clf.decision_function(X_test_processed)

        # record importances (applicable models will be checked dynamically)
        if hasattr(clf, 'feature_importances_'):
            importance = clf.feature_importances_
        elif hasattr(clf, 'coef_'):
            importance = np.abs(clf.coef_[0])
        else:
            importance = np.zeros(X_train.shape[1])

        feature_importances.append(importance)

        # calculate metrics
        metrics = cal_metrics(y_test, y_pred, y_pred_prob)

        # save this round result
        accuracies.append(metrics['ACC'])
        precisions.append(metrics['P'])
        recalls.append(metrics['R'])
        f1_scores.append(metrics['F1'])
        AUCs.append(metrics['AUC'])
        MCCs.append(metrics['MCC'])

    # calculate average
    mean_accuracy = sum(accuracies) / k_fold
    mean_precision = sum(precisions) / k_fold
    mean_recall = sum(recalls) / k_fold
    mean_f1_score = sum(f1_scores) / k_fold
    cost_time = time.time() - start_time
    mean_auc = sum(AUCs) / k_fold
    mean_mcc = sum(MCCs) / k_fold
    mean_feature_importances = np.mean(np.array(feature_importances), axis=0)
    mean_feature_importances = normalize_list(mean_feature_importances)

    print("Mean Accuracy:{:.2f}".format(mean_accuracy))
    print("Mean Precision:{:.2f}".format(mean_precision))
    print("Mean Recall:{:.2f}".format(mean_recall))
    print("Mean F1-score:{:.2f}".format(mean_f1_score))
    print("Mean AUC: {:.2f}".format(mean_auc))
    print("Mean MCC: {:.2f}".format(mean_mcc))
    print("Cost Time: {:.2f} seconds".format(cost_time))

    return mean_precision, mean_recall, mean_f1_score, cost_time, mean_feature_importances, mean_auc, mean_mcc

def normalize_list(lst):
    # Convert to a numpy array to easily check shape and handle arrays/scalars safely
    lst = np.asarray(lst)
    if lst.ndim == 0:  # It's a single scalar value, can't normalize it
        return [0.0]

    min_value = min(lst)
    max_value = max(lst)
    if max_value == min_value:
        return [0.0 for x in lst]
    return [(x - min_value) / (max_value - min_value) for x in lst]

parser = get_parser()
args = parser.parse_args()

latex_matrix = []
importances = []
times = []

for project in projects:
    latex_line = []
    for granularity in granularities:
        file = f'../code snippets-with-labels&metrics/{granularity}/{project}_{granularity}Level.csv'
        print('===='+project, granularity+'====')
        p, r, f, t, i, auc, mcc = ten_folds(file, granularity)
        latex_line = latex_line + [p, r, f, auc, mcc]
        times.append(t)
        importances.append(i)
    latex_matrix.append(latex_line)

avgs = avgEachColumn(latex_matrix)
matrix = insertRow(latex_matrix, avgs, len(latex_matrix))
project_names.append('\\textbf{Average}')
matrix = insertColumn(matrix, project_names, 0)
writeTable(matrix, f'results/within_project_{args.technique}_{args.classifier}.txt')

with open(f'results/time_{args.technique}_{args.classifier}.txt', 'w') as f:
    for idx, item in enumerate(granularities):
        f.write(f"====={item}=====\n")
        tmp = times[idx*18:(idx+1)*18]
        for ttt in tmp:
            f.write("{:.2f}\n".format(ttt))
        f.write("{:.2f}\n".format(np.median(tmp)))
        f.write("{:.2f}\n".format(np.sum(tmp)))

sio.savemat(f'results/importance_{args.technique}_{args.classifier}.mat', {
    'data': importances,
    'file_feature_names': file_feature_names,
    'class_feature_names': class_feature_names,
    'method_feature_names': method_feature_names,
    'block_feature_names': block_feature_names,
})