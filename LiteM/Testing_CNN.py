import sys
sys.path.append("..")
from project_Info import *
from LatexTable import *
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from sklearn.preprocessing import StandardScaler
from ASMOTE import ASMOTE, NoSMOTE
import numpy as np
import scipy.io as sio
import time
import argparse
from utils import cal_metrics

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin, clone

# ==================== PyTorch 1D-CNN Classifier ====================

class CNN1DNet(nn.Module):
    def __init__(self, input_dim=None):
        super(CNN1DNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

class CNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, epochs=50, batch_size=64, lr=0.001, random_state=1, device=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_ = None

    def _set_seed(self):
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        np.random.seed(self.random_state)

    def fit(self, X, y):
        self._set_seed()
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.int64)

        # Reshape to (Batch, Channel=1, Features)
        X_arr = X_arr[:, np.newaxis, :]

        tensor_x = torch.tensor(X_arr, dtype=torch.float32)
        tensor_y = torch.tensor(y_arr, dtype=torch.long)
        dataset = TensorDataset(tensor_x, tensor_y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_ = CNN1DNet(input_dim=X.shape[1]).to(self.device)

        # 1. Calculate class weights to counter imbalance
        classes, counts = np.unique(y_arr, return_counts=True)
        total_samples = len(y_arr)
        weights = total_samples / (len(classes) * counts.astype(np.float32))
        class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)

        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=1e-4)

        self.model_.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model_(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)
                optimizer.step()

        return self

    def predict_proba(self, X):
        self.model_.eval()
        X_arr = np.asarray(X, dtype=np.float32)[:, np.newaxis, :]
        tensor_x = torch.tensor(X_arr, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model_(tensor_x)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        return probabilities

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        # Returns 1 if probability of positive class >= threshold
        return (proba[:, 1] >= threshold).astype(int)
    
# ==================== Argument Parsing ====================

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--technique', choices=['ADASYN', 'ASMOTE', 'NoSMOTE', 'SMOTE', 'RUS', 'TL', 'ENN'], default='ASMOTE', help='Data augmentation technique to use')
    parser.add_argument('--classifier', type=str, choices=['CNN'], default='CNN')
    parser.add_argument('--label_column_name', type=str, default='CommentsAssociatedLabel')
    parser.add_argument('--random_state', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)

    return parser

def create_classifier(random_state: int, epochs: int = 50, batch_size: int = 64, lr: float = 0.001):
    return CNNClassifier(epochs=epochs, batch_size=batch_size, lr=lr, random_state=random_state)

def create_asmote_classifier(random_state: int):
    # Lightweight instance for internal ASMOTE synthetic boundary generation
    return CNNClassifier(epochs=20, batch_size=64, lr=0.001, random_state=random_state)

# ==================== Cross Validation ====================

def ten_folds(file_name, level, k_fold=10):
    data = pd.read_csv(file_name)
    data.columns = data.columns.str.lower()

    if level == 'file':
        X = data[file_feature_names_lowercase]
    elif level == 'class':
        X = data[class_feature_names_lowercase]
    elif level == 'method':
        X = data[method_feature_names_lowercase]
    else:
        X = data[block_feature_names_lowercase]
    y = data[args.label_column_name.lower()]

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
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        clf = create_classifier(
            random_state=args.random_state,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )

        if args.technique == 'ASMOTE':
            tech = ASMOTE(random_state=args.random_state, clf=create_asmote_classifier(random_state=args.random_state))
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

        # Scale features (preserve original index)
        scaler = StandardScaler()
        X_train_processed = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test_processed = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

        # Removing duplicates from test set (now indexes match)
        mask = ~X_test_processed.duplicated()
        X_test_processed = X_test_processed[mask]
        y_test = y_test[mask]

        # Apply the resampler on the processed training data
        X_train_resample, y_train_resample = tech.fit_resample(X_train_processed, y_train)

        # Train CNN
        clf.fit(X_train_resample, y_train_resample)

        # Predict
        y_pred = clf.predict(X_test_processed)
        y_pred_prob = clf.predict_proba(X_test_processed)[:, 1]

        # Placeholder feature importance for CNN
        importance = np.zeros(X_train.shape[1])
        feature_importances.append(importance)

        # Calculate metrics
        metrics = cal_metrics(y_test, y_pred, y_pred_prob)

        accuracies.append(metrics['ACC'])
        precisions.append(metrics['P'])
        recalls.append(metrics['R'])
        f1_scores.append(metrics['F1'])
        AUCs.append(metrics['AUC'])
        MCCs.append(metrics['MCC'])

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
    lst = np.asarray(lst)
    if lst.ndim == 0:
        return [0.0]

    min_value = min(lst)
    max_value = max(lst)
    if max_value == min_value:
        return [0.0 for x in lst]
    return [(x - min_value) / (max_value - min_value) for x in lst]

# ==================== Main Execution ====================

parser = get_parser()
args = parser.parse_args()

latex_matrix = []
importances = []
times = []

for project in projects:
    latex_line = []
    for granularity in granularities:
        file = f'../code snippets-with-labels&metrics/{granularity}/{project}_{granularity}Level.csv'
        print('====' + project, granularity + '====')
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