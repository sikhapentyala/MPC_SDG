import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

def train_and_evaluate(df_train, df_test, syn_train, target_column):
    # Split the data into features and target
    X_train = df_train.drop(target_column, axis=1)
    y_train = df_train[target_column]

    X_test = df_test.drop(target_column, axis=1)
    y_test = df_test[target_column]

    # Train LR on original data
    lr_original = LogisticRegression()
    lr_original.fit(X_train, y_train)

    # Train RF on original data
    rf_original = RandomForestClassifier()
    rf_original.fit(X_train, y_train)

    # Train LR on synthetic data
    lr_synthetic = LogisticRegression()
    lr_synthetic.fit(syn_train.drop(target_column, axis=1), syn_train[target_column])

    # Train RF on synthetic data
    rf_synthetic = RandomForestClassifier()
    rf_synthetic.fit(syn_train.drop(target_column, axis=1), syn_train[target_column])

    # Evaluate LR and RF on the original test data
    lr_auc_original = roc_auc_score(y_test, lr_original.predict_proba(X_test)[:, 1])
    lr_f1_original = f1_score(y_test, lr_original.predict(X_test))

    rf_auc_original = roc_auc_score(y_test, rf_original.predict_proba(X_test)[:, 1])
    rf_f1_original = f1_score(y_test, rf_original.predict(X_test))

    # Evaluate LR and RF on the synthetic test data
    lr_auc_synthetic = roc_auc_score(y_test, lr_synthetic.predict_proba(X_test)[:, 1])
    lr_f1_synthetic = f1_score(y_test, lr_synthetic.predict(X_test))

    rf_auc_synthetic = roc_auc_score(y_test, rf_synthetic.predict_proba(X_test)[:, 1])
    rf_f1_synthetic = f1_score(y_test, rf_synthetic.predict(X_test))

    return {
        'LR_AUC_Original': lr_auc_original,
        'LR_F1_Original': lr_f1_original,
        'RF_AUC_Original': rf_auc_original,
        'RF_F1_Original': rf_f1_original,
        'LR_AUC_Synthetic': lr_auc_synthetic,
        'LR_F1_Synthetic': lr_f1_synthetic,
        'RF_AUC_Synthetic': rf_auc_synthetic,
        'RF_F1_Synthetic': rf_f1_synthetic
    }

'''
def stats_evaluate(df_train, syn_train):

    workload = list(itertools.combinations(data.domain, args.degree))
    workload = [cl for cl in workload if data.domain.size(cl) <= args.max_cells]
    if args.num_marginals is not None:
        workload = [workload[i] for i in prng.choice(len(workload), args.num_marginals, replace=False)]
        
    errors = []
    for proj, wgt in workload:
        X = df_train.project(proj).datavector()
        Y = syn_train.project(proj).datavector()
        e = 0.5*wgt*np.linalg.norm(X/X.sum() - Y/Y.sum(), 1)
        errors.append(e)
    print('Average Error: ', np.mean(errors))
'''
path = '/Users/sikha/Development/MPC_SDG/private-pgm-master/data/'
dataset_name = 'compass'
df_train = pd.read_csv(path+dataset_name+'_train.csv')
df_test = pd.read_csv(path+dataset_name+'_test.csv')
syn_train = pd.read_csv(path+dataset_name+'_syn_train.csv')
target_column = 'label'
eval_result = train_and_evaluate(df_train, df_test, syn_train, target_column)
print(eval_result)