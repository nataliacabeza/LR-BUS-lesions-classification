import numpy as np
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
import pandas as pd
from lr_f import MyLogisticRegression
from scores_f import getperf

def evaluate(db, cols, name_model, params, nfolds, nfolds_cv, seed, seed_model= None):
    Y_prob = np.array([])

    folds = LeaveOneGroupOut()
    print(f'Training {name_model} model ...')

    df = pd.DataFrame(columns=['Group', 'Y_true', 'Y_prob'])

    for i, (train, test) in enumerate(folds.split(db, db[cols['classes']], groups=db[cols['group']]), 1):
        print(f'Fold {i}', end='\r')

        X_train = np.array(db.iloc[train][cols['features']].tolist())
        Y_train = np.array(db.iloc[train][cols['classes']])
        X_test = np.array(db.iloc[test][cols['features']].tolist())
        Y_test = np.array(db.iloc[test][cols['classes']])

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if name_model == 'LR_custom':

            model = MyLogisticRegression()

            param_grid = {
                'penalty': ['l1', 'l2'],
                'C': np.logspace(0, 4, 5), # antes -4, 4, 10
                'solver': ['lilblinear']

            }

            grid_search = GridSearchCV(model, param_grid, scoring='roc_auc', cv=nfolds_cv, n_jobs=12)
            grid_search.fit(X_train, Y_train)

            best_model = grid_search.best_estimator_
            Y_prob = best_model.predict_proba(X_test)[:, 1]  # Se usa Y_prob en lugar de y_pred_proba

        temp_df = pd.DataFrame({'Fold': i,
                                'Group': db.iloc[test][cols['group']],
                                'Y_true': Y_test,
                                'Y_prob': Y_prob})
        df = pd.concat([df, temp_df], ignore_index=True)

    df = df.groupby(['Fold', 'Group']).agg({'Y_true': 'first', 'Y_prob': 'mean'}).reset_index()
    df['Y_true'] = df['Y_true'].astype('int64')

    print('\n\nResults')
    getperf(df['Y_true'], df['Y_prob'], seed)

    return df

