import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix


def glmfit(x, y):
    x_with_intercept = sm.add_constant(x)
    model = sm.GLM(y, x_with_intercept, family=sm.families.Binomial())
    result = model.fit()
    return result

def getperf(labels, scores, seed=None, dispflag=True):
    x = StandardScaler().fit_transform(np.array(scores).reshape(-1, 1))
    y = np.array(labels)

    model = glmfit(x, y)
    B = model.params
    se = model.bse
    p = model.pvalues

    B_lo = B[1] - 1.965 * se[1]
    B_hi = B[1] + 1.965 * se[1]
    or_value = np.exp(B[1])
    or_CI = [np.exp(B_lo), np.exp(B_hi), p[1]]

    auc = roc_auc_score(y, x)
    auc_CI = [auc]

    np.random.seed(seed)
    auc_bootstrap = []
    num_bootstrap = 2000
    for _ in range(num_bootstrap):
        indices = np.random.choice(len(y), len(y), replace=True)
        auc_bootstrap.append(roc_auc_score(y[indices], x[indices]))

    auc_bootstrap.sort()
    lower_idx = int(0.025 * num_bootstrap)
    upper_idx = int(0.975 * num_bootstrap)
    auc_CI.extend([auc_bootstrap[lower_idx], auc_bootstrap[upper_idx], 0])

    
    y_pred = (x >= 0).astype(int)

    
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    
    sensitivity = tp / (tp + fn)

    
    specificity = tn / (tn + fp)

    if dispflag:
        print("AUC: {:.4f}".format(auc))
        print("AUC CI: [{:.4f}, {:.4f}]".format(auc_CI[1], auc_CI[2]))
        print("OR: {:.4f}".format(or_value))
        print("OR CI: [{:.4f}, {:.4f}], pvalue: {:.4f}".format(or_CI[0], or_CI[1], or_CI[2]))
        print("Sensitivity (Recall para 1): {:.4f}".format(sensitivity))
        print("Specificity (Recall para 0): {:.4f}".format(specificity))


    return auc, or_value, auc_CI[1:3], or_CI, sensitivity, specificity