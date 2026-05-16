import pandas as pd
import numpy as np
import patsy
from sklearn.linear_model import LogisticRegression, LinearRegression

def ipw(df: pd.DataFrame, ps_formula: str, T: str, Y: str) -> float:
    """
    Calculate the Average Treatment Effect (ATE) using Inverse Propensity Weighting.
    
    """

    X = patsy.dmatrix(ps_formula, df, return_type='dataframe')
    
    lr = LogisticRegression(penalty=None, max_iter=1000)
    lr.fit(X, df[T])

    ps = lr.predict_proba(X)[:, 1]
    
    t_vec = df[T].values
    y_vec = df[Y].values
    
    weight_component = (t_vec - ps) / (ps * (1 - ps))
    ate = np.mean(weight_component * y_vec)
    
    return float(ate)



def doubly_robust(df: pd.DataFrame, formula: str, T: str, Y: str) -> float:
    """
    Calculate the Average Treatment Effect (ATE) using Doubly Robust Estimation.
    """
    X = patsy.dmatrix(formula, df, return_type='dataframe')
    y_vec = df[Y].values
    t_vec = df[T].values

    ps_model = LogisticRegression(penalty=None, max_iter=1000)
    ps_model.fit(X, t_vec)
    ps = ps_model.predict_proba(X)[:, 1]

    model_t1 = LinearRegression()
    model_t1.fit(X[t_vec == 1], y_vec[t_vec == 1])
    mu_1 = model_t1.predict(X)

    model_t0 = LinearRegression()
    model_t0.fit(X[t_vec == 0], y_vec[t_vec == 0])
    mu_0 = model_t0.predict(X)


    dr_1 = (t_vec * (y_vec - mu_1) / ps) + mu_1
    
    dr_0 = ((1 - t_vec) * (y_vec - mu_0) / (1 - ps)) + mu_0

    ate = np.mean(dr_1) - np.mean(dr_0)
    
    return float(ate)