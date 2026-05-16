import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression

def s_learner_discrete(train, test, X, T, y):

    model = LGBMRegressor(random_state=42)
    features = X + [T]
    model.fit(train[features], train[y])
    
    test_t1 = test.copy()
    test_t1[T] = 1
    test_t0 = test.copy()
    test_t0[T] = 0
    
    cate = model.predict(test_t1[features]) - model.predict(test_t0[features])
    
    res = test.copy()
    res['cate'] = cate
    return res

def t_learner_discrete(train, test, X, T, y):

    train_0 = train[train[T] == 0]
    train_1 = train[train[T] == 1]
    
    mu_0 = LGBMRegressor(random_state=42).fit(train_0[X], train_0[y])
    mu_1 = LGBMRegressor(random_state=42).fit(train_1[X], train_1[y])
    
    cate = mu_1.predict(test[X]) - mu_0.predict(test[X])
    
    res = test.copy()
    res['cate'] = cate
    return res

def x_learner_discrete(train, test, X, T, y):

    train_0 = train[train[T] == 0]
    train_1 = train[train[T] == 1]
    mu0 = LGBMRegressor(random_state=42).fit(train_0[X], train_0[y])
    mu1 = LGBMRegressor(random_state=42).fit(train_1[X], train_1[y])
    
    d1 = train_1[y] - mu0.predict(train_1[X])
    d0 = mu1.predict(train_0[X]) - train_0[y]
    
    tau1 = LGBMRegressor(random_state=42).fit(train_1[X], d1)
    tau0 = LGBMRegressor(random_state=42).fit(train_0[X], d0)
    
    e_model = LogisticRegression(penalty=None, random_state=42).fit(train[X], train[T])
    e_x = e_model.predict_proba(test[X])[:, 1]
    
    cate = e_x * tau0.predict(test[X]) + (1 - e_x) * tau1.predict(test[X])
    
    res = test.copy()
    res['cate'] = cate
    return res

def double_ml_cate(train, test, X, T, y):

    model_y = LGBMRegressor(random_state=42).fit(train[X], train[y])
    y_res = train[y] - model_y.predict(train[X])
    
    model_t = LGBMRegressor(random_state=42).fit(train[X], train[T])
    t_res = train[T] - model_t.predict(train[X])

    y_star = y_res / t_res
    weights = t_res ** 2
    
    cate_model = LGBMRegressor(random_state=42)
    cate_model.fit(train[X], y_star, sample_weight=weights)
    
    res = test.copy()
    res['cate'] = cate_model.predict(test[X])
    return res