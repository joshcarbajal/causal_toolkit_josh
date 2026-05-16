import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple

def calculate_ate_ci(data: pd.DataFrame, alpha: float = 0.05) -> Tuple[float, float, float]:

    y1 = data[data['T'] == 1]['Y']
    y0 = data[data['T'] == 0]['Y']
    
    ate_estimate = y1.mean() - y0.mean()
    
    se_ate = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    
    z_score = stats.norm.ppf(1 - alpha / 2)
    
    ci_lower = ate_estimate - z_score * se_ate
    ci_upper = ate_estimate + z_score * se_ate
    
    return float(ate_estimate), float(ci_lower), float(ci_upper)

def calculate_ate_pvalue(data: pd.DataFrame) -> Tuple[float, float, float]:

    y1 = data[data['T'] == 1]['Y']
    y0 = data[data['T'] == 0]['Y']
    
    ate_estimate = y1.mean() - y0.mean()
    se_ate = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    
    t_stat = ate_estimate / se_ate
    
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    
    return float(ate_estimate), float(t_stat), float(p_value)