# causal-toolkit-josh

A Python package for causal inference methods.

## Installation

```bash
git clone https://github.com/joshcarbajal/causal_toolkit_josh
cd causal-toolkit-josh
uv pip install -e .
```

## Usage

```python
from causal_toolkit_josh import calculate_ate_ci, ipw, doubly_robust
from causal_toolkit_josh import s_learner_discrete, t_learner_discrete
```

## Methods

- **rct.py** — `calculate_ate_ci()`, `calculate_ate_pvalue()`
- **propensity.py** — `ipw()`, `doubly_robust()`
- **meta_learners.py** — `s_learner_discrete()`, `t_learner_discrete()`, `x_learner_discrete()`, `double_ml_cate()`