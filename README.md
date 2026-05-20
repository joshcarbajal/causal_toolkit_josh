# causal-toolkit-josh

[![Tests](https://github.com/joshcarbajal/causal_toolkit_josh/workflows/Tests/badge.svg)](https://github.com/joshcarbajal/causal_toolkit_josh/actions)
[![PyPI version](https://badge.fury.io/py/causal-toolkit-josh.svg)](https://pypi.org/project/causal-toolkit-josh/)

A Python package for causal inference methods including ATE estimation, propensity score methods, and meta-learners.

## Installation

```bash
pip install causal-toolkit-josh
```

## Quick Start

```python
from causal_toolkit_josh import calculate_ate_ci, calculate_ate_pvalue
from causal_toolkit_josh import ipw, doubly_robust
from causal_toolkit_josh import s_learner_discrete, t_learner_discrete, x_learner_discrete, double_ml_cate
```

## Methods

- **rct.py** — `calculate_ate_ci()`, `calculate_ate_pvalue()`
- **propensity.py** — `ipw()`, `doubly_robust()`
- **meta_learners.py** — `s_learner_discrete()`, `t_learner_discrete()`, `x_learner_discrete()`, `double_ml_cate()`

## License

MIT