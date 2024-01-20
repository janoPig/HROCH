# Symbolic regression and classification library  

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) [![PyPI version](https://badge.fury.io/py/HROCH.svg)](https://badge.fury.io/py/HROCH) [![Downloads](https://pepy.tech/badge/hroch)](https://pepy.tech/project/hroch) [![CodeQL](https://github.com/janoPig/HROCH/actions/workflows/codeql.yml/badge.svg)](https://github.com/janoPig/HROCH/actions/workflows/codeql.yml) [![Unittests](https://github.com/janoPig/HROCH/actions/workflows/unittests.yml/badge.svg)](https://github.com/janoPig/HROCH/actions/workflows/unittests.yml) [![pages-build-deployment](https://github.com/janoPig/HROCH/actions/workflows/pages/pages-build-deployment/badge.svg?branch=main)](https://github.com/janoPig/HROCH/actions/workflows/pages/pages-build-deployment)[![Upload Python Package](https://github.com/janoPig/HROCH/actions/workflows/python-publish.yml/badge.svg?event=release)](https://github.com/janoPig/HROCH/actions/workflows/python-publish.yml)

**High-Performance python symbolic regression library based on parallel local search**

- Zero hyperparameter tunning.
- Accurate results in seconds or minutes, in contrast to slow GP-based methods.
- Small models size.
- Support for regression, classification and fuzzy math.
- Support 32 and 64 bit floating point arithmetic.
- Work with unprotected version of math operators (log, sqrt, division)
- Speedup search by using feature importances computed from bbox model

|**Supported instructions**||
| ----------- | ----------- |
|**math**|add, sub, mul, div, pdiv, inv, minv, sq2, pow, exp, log, sqrt, cbrt, aq|
|**goniometric**|sin, cos, tan, asin, acos, atan, sinh, cosh, tanh|
|**other**|nop, max, min, abs, floor, ceil, lt, gt, lte, gte|
|**fuzzy**|f_and, f_or, f_xor, f_impl, f_not, f_nand, f_nor, f_nxor, f_nimpl|

## Sources

C++20 source code available in separate repo [sr_core](<https://github.com/janoPig/sr_core>)

## Dependencies

- AVX2 instructions set(all modern CPU support this)
- numpy
- sklearn

## Installation

```sh
pip install HROCH
```

## Usage

[Symbolic_Regression_Demo.ipynb](https://github.com/janoPig/HROCH/blob/main/examples/Symbolic_Regression_Demo.ipynb)  [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/janoPig/HROCH/blob/main/examples/Symbolic_Regression_Demo.ipynb)

[Documentation](https://janopig.github.io/HROCH/HROCH.html)

```python
from HROCH import SymbolicRegressor

reg = SymbolicRegressor(num_threads=8, time_limit=60.0, problem='math', precision='f64')
reg.fit(X_train, y_train)
yp = reg.predict(X_test)
```

## Changelog

### v1.4

- sklearn compatibility
- Classificators:
  - NonlinearLogisticRegressor for a binary classification
  - SymbolicClassifier for multiclass classification
  - FuzzyRegressor for a special binary classification

<details>
<summary>Older versions</summary>

### v1.3

- Public c++ sources
- Commanline interface changed to cpython
- Support for classification score logloss and accuracy
- Support for final transformations:
  - ordinal regression
  - logistic function
  - clipping
- Acess to equations from all paralel hillclimbers
- User defined constants

### v1.2

- Features probability as input parameter
- Custom instructions set
- Parallel hilclimbing parameters
  
### v1.1

- Improved late acceptance hillclimbing

### v1.0

- First release

</details>

## SRBench

[*full results*](https://github.com/janoPig/HROCH/blob/main/benchmarks/SRBench.md)

<img src="https://github.com/janoPig/HROCH/assets/75015989/3fa087dc-8caf-4301-86d7-4e79a4e84402" alt="SRBench" style="width:800px;"/>
