# HROCH  

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/HROCH) [![PyPI version](https://badge.fury.io/py/HROCH.svg)](https://badge.fury.io/py/HROCH) [![Downloads](https://pepy.tech/badge/hroch)](https://pepy.tech/project/hroch) [![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=janoPig_HROCH&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=janoPig_HROCH) [![CodeQL](https://github.com/janoPig/HROCH/actions/workflows/codeql.yml/badge.svg)](https://github.com/janoPig/HROCH/actions/workflows/codeql.yml)

| Windows |Results|Linux|Results|
| ----------- | ----------- | ----------- | ----------- |
|[![Strogatz - windows](https://github.com/janoPig/HROCH/actions/workflows/strogatz_test_win.yml/badge.svg)](https://github.com/janoPig/HROCH/actions/workflows/strogatz_test_win.yml)|[results.csv](ci-test/results/strogatz_win.csv)|[![Strogatz - linux](https://github.com/janoPig/HROCH/actions/workflows/strogatz_test_linux.yml/badge.svg)](https://github.com/janoPig/HROCH/actions/workflows/strogatz_test_linux.yml)|[results.csv](ci-test/results/strogatz_linux.csv)|
|[![Feynman - windows](https://github.com/janoPig/HROCH/actions/workflows/feynman_test_win.yml/badge.svg)](https://github.com/janoPig/HROCH/actions/workflows/feynman_test_win.yml)|[results.csv](ci-test/results/feynman_win.csv)|[![Feynman - linux](https://github.com/janoPig/HROCH/actions/workflows/feynman_test_linux.yml/badge.svg)](https://github.com/janoPig/HROCH/actions/workflows/feynman_test_linux.yml)|[results.csv](ci-test/results/feynman_linux.csv)|

**High-Performance python symbolic regression library based on parallel late acceptance hill-climbing**

- Zero hyperparameter tunning.
- Accurate results in seconds or minutes, in contrast to slow GP-based methods.
- Small models size.
- Support mathematic equations and fuzzy logic operators.
- Support 32 and 64 bit floating point arithmetic.
- Work with unprotected version of math operators (log, sqrt, division)
- Speedup search by using feature importances computed from bbox model
- [CLI](README_CLI.md)

|**Supported instructions**||
| ----------- | ----------- |
|**math**|add, sub, mul, div, inv, minv, sq2, pow, exp, log, sqrt, cbrt, aq|
|**goniometric**|sin, cos, tan, asin, acos, atan, sinh, cosh, tanh|
|**other**|nop, max, min, abs, floor, ceil, lt, gt, lte, gte|
|**fuzzy**|f_and, f_or, f_xor, f_impl, f_not, f_nand, f_nor, f_nxor, f_nimpl|

## Dependencies

- AVX2 instructions set(all modern CPU support this)
- numpy

## Installation

```sh
pip install HROCH
```

## Usage

[Symbolic_Regression_Demo.ipynb](examples/Symbolic_Regression_Demo.ipynb)  [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/janoPig/HROCH/blob/main/examples/Symbolic_Regression_Demo.ipynb)

```python
from HROCH import PHCRegressor

reg = PHCRegressor(num_threads=8, time_limit=60.0, problem='math', precision='f64')
reg.fit(X_train, y_train)
yp = reg.predict(X_test)
```

## Changelog

### v1.2

- Features probability as input parameter
- Custom instructions set
- Parallel hilclimbing parameters
  
### v1.1

- Improved late acceptance hillclimbing

### v1.0

- First release


## SRBench[*](benchmarks/SRBench.md)


![image](https://user-images.githubusercontent.com/75015989/212561560-39393068-8d72-48f4-b11c-7a14db029faf.png)




