# HROCH

```txt
      c~~p ,---------.
 ,---'oo  )           \
( O O                  )/
 `=^='                 /
       \    ,     .   /
       \\  |-----'|  /
       ||__|    |_|__|
```

  Simple and fast hillclimb algorithm for symbolic regression.
  Python wraper(a sklearn-compatible Regressor) for [CLI](README_CLI.md)

  Zero hyperparameter tunning. Only parameters to set are time limit and r2 error as stopping criterium.

  *CLI builded on Ubuntu 20.04 with g++-9
  Tested on Ubuntu 20.04 and Pop!_OS 22.04*

  Search space: add, mul, sq2, sub, div, sqrt, exp, log, asin, acos, sin, cos, tanh

  <span style="color:red"> HROCH use unprotected version of math operations (eg. log or division)</span>

## Requirements

- AVX2 instructions set(all modern CPU support this)
- pandas
- sympy
- numpy

## Performance  

Feynman dataset(all 119 samples from  [PMLB](https://github.com/EpistasisLab/pmlb))  

**5 seconds** time limit, 8 threads, AMD Ryzen 5 1600

| **target noise** | **r2 > 0.999** | **r2 = 1.0** | **r2 mean** | **r2 median** | **average model complexity** |
|:----------------:|:--------------:|:------------:|:-----------:|:-------------:|:----------------------------:|
| **0**            | 69%            | 52%          | 0.98        | 1.0           | 14                           |
| **0.001**        | 68%            | 34%          | 0.97        | 1.0           | 14.5                         |
| **0.01**         | 68%            | 34%          | 0.97        | 1.0           | 13.5                         |
| **0.1**          | 63%            | 30%          | 0.97        | 1.0           | 12.5                         |

**5 minutes** time limit, 8 threads, AMD Ryzen 5 1600

| **target noise** | **r2 > 0.999** | **r2 = 1.0** | **r2 mean** | **r2 median** | **average model complexity** |
|:----------------:|:--------------:|:------------:|:-----------:|:-------------:|:----------------------------:|
| **0**            | 89%            | 67%          | 0.999       | 1.0           | 16.5                         |
| **0.001**        | 89%            | 42%          | 0.999       | 1.0           | 16                           |
| **0.01**         | 89%            | 35%          | 0.999       | 1.0           | 15                           |
| **0.1**          | 73%            | 32%          | 0.998       | 1.0           | 13                           |

## Installation

```sh
pip install git+https://github.com/janoPig/HROCH.git
```

## Usage

```python
from HROCH import Hroch
...
reg = Hroch()
reg.numThreads = 8
reg.stopingCriteria = 1e-3 #stop searching when r2 reach 0.999
reg.timeLimit = 5000 #5 seconds time limit
train_rms, train_r2, complexity = reg.fit(X_train, y_train)

yp = reg.predict(X_test)

test_r2 = r2_score(y_test, yp)
test_rms = np.sqrt(mean_squared_error(y_test.to_numpy(), yp))
...
```
