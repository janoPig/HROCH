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
  Python wraper(a sklearn-compatible Regressor) for CLI

  Zero hyperparameter tunning. Only parameters to set are time limit and r2 error as stopping criterium.

  *CLI builded on Ubuntu 20.04 with g++-9
  Tested on Ubuntu 20.04 and Pop!_OS 22.04*

  Search space: add, mul, sq2, sub, div, sqrt, exp, log, asin, acos, sin, cos

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
| **0**            | 72.3%          | 51.3%        | 0.979664    | 1.0           | 16.1                         |
| **0.001**        | 71.4%          | 36.1%        | 0.978254    | 1.0           | 15.3                         |
| **0.01**         | 68.9%          | 31,1%        | 0.980839    | 1.0           | 14.1                         |
| **0.1**          | 59.7%          | 26.9%        | 0.976934    | 0.999995      | 12.8                         |

**5 minutes** time limit, 8 threads, AMD Ryzen 5 1600

| **target noise** | **r2 > 0.999** | **r2 = 1.0** | **r2 mean** | **r2 median** | **average model complexity** |
|:----------------:|:--------------:|:------------:|:-----------:|:-------------:|:----------------------------:|
| **0**            | 90.8%          | 69.7%        | 0.999434    | 1.0           | 19.1                         |
| **0.001**        |                |              |             |               |                              |
| **0.01**         |                |              |             |               |                              |
| **0.1**          |                |              |             |               |                              |

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
