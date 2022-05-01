# HROCH

```
      c~~p ,---------.
 ,---'oo  )           \
( O O                  )/
 `=^='                 /
       \    ,     .   /
       \\  |-----'|  /
       ||__|    |_|__|
```

  Simple and fast hillclimb algorithm for symbolic regression.
  Python wraper(a sklearn-compatible Regressor) for CLI(command line interface)

  CLI builded on Ubuntu 20.04 with g++-9
  Tested on Ubuntu 20.04 and Pop!_OS 22.04

  Requirements:

- AVX2 instructions set(all modern CPU support this)
- pandas
- sympy
- numpy

Installation:

```sh
pip install git+https://github.com/janoPig/HROCH.git
```

Usage:

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
