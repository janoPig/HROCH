# HROCH  

**The fastest symbolic regression algorithm in the world.**

Support mathematic equations and fuzzy logic operators

---

```txt
  Hrochy určite nemajú choboty.
      C~~P ,---------.
 ,---'oo  )           \
( O O                  )/
 `=^='                 /
       \    ,     .   /
       \\  |-----'|  /
       ||__|    |_|__|
```

---

  Python wraper(a sklearn-compatible Regressor) for [CLI](README_CLI.md)

  Zero hyperparameter tunning. Only parameters to set are time limit and r2 error as stopping criterium.

  Support simple mode [add, mul, sq2, sub, div]
  Search space: add, mul, sq2, sub, div, sqrt, exp, log, asin, acos, sin, cos, tanh, pow

  <span style="color:red"> HROCH use unprotected version of math operations (eg. log or division)</span>

## Dependencies

- AVX2 instructions set(all modern CPU support this)
- numpy

## Installation

```sh
pip install git+https://github.com/janoPig/HROCH.git
```

## Usage

```python
from HROCH import Hroch
...

reg = Hroch(numThreads=8, timeLimit=60.0, problem='math', precision='f64')

reg.fit(X_train, y_train)
yp = reg.predict(X_test)

test_r2 = r2_score(y_test, yp)
test_rms = np.sqrt(mean_squared_error(y_test, yp))
...
```

Floaing precision can be set to 32 or 64 bit.

```precision='f64|f32'```

The search space is governed by the "problem" parameter. To solve fuzzy equations, it is recommended that all values in the dataset be in the range [0,0, 1,0], where 0,0 means exactly False and 1,0 means exactly True.

```problem='math|simple|fuzzy'```

- "simple" [add, mul, sq2, sub, div]
- "math" simple + [sqrt, exp, log, asin, acos, sin, cos, tanh, pow]
- "fuzzy" [Dyadic Operators based on a Hyperbolic Paraboloid](https://commons.wikimedia.org/wiki/Fuzzy_operator#Dyadic_Operators_based_on_a_Hyperbolic_Paraboloid) [and, or, xor, impl, nand, nor, nxor, nimpl]
  