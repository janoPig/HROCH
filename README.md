# HROCH  

**[The fastest symbolic regression algorithm in the world.](#performance)**

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

Python wraper(a sklearn-compatible Regressor) for [CLI](README_CLI.md).
Hroch support mathematic equations and fuzzy logic operators.
Zero hyperparameter tunning.

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

Floating precision can be set to 32 or 64 bit.

```precision='f64|f32'```

The search space is governed by the "problem" parameter. To solve fuzzy equations, it is recommended that all values in the dataset be in the range [0,0, 1,0], where 0,0 means exactly False and 1,0 means exactly True.

```problem='math|simple|fuzzy'```

- "simple" [add, mul, sq2, sub, div]
- "math" simple + [sqrt, exp, log, asin, acos, sin, cos, tanh, pow]
- "fuzzy" [Dyadic Operators based on a Hyperbolic Paraboloid](https://commons.wikimedia.org/wiki/Fuzzy_operator#Dyadic_Operators_based_on_a_Hyperbolic_Paraboloid) [and, or, xor, impl, nand, nor, nxor, nimpl]

> __Warning__ HROCH use unprotected version of math operations (eg. log or division)

## Performance

Reproduction of GECCO2022 competition. HROCH run 4 threads only 5 seconds per job.
https://github.com/janoPig/srbench/tree/srcomp

**Rank**

![rank_1](https://user-images.githubusercontent.com/75015989/188947889-d609361e-ccb8-4478-8b8d-63080d01fc54.png)

**Time**

![time_1](https://user-images.githubusercontent.com/75015989/188948000-3d6a55f5-9ef5-42dc-9d84-a46a175b72ae.png)

**Individual task results**

![exact_1](https://user-images.githubusercontent.com/75015989/188952664-082ba4b6-a9e1-4cd5-a7df-9205953b1c97.png)
![extrapolation_1](https://user-images.githubusercontent.com/75015989/188952899-c32005d0-8409-4aaa-a137-3d77f96346dc.png)
![feature_1](https://user-images.githubusercontent.com/75015989/188953040-00d40a47-d4a6-4703-bc1f-9f11e2f3c337.png)
![localopt_1](https://user-images.githubusercontent.com/75015989/188953060-346ed0a8-e0d8-46f8-8dbe-0d2cb18d967d.png)
![noise_1](https://user-images.githubusercontent.com/75015989/188953075-a2735263-42ec-4852-9177-fb7a894a89a4.png)
