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

> **Warning** HROCH use unprotected version of math operations (eg. log or division)

## Performance

### Feynman dataset

Approximate comparison with methods tested in [srbench](https://cavalab.org/srbench/results/#results-for-ground-truth-problems).

| **Algorithm**  | **Training time (s)** | **R2 > 0.999** | **Model size** | **R2 mean** |
|:---------------|----------------------:|---------------:|---------------:|------------:|
| **MRGP**       | 14893                 | 93.1%          | 3177           | 0.9988      |
| **HROCH_1000** | 252                   | 87.4%          | 19             | 0.9951      |
| **Operon**     | 2093                  | 86.2%          | 70             | 0.9908      |
| **AIFeynman**  | 26822                 | 78.5%          | 121            | 0.9237      |
| **HROCH_100**  | 34                    | 76.5%          | 17             | 0.9922      |
| **SBP-GP**     | 28944                 | 73.7%          | 487            | 0.9946      |
| **GP-GOMEA**   | 3677                  | 71.6%          | 34             | 0.9969      |
| **HROCH_10**   | 6                     | 69.7%          | 17             | 0.9693      |
| **AFP_FE**     | 17682                 | 59.1%          | 41             | 0.9859      |
| **HROCH_1**    | <1                    | 54.6%          | 15             | 0.9237      |
| **EPLEX**      | 10599                 | 47.0%          | 56             | 0.9918      |
| **AFP**        | 2895                  | 44.8%          | 37             | 0.9685      |
| **FEAT**       | 1561                  | 39.7%          | 195            | 0.9325      |
| **gplearn**    | 3716                  | 32.8%          | 78             | 0.9010      |
| **ITEA**       | 1435                  | 27.6%          | 21             | 0.9117      |
| **DSR**        | 615                   | 25.0%          | 15             | 0.8758      |
| **BSR**        | 28800                 | 10.8%          | 25             | 0.6940      |
| **FFX**        | 19                    |  0.0%          | 268            | 0.9082      |

Notes:

- *Tested feynman dataset with noise 0*

- *HROCH was run with a 1 thread [1, 10, 100, 1000] seconds timeout limit.*

- *Because the thing was measuring R2 > 0.999 criterium the stoppingCriteria was set to 1e-5 (stops when r2 > 0.99999).*

### Reproduction of GECCO2022 competition. HROCH run 4 threads only 5 seconds per job

<https://github.com/janoPig/srbench/tree/srcomp>

**Rank**

![rank_1](https://user-images.githubusercontent.com/75015989/188947889-d609361e-ccb8-4478-8b8d-63080d01fc54.png)

**Training Time**

![time_1](https://user-images.githubusercontent.com/75015989/188948000-3d6a55f5-9ef5-42dc-9d84-a46a175b72ae.png)

**Individual task results**

![exact_1](https://user-images.githubusercontent.com/75015989/188952664-082ba4b6-a9e1-4cd5-a7df-9205953b1c97.png)
![extrapolation_1](https://user-images.githubusercontent.com/75015989/188952899-c32005d0-8409-4aaa-a137-3d77f96346dc.png)
![feature_1](https://user-images.githubusercontent.com/75015989/188953040-00d40a47-d4a6-4703-bc1f-9f11e2f3c337.png)
![localopt_1](https://user-images.githubusercontent.com/75015989/188953060-346ed0a8-e0d8-46f8-8dbe-0d2cb18d967d.png)
![noise_1](https://user-images.githubusercontent.com/75015989/188953075-a2735263-42ec-4852-9177-fb7a894a89a4.png)
