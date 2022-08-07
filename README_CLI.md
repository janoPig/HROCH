# HROCH command line inteface

Hroch is a command line symbolic regression solver. Internally it works as virtual machine...

## Parameters

- **--task** (fit|predict) mandatory
- **--inputFile** (path) mandatory
- **--outFile** (path) mandatory for predict task
- **--modelFile** (path) mandatory for fit task, if file exist parameters '--numThreads' and '--problem' partameters are loaded from model
- **--problem** (math|simple|fuzzy) default math, ignored for predict task
  -**math** default, all defined math symbols [simple + [sqrt, exp, log, asin, acos, sin, cos, tanh]]
  - **simple** restricted math to [add, mul, sq2, sub, div]
  - **fuzzy** [Dyadic Operators based on a Hyperbolic Paraboloid](https://commons.wikimedia.org/wiki/Fuzzy_operator#Dyadic_Operators_based_on_a_Hyperbolic_Paraboloid) and, or, xor, impl, nand, nor, nxor, nimpl
- **--precision** (f32|f64) default f32
- **--timeLimit** (miliseconds) default 5000
- **--numThreads** (number) default 8

## Example

Fit task, trained model saved to model.hrx file

```bash
./hroch --task fit --inputFile data_train.csv --modelFile model.hrx --precision f32 --timeLimit 5000 --numThreads 8 --problem math
```

Predict task

```bash
./hroch --task predict --inputFile  data_test.csv --modelFile model.hrx --outFile results.csv
```

Continue fit task (if model.hrx exist numThreads and problem partameters are loaded from model)

```bash
./hroch --task fit --inputFile data_train.csv --modelFile model.hrx --timeLimit 5000
```
