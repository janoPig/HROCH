# HROCH command line inteface

HROCH is a free command line symbolic regression solver. Internally, it works as a virtual machine. The first step is to run HROCH with the fit task while generating a program or model file. The program file contains only the program with the best solution, the model file contains the entire population so that the search process can be continued next time.

> __Note__ HROCH use space delimited csv file format.

## Parameters

---

- __--task__ (fit|predict) Mandatory. Task type.
  - __predict__
    - __--x__ (path) Mandatory. Features csv file. The file must exist and contain valid data.
    - __--y__ (path) Mandatory. Target csv file. If the file exists, it will be overwritten.
    - __--modelFile or --programFile__ (path) Mandatory. File must exist and contain a valid model or program.
  - __fit__
    - __--x__ (path) Mandatory. Features csv file. The file must exist and contain valid data.
    - __--y__ (path) Mandatory. Target csv file. The file must exist and contain valid data.
    - __--modelFile or --programFile__ (path) Mandatory. If a model file exists, HROCH continues the fitting task from the stored model.
    - __--timeLimit__ (unsigned number) Timeout in milliseconds, 5000 by default.
    - __--iterLimit__ (unsigned number) Iterations limit.
    - __--problem__ (math|simple|fuzzy|custom set, default math)
      - __math:__ All defined math symbols [simple + [pow, exp, log, sqrt, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh]]
      - __simple:__ Restricted math to [nop, add, sub, mul, div, sq2]
      - __fuzzy:__ [Dyadic Operators based on a Hyperbolic Paraboloid](https://commons.wikimedia.org/wiki/Fuzzy_operator#Dyadic_Operators_based_on_a_Hyperbolic_Paraboloid) [f_and, f_or, f_xor, f_not]
      - __custom set__ (array of pairs instruction name and probability {string, real number} separated by semicolons) for example "add 1.0;sub 0.1;mul 1.0;exp 0.01"
    - __--stoppingCriteria__ (real number) R2 error when search stop before time limit, default zero.
    - __--randomState__ (64bit unsigned integer number) Random generator seed. If zero(default) then random generator will be initialized by system time.
    - __--featProbs__ (array of real numbers separated by semicolons) for example "1.0;0.1;1.0;0.01"
    <br>

    > __Warning__ *If a model file exists, the following parameters are ignored*</ins></span>  

    - __--precision__ (f32|f64, default f32) Internal floating point representation 32 or 64 bit. Default f32.
    - __--numThreads__ (unsigned number) Number of used threads, default 8.
    - __--popSize__ (unsigned number) Population size.
    - __--popSel__ (unsigned number) Selection mode(tournament).
    - __--constSize__ (unsigned number) Max used constants.
    - __--codeSize__ (unsigned number) Max code size.
    

---

- __--help__ Print help.

---

- __--version__ Print hroch version.

---

- __--logo__ Print hroch logo.

---
---

|**supported instructions**||
| ----------- | ----------- |
|**math**|add, sub, mul, div, inv, minv, sq2, pow, exp, log, sqrt, cbrt, aq|
|**goniometric**|sin, cos, tan, asin, acos, atan, sinh, cosh, tanh|
|**other**|nop, max, min, abs, floor, ceil, lt, gt, lte, gte|
|**fuzzy**|f_and, f_or, f_xor, f_impl, f_not, f_nand, f_nor, f_nxor, f_nimpl|

---

## Example

Run the fit task for 5 seconds, save the trained model to the model.hrx file.

```bash
./hroch.bin --task fit --x feature_train.csv --y target_train.csv --modelFile model.hrx --precision f32 --timeLimit 5000 --numThreads 8 --problem math
```

Run the predict task, save computed output to the target_test_predicted.csv file.

```bash
./hroch.bin --task predict --x feature_test.csv --y target_test_predicted.csv --modelFile model.hrx
```

Continue fit task for the next 5 minutes.

```bash
./hroch.bin --task fit --x feature_train.csv --y target_train.csv --modelFile model.hrx --timeLimit 300000
```
