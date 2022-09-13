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
    <br>

    > __Warning__ *If a model file exists, the following parameters are ignored*</ins></span>  

    - __--problem__ (math|simple|fuzzy, default math)
      - __math:__ All defined math symbols [simple + [sqrt, exp, log, asin, acos, sin, cos, tanh, pow]]
      - __simple:__ Restricted math to [add, mul, sq2, sub, div]
      - __fuzzy:__ [Dyadic Operators based on a Hyperbolic Paraboloid](https://commons.wikimedia.org/wiki/Fuzzy_operator#Dyadic_Operators_based_on_a_Hyperbolic_Paraboloid) [and, or, xor, impl, nand, nor, nxor, nimpl]
    - __--precision__ (f32|f64, default f32) Internal floating point representation 32 or 64 bit. Default f32.
    - __--numThreads__ (unsigned number) Number of used threads, default 8.
    - __--stoppingCriteria__ (real number) R2 error when search stop before time limit, default zero.
    - __--randomState__ (64bit unsigned integer number) Random generator seed. If zero(default) then random generator will be initialized by system time.

---

- __--help__ Print help.

---

- __--version__ Print hroch version.

---

- __--logo__ Print hroch logo.

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
