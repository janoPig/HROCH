# HROCH  

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/HROCH) [![PyPI version](https://badge.fury.io/py/HROCH.svg)](https://badge.fury.io/py/HROCH) [![Downloads](https://pepy.tech/badge/hroch)](https://pepy.tech/project/hroch) [![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=janoPig_HROCH&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=janoPig_HROCH) [![CodeQL](https://github.com/janoPig/HROCH/actions/workflows/codeql.yml/badge.svg)](https://github.com/janoPig/HROCH/actions/workflows/codeql.yml)

| Windows |Results|Linux|Results|
| ----------- | ----------- | ----------- | ----------- |
|[![Strogatz - windows](https://github.com/janoPig/HROCH/actions/workflows/strogatz_test_win.yml/badge.svg)](https://github.com/janoPig/HROCH/actions/workflows/strogatz_test_win.yml)|[results.csv](ci-test/results/strogatz_win.csv)|[![Strogatz - linux](https://github.com/janoPig/HROCH/actions/workflows/strogatz_test_linux.yml/badge.svg)](https://github.com/janoPig/HROCH/actions/workflows/strogatz_test_linux.yml)|[results.csv](ci-test/results/strogatz_linux.csv)|
|[![Feynman - windows](https://github.com/janoPig/HROCH/actions/workflows/feynman_test_win.yml/badge.svg)](https://github.com/janoPig/HROCH/actions/workflows/feynman_test_win.yml)|[results.csv](ci-test/results/feynman_win.csv)|[![Feynman - linux](https://github.com/janoPig/HROCH/actions/workflows/feynman_test_linux.yml/badge.svg)](https://github.com/janoPig/HROCH/actions/workflows/feynman_test_linux.yml)|[results.csv](ci-test/results/feynman_linux.csv)|

**[The fastest symbolic regression algorithm in the world.](#performance)**

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

## Performance

### SRBench

![output](https://user-images.githubusercontent.com/75015989/212184959-c462beae-d145-48ad-adc9-52aaefc2a380.png)

### Reproduction of GECCO2022 competition. HROCH run 4 threads only 5 seconds per job

<https://github.com/janoPig/srbench/tree/srcomp>

```bash
git clone https://github.com/janoPig/srbench.git
cd srbench
git checkout srcomp
#Install the conda environment
cd competition-2022
conda env create -f environment.yml
conda activate srcomp
#Install method
#To test the current version, change pip 'install git+...' to 'pip install HROCH' in /official_competitors/HROCH/install.sh
#and rename Hroch to PHCRegressor in the file /official_competitors/HROCH/regressor.py
bash install_competitors.sh HROCH
#Testing installation
bash test.sh HROCH
#Download result data for other methods
wget -O res.zip https://zenodo.org/record/6842176/files/srbench_competition_results.zip
unzip res.zip
mv ./zenodo/results_stage1 ./results_stage1
rm -r ./zenodo
rm -r ./res.zip
#Run experiment (cca 10-15min)
#For PC whitch have cpu cores < 12 set parameter -n_jobs in submit_stage1_local.sh to smaller value
#To repeat the experiment, add the --noskips parameter to overwrite the results
cd experiment
bash submit_stage1_local.sh HROCH
```

To show results run competition-2022/postprocessing/stage1.ipynb

:warning: There is an bug with the featureselection score. Test here [test_feature_selection.py](https://github.com/janoPig/srbench/blob/a29e3ec49d3eda72e67af35ac7e12711bda6fbd7/competition-2022/experiment/data/stage1/test_feature_selection.py) and possible fix here [fix_feature_selection.py](https://github.com/janoPig/srbench/blob/a29e3ec49d3eda72e67af35ac7e12711bda6fbd7/competition-2022/experiment/data/stage1/fix_feature_selection.py)

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
