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

|**supported instructions**||
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

```python
from HROCH import PHCRegressor

reg = PHCRegressor(numThreads=8, timeLimit=60.0, problem='math', precision='f64')
reg.fit(X_train, y_train)
yp = reg.predict(X_test)
```

## Performance

### Feynman dataset

Approximate comparison with methods tested in [srbench](https://cavalab.org/srbench/results/#results-for-ground-truth-problems).

|Algorithm|Training time (s)|Model size|R2 > 0.999|R2 > 0.999999|R2 > 0.999999999|R2 mean          |
|---------|----------------:|---------:|:--------:|:-----------:|:--------------:|:---------------:|
|MRGP     |14893            |3177      |0.931     |0.000        |0.000           |0.998853549755939|
|Operon   |2093             |70        |0.862     |0.655        |0.392           |0.990832974928022|
|AIFeynman|26822            |121       |0.785     |0.689        |0.680           |0.923670858619585|
|**HROCH_100**|**42**       |**17**    |**0.781** |**0.679**    |**0.633**       |**0.988862822072670**|
|SBP-GP   |28944            |487       |0.737     |0.388        |0.246           |0.994645420032544|
|GP-GOMEA |3677             |34        |0.716     |0.539        |0.504           |0.996850949284431|
|AFP_FE   |17682            |41        |0.591     |0.315        |0.185           |0.985876419645066|
|**HROCH_1**|**2**          |**14**    |**0.544** |**0.432**    |**0.421**       |**0.911182785072874**|
|EPLEX    |10599            |56        |0.470     |0.121        |0.082           |0.991763792716299|
|AFP      |2895             |37        |0.448     |0.263        |0.159           |0.968488776363814|
|FEAT     |1561             |195       |0.397     |0.121        |0.112           |0.932465581448533|
|gplearn  |3716             |78        |0.328     |0.151        |0.151           |0.901020570640627|
|ITEA     |1435             |21        |0.276     |0.233        |0.224           |0.911713461958873|
|DSR      |615              |15        |0.250     |0.207        |0.207           |0.875784840006460|
|BSR      |28800            |25        |0.108     |0.073        |0.043           |0.693995349495648|
|FFX      |19               |268       |0.000     |0.000        |0.000           |0.908164756903951|

Notes:

- *Tested feynman dataset with noise 0*, 10 trials for dataset, 1 thread.

- *HROCH_1 use 1 second timeout limit, HROCH_100 100 seconds.*

- *HROCH stoppingCriteria was set to 1e-12 (stops when r2 > 0.999999999999).*

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
