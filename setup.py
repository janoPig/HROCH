from setuptools import setup


ldesc = """
# HROCH  

**High-Performance python symbolic regression library based on parallel late acceptance hill-climbing**

- Zero hyperparameter tunning.
- Accurate results in seconds or minutes, in contrast to slow GP-based methods.
- Small models size.
- Support mathematic equations and fuzzy logic operators.
- Support 32 and 64 bit floating point arithmetic.
- Work with unprotected version of math operators (log, sqrt, division)
- Speedup search by using feature importances computed from bbox model
- CLI

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

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/janoPig/HROCH/blob/main/examples/Symbolic_Regression_Demo.ipynb)

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


## SRBench


![image](https://user-images.githubusercontent.com/75015989/212561560-39393068-8d72-48f4-b11c-7a14db029faf.png)

"""

setup(
    name='HROCH',
    version='1.2.1',
    description='Symbolic regression',
    long_description=ldesc,
    long_description_content_type="text/markdown",
    author='Jano',
    author_email='hroch.regression@gmail.com',
    url='https://github.com/janoPig/HROCH/',
    project_urls={
        'Documentation': 'https://github.com/janoPig/HROCH/tree/main/docs',
        'Source': 'https://github.com/janoPig/HROCH',
        'Tracker': 'https://github.com/janoPig/HROCH/issues',
    },

    keywords=['machine-learning', 'numpy', 'symbolic-regression', 'fuzzy'],
    classifiers=['Operating System :: Microsoft :: Windows',
                 'Operating System :: POSIX :: Linux',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 'Programming Language :: Python :: 3.9',
                 'Programming Language :: Python :: 3.10'],
    packages=['HROCH'],
    license='MIT',
    include_package_data=True,
    install_requires=['numpy'],
)
