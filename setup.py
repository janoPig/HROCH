from setuptools import setup


ldesc = """
# HROCH  

**High-Performance c++ symbolic regression library based on parallel local search**

- Zero hyperparameter tunning.
- Accurate results in seconds or minutes, in contrast to slow GP-based methods.
- Small models size.
- Support for regression, classification and fuzzy math.
- Support 32 and 64 bit floating point arithmetic.
- Work with unprotected version of math operators (log, sqrt, division)
- Speedup search by using feature importances computed from bbox model

|**Supported instructions**||
| ----------- | ----------- |
|**math**|add, sub, mul, div, pdiv, inv, minv, sq2, pow, exp, log, sqrt, cbrt, aq|
|**goniometric**|sin, cos, tan, asin, acos, atan, sinh, cosh, tanh|
|**other**|nop, max, min, abs, floor, ceil, lt, gt, lte, gte|
|**fuzzy**|f_and, f_or, f_xor, f_impl, f_not, f_nand, f_nor, f_nxor, f_nimpl|

## Dependencies

- AVX2 instructions set(all modern CPU support this)
- numpy
- sklearn
- scipy

## Installation

```sh
pip install HROCH
```

## Usage

[Symbolic_Regression_Demo.ipynb](examples/Symbolic_Regression_Demo.ipynb)  [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/janoPig/HROCH/blob/main/examples/Symbolic_Regression_Demo.ipynb)

[Documentation](https://janopig.github.io/HROCH/HROCH.html)

```python
from HROCH import SymbolicRegressor

reg = SymbolicRegressor(num_threads=8, time_limit=60.0, problem='math', precision='f64')
reg.fit(X_train, y_train)
yp = reg.predict(X_test)
```

## Changelog

### v1.4

- sklearn compatibility
- Classificators:
  - NonlinearLogisticRegressor for a binary classification
  - SymbolicClassifier for multiclass classification
  - FuzzyRegressor for a special binary classification

### v1.3

- Public c++ sources
- Commanline interface changed to cpython
- Support for classification score logloss and accuracy
- Support for final transformations:
  - ordinal regression
  - logistic function
  - clipping
- Acess to equations from all paralel hillclimbers
- User defined constants

### v1.2

- Features probability as input parameter
- Custom instructions set
- Parallel hilclimbing parameters
  
### v1.1

- Improved late acceptance hillclimbing

### v1.0

- First release

## SRBench

![image](https://github.com/janoPig/HROCH/assets/75015989/3fa087dc-8caf-4301-86d7-4e79a4e84402)

"""

setup(
    name='HROCH',
    version='1.4.8',
    description='Symbolic regression and classification',
    long_description=ldesc,
    long_description_content_type="text/markdown",
    author='Jano',
    author_email='hroch.regression@gmail.com',
    url='https://github.com/janoPig/HROCH/',
    project_urls={
        'Documentation': 'https://janopig.github.io/HROCH/HROCH.html',
        'Source': 'https://github.com/janoPig/HROCH',
        'Tracker': 'https://github.com/janoPig/HROCH/issues',
    },

    keywords=['scikit-learn', 'symbolic-regression', 'classification', 'fuzzy-logic', 'interpretable-ml', 'machine-learning'],
    classifiers=['Operating System :: Microsoft :: Windows',
                 'Operating System :: POSIX :: Linux',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.8',
                 'Programming Language :: Python :: 3.9',
                 'Programming Language :: Python :: 3.10',
                 'Programming Language :: Python :: 3.11',
                 'Programming Language :: Python :: 3.12'],
    packages=['HROCH'],
    license='MIT',
    include_package_data=True,
    install_requires=['numpy', 'scikit-learn', 'scipy'],
)
