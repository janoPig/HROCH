from setuptools import setup


ldesc = """
# HROCH  

**High optimized symbolic regression.**

- Zero hyperparameter tunning.
- Accurate results in seconds or minutes, in contrast to slow GP-based methods.
- Small models size.
- Good results with noise data.
- Support mathematic equations and fuzzy logic operators.
- Support 32 and 64 bit floating point arithmetic.
- Work with unprotected version of math operators (log, sqrt, division)
- CLI

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
# print symbolic expression
print(reg.sexpr)
```

## Performance

### Feynman dataset

Approximate comparison with methods tested in [srbench](https://cavalab.org/srbench/results/#results-for-ground-truth-problems).

|Algorithm|Training time (s)|Model size|R2 > 0.999|R2 > 0.999999|R2 > 0.999999999|R2 mean          |
|---------|----------------:|---------:|:--------:|:-----------:|:--------------:|:---------------:|
|MRGP     |14893            |3177      |0.931     |0.000        |0.000           |0.998853549755939|
|Operon   |2093             |70        |0.862     |0.655        |0.392           |0.990832974928022|
|AIFeynman|26822            |121       |0.785     |0.689        |0.680           |0.923670858619585|
|__HROCH__|__42__               |__17__        |__0.781__     |__0.679__        |__0.633__           |__0.988862822072670__|
|SBP-GP   |28944            |487       |0.737     |0.388        |0.246           |0.994645420032544|
|GP-GOMEA |3677             |34        |0.716     |0.539        |0.504           |0.996850949284431|
|AFP_FE   |17682            |41        |0.591     |0.315        |0.185           |0.985876419645066|
|EPLEX    |10599            |56        |0.470     |0.121        |0.082           |0.991763792716299|
|AFP      |2895             |37        |0.448     |0.263        |0.159           |0.968488776363814|
|FEAT     |1561             |195       |0.397     |0.121        |0.112           |0.932465581448533|
|gplearn  |3716             |78        |0.328     |0.151        |0.151           |0.901020570640627|
|ITEA     |1435             |21        |0.276     |0.233        |0.224           |0.911713461958873|
|DSR      |615              |15        |0.250     |0.207        |0.207           |0.875784840006460|
|BSR      |28800            |25        |0.108     |0.073        |0.043           |0.693995349495648|
|FFX      |19               |268       |0.000     |0.000        |0.000           |0.908164756903951|

"""

setup(
    name='HROCH',
    version='1.0.11',
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
    setup_requires=['setuptools_scm'],
    include_package_data=True,
    install_requires=['numpy'],
)
