# PHCRegressor

Symbolic Regressor based on parallel hill climbing algorithm. Unlike genetic programming, no crossover operation is performed. Thus, each individual is independent of the others.
---

## __Parameters__

### __Controlling used resources__

- __num_threads : int, default=8__
  
    Number of used threads. Each thread run his own population. More threads mean a better chance of getting a good result.
- __pop_size: int, default=64__

    Number of individuals in the population. More individuals means a more robust but slower search.
- __pop_sel: int, default=4__

    In each iteration, an individual is selected in a tournament fashion to be mutated. pop_sel = 4 means that 4 random individuals are selected and the one with the best score is chosen.
  
### __Controlling running time__

The algorithm stops if any stopping condition is met. It can solve many cases in a very short time from a few seconds to minutes, but symbolic regression is NP-hard, and there are certainly cases where it needs to run for a really long time.

- __time_limit : float, default=5.0__

    Timeout in seconds. If is set to 0 there is no limit and the algorithm runs until some other condition is met.
- __iter_limit : int, default=0__

    Iterations limit. If is set to 0 there is no limit and the algorithm runs until some other condition is met.

### __Controlling search space__

- __precision : str, defaul='f32'__

    'f64' or 'f32'. Internal floating number representation. 32bit AVX2 instructions are 2x faster as 64bit.
- __problem : str or set of instructions, default='math'__

    Predefined instructions sets 'mat' or 'simple' or 'fuzzy' or custom defines set of instructions with mutation probability.

  - __Predefined instructions sets:__
    - __'math':__ [simple + [pow, exp, log, sqrt, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh]]
    - __'simple:'__ Restricted math to [nop, add, sub, mul, div, sq2]
    - __'fuzzy':__: [Dyadic Operators based on a Hyperbolic Paraboloid](https://commons.wikimedia.org/wiki/Fuzzy_operator#Dyadic_Operators_based_on_a_Hyperbolic_Paraboloid) [f_and, f_or, f_xor, f_not]

    ```python
    simple = {'nop': 0.01, 'add': 1.0, 'sub': 1.0,
          'mul': 1.0, 'div': 0.1, 'sq2': 0.05}

    math = {'nop': 0.01, 'add': 1.0, 'sub': 1.0, 'mul': 1.0,
            'div': 0.1, 'sq2': 0.05, 'pow': 0.001, 'exp': 0.001,
            'log': 0.001, 'sqrt': 0.1, 'sin': 0.005, 'cos': 0.005,
            'tan': 0.001, 'asin': 0.001, 'acos': 0.001, 'atan': 0.001,
            'sinh': 0.001, 'cosh': 0.001, 'tanh': 0.001}

    fuzzy = {'nop': 0.01, 'f_and': 1.0, 'f_or': 1.0, 'f_xor': 1.0, 'f_not': 1.0}
    ```

  - __Custom set of instructions with mutation probability.__
  
     Probability is calculated as the given value divided by the sum of all probabilities.

     ```python
     # add, mul probability 10/23, lt, gt, nop probablity 1/23
    instr_set = {'add':10.0, 'mul':10.0, 'gt':1.0, 'lt':1.0, 'nop':1.0}

    reg = PHCRegressor(problem=instr_set)
    reg.fit(train_X, train_y)
    ```

    |__supported instructions__||
    | ----------- | ----------- |
    |__math__|add, sub, mul, div, pdiv, inv, minv, sq2, pow, exp, log, sqrt, cbrt, aq|
    |__goniometric__|sin, cos, tan, asin, acos, atan, sinh, cosh, tanh|
    |__other__|nop, max, min, abs, floor, ceil, lt, gt, lte, gte|
    |__fuzzy__|f_and, f_or, f_xor, f_impl, f_not, f_nand, f_nor, f_nxor, f_nimpl|

    *_nop - no operation_

    *_pdiv - protested division_

    *_inv - inverse_ $(-x)$

    *_minv - multiplicative inverse_ $(1/x)$

    *_lt, gt, lte, gte - <, >, <=, >=_

- __feature_probs : any, default=None__

    The probability that a mutation will select a feature. If we have a equation, say $x_1 + 0.1$, and the mutation wants to change the constant $0.1$, it can choose another constant, or it can change it to a variable. If the probabilities defined in function_probs are [2, 3, 5], this means that there is a 20% probability that the mutation operator will choose $x_1$ and mutate $x_1 + 0.1$ to $x_1 + x_1$, a 30% probability for $x_1 + x_2$, and a 50% probability for $x_1 + x_3$.

    ```python
    # x1 probability 1.0/2.01, x2 probablity 1.0/2.01, x3 probablity 0.01/2.01
    probs=[1.0,1.0, 0.01]

    reg = PHCRegressor(feature_probs=fprobs)
    reg.fit(train_X, train_y)
    ```

- __const_size: int, default=8__

    Maximum alloved constants in symbolic model, accept also 0.
- __code_min_size, code_max_size: int, default=32__

   Each symbolic model in PHCRegressor is internally represented as a linear code. For example $x_1*(x_2 + 0.1)^2$  can be represented as a linear program with code:

    ```python
    x4 = x2 + 0.1
    x5 = x4**2
    y = x1*x5
    ```

    code_min_size and code_max_size parameters controls the minimum/maximum allowed size of such a linear program.

- __init_const_min: float, default=-1.0__
  
    Lower range for initializing constants.
- __init_const_max: float, default=1.0__
  
    Upper range for initializing constants.
- __init_predefined_const_prob: float, default=0.0__

    Probability of selecting one of the predefined constants during initialization.
- __init_predefined_const_set: list of floats, default=[]__

    Predefined constants used during initialization.
- __clip_min: float, default=0.0__

    Lower limit for calculated values. If both values (clip_min and clip_max) are the same, then no clip is performed.
- __clip_max: float, default=0.0__

    Upper limit for calculated values. If both values (clip_min and clip_max) are the same, then no clip is performed.
- __const_min: float, default=-1e30__

    Lower bound for constants used in generated equations.
- __const_max: float, default=1e30__
  
    Upper bound for constants used in generated equations.
- __predefined_const_prob: float, default=0.0__

    Probability of selecting one of the predefined constants during equations search.
- __predefined_const_set: list of floats, default=[]__

    Predefined constants used during equations search.

### __Other__

- __metric: str, default='MSE'__

    Metric used for evaluating error. Choose from {'MSE', 'MAE', 'MSLE', 'LogLoss'}
- __transformation: str, default=None__

    Final transformation for computed value. Choose from { None, 'LOGISTIC', 'PSEUDOLOG', 'ORDINAL'}
- __random_state: int, default=0__

    Random generator seed. If 0 then random generator will be initialized by system time.
- __verbose: int, default=0__

    Controls the verbosity when fitting and predicting.

---

## Attributes

- __sexpr: string__

    Resulting symbolic model as expression. Features are named $x_1, x_2, ... x_n$. For mathematical problems is recomended use some simplifications with sympy or similar library.

    Example of getting mathematical expression with feature names from pandas DataFrame

    ```python
    def get_eq(X : pd.DataFrame, expr : str):
        model_str = str(sp.parse_expr(expr))
        #mapping = {'x'+str(i+1): k for i, k in enumerate(X.columns)}
        mapping = OrderedDict({'x'+str(i+1): k for i, k in enumerate(X.columns)})
        new_model = model_str
        for k, v in reversed(mapping.items()):
            new_model = new_model.replace(k, v)

        return new_model

    reg = PHCRegressor()
    reg.fit(X_train, y_train)
    print(f'eq: {get_eq(X, reg.sexpr)}')
    ```

---

## Methods

- __fit(X: np.ndarray, y: np.ndarray)__

    Fit symbolic model.

    __Parameters:__

  - __X__ Training data.

  - __y__ Target values.

- __predict(X: np.ndarray):__

    Predict using the symbolic model.

    __Parameters:__

  - __X__ Samples.
  - __id__ (int) Hillclimber id, default=None. id can be obtained from get_models method. If its none prediction use best hillclimber.

    __Returns:__ Returns predicted values.

- __score(X: np.ndarray, y: np.ndarray)__

    Return the coefficient of determination of the prediction.

    The coefficient of determination $R^2$ is defined as $(1-u/v)$, where $u$ is the residual sum of squares ```((y_true - y_pred)**2).sum()``` and is the total sum of squares ```((y_true - y_true.mean())** 2).sum()```. The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always predicts the expected value of y, disregarding the input features, would get a score of 0.0.

    __Parameters:__
    __X__ Test samples.
    __y__ True values for X.

    __Returns:__
    $R^2$ of ```self.predict(X)``` wrt. ```y```.
- __get_params()__

    Get parameters for this estimator.
- __set_params(**params)__

    Set the parameters of this estimator.

- __get_models()__

    Return list of equations from all paralel hillclimbers.

---

## Demo

<a href="https://colab.research.google.com/github/janoPig/HROCH/blob/main/examples/Symbolic_Regression_Demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

[![!Symbolic Regression Demo](https://img.shields.io/badge/kaggle-Symbolic%20Regression%20Demo-035a7d)](https://www.kaggle.com/code/jano123/symbolic-regression-demo)
