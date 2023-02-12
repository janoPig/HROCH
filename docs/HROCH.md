# PHCRegressor

Symbolic Regressor based parallel hill climbing algorithm with late acceptance. Unlike genetic programming, no crossover operation is performed. Thus, each individual is independent of the others. Several techniques avoid getting stuck in the local minimum. Parallelism. More parallel runs means a better chance of reaching the global minimum, even if it means a slower run. Late acceptance, which allows for some degradation steps. Mutation in 3 steps. Each individual is mutated multiple times during an iteration, regardless of its fitness value, allowing some hurdles to be skipped.

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
- __stopping_criteria : float, default=0.0__

    Error when search stop before time limit or iter_limit. Exactly it mean $1.0 - R^2$ value. stopping_criteria = 0.001 stops the serch when is found solution with score better as $R^2 = 0.999$

### __Controlling search space__

- __precision : str, defaul='f32'__

    'f64' or 'f32'. Internal floating number representation. 32bit AVX2 instructions are 2x faster as 64bit.
- __problem : str or set of instructions, default='math'__

    Predefined instructions sets 'mat' or 'simple' or 'fuzzy' or custom defines set of instructions with mutation probability.

  - __Predefined instructions sets:__
    - __'math':__ [simple + [pow, exp, log, sqrt, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh]]
    - __'simple:'__ Restricted math to [nop, add, sub, mul, div, sq2]
    - __'fuzzy':__: [Dyadic Operators based on a Hyperbolic Paraboloid](https://commons.wikimedia.org/wiki/Fuzzy_operator#Dyadic_Operators_based_on_a_Hyperbolic_Paraboloid) [f_and, f_or, f_xor, f_not]

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
    |__math__|add, sub, mul, div, inv, minv, sq2, pow, exp, log, sqrt, cbrt, aq|
    |__goniometric__|sin, cos, tan, asin, acos, atan, sinh, cosh, tanh|
    |__other__|nop, max, min, abs, floor, ceil, lt, gt, lte, gte|
    |__fuzzy__|f_and, f_or, f_xor, f_impl, f_not, f_nand, f_nor, f_nxor, f_nimpl|

    *_nop - no operation_

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
- __code_size: int, default=32__

   Each symbolic model in PHCRegressor is internally represented as a linear code. For example $x_1*(x_2 + 0.1)^2$  can be represented as a linear program with code:

    ```python
    x4 = x2 + 0.1
    x5 = x4**2
    y = x1*x5
    ```

    code_size parameter controls the maximum allowed size of such a linear program.

### __Other__

- __random_state: int, default=0__

    Random generator seed. If 0 then random generator will be initialized by system time.
- __save_model: bool, default=False__

    Save whole search model. Allow continue fit task.
- __verbose: bool, default=False__

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

  - __X__ Training data.

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

---

## Demo

<a href="https://colab.research.google.com/github/janoPig/HROCH/blob/main/examples/Symbolic_Regression_Demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

[![!Symbolic Regression Demo](https://img.shields.io/badge/kaggle-Symbolic%20Regression%20Demo-035a7d)](https://www.kaggle.com/code/jano123/symbolic-regression-demo)
