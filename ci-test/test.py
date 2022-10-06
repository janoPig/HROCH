from HROCH import PHCRegressor
import numpy as np
import pandas as pd
import sympy as sp
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import os


target_noise = 0.0

true_eq = {'strogatz_bacres1': '20 - x - (x * y)/(1+0.5 * x**2)',
           'strogatz_bacres2': '10 - (x * y)/(1+0.5 * x**2)',
           'strogatz_barmag1': '0.5 * sin(x - y) - sin(x)',
           'strogatz_barmag2': '0.5 * sin(y - x) - sin(y)',
           'strogatz_glider1': '-0.05 * x**2 - sin(y)',
           'strogatz_glider2': 'x - cos(y)/x',
           'strogatz_lv1': '3  * x - 2  * x * y - x**2',
           'strogatz_lv2': '2 * y - x * y - y**2',
           'strogatz_predprey1': 'x  * ( 4 - x - (y)/(1+x) )',
           'strogatz_predprey2': 'y * ( (x)/(1+x) - 0.075 * y )',
           'strogatz_shearflow1': 'cot(y) * cos(x)',
           'strogatz_shearflow2': '(cos(y)**2 + 0.1 *  sin(y)**2) * sin(x)',
           'strogatz_vdp1': '10 *  (y - (1)/(3) * (x**3-x))',
           'strogatz_vdp2': '-(1)/(10) * x'}


def test_sample(name, timeLimit, numThreads, stoppingCriteria, verbose):
    f = pd.read_csv(name, sep="\t", compression='gzip')
    # normalize feature names
    f.columns = f.columns.str.replace(r"[-.]", "_", regex=True)

    Y = pd.DataFrame(f, columns=['target'])
    X = f.drop(columns=['target'])

    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        train_size=0.75,
                                                        test_size=0.25, random_state=42)

    y_train = y_train.to_numpy()

    if target_noise > 0:
        noise = np.random.normal(0,
                                 target_noise *
                                 np.sqrt(
                                     np.mean(np.square(y_train))),
                                 size=[len(y_train), 1])
        y_train = y_train + noise

    reg = PHCRegressor(numThreads, timeLimit, stoppingCriteria,
                       "f32", "math", verbose=verbose)
    reg.fit(X_train.to_numpy(), y_train)
    yp = reg.predict(X_test)

    r2 = r2_score(y_test, yp)
    rms = np.sqrt(mean_squared_error(y_test.to_numpy(), yp))
    # get model string
    model_str = str(reg.sexpr)
    mapping = {'x'+str(i+1): k for i, k in enumerate(X.columns)}
    new_model = model_str
    for k, v in reversed(mapping.items()):
        new_model = new_model.replace(k, v)

    return [rms, r2, sp.parse_expr(new_model)]


def all_samples(path: str, timeLimit: float, numThreads: int, stopingCriteria: float, verbose: bool = False):
    data_dir = os.path.join(path, 'data')
    out_dir = os.path.join(path, 'results')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    results = pd.DataFrame(
        columns=['name', 'r2', 'rms', 'true equation', 'found equation'])
    idx = 0
    fp = os.path.dirname(data_dir)
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            suffix = '.tsv.gz'
            if file.endswith(suffix):
                name = file[:-len(suffix)]
                full_path = os.path.join(root, file)
                rms, r2, eq = test_sample(full_path, timeLimit, numThreads,
                                          stopingCriteria, verbose)
                results.loc[idx] = [name, r2, rms, true_eq[name], str(eq)]
                idx = idx + 1
                print(
                    f"{file}: r2={r2}, rms={rms} eq={str(eq)}")

    results.sort_values(by=['name'], inplace=True)
    csv_path = out_dir + '/results.csv'
    print('save to ', csv_path)
    results.to_csv(csv_path, index=False)


all_samples(os.path.dirname(os.path.abspath(__file__)), numThreads=1, timeLimit=5.0,
            stopingCriteria=0.0, verbose=False)
