from HROCH import Hroch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import glob
import os

import statistics

target_noise = 0.0


def test_sample(name, timeLimit, numThreads, stoppingCriteria, verbose):
    print(name)
    f = pd.read_csv(name, sep="\t")

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

    reg = Hroch(numThreads, timeLimit, stoppingCriteria,
                "f32", "math", verbose=verbose)
    reg.fit(X_train.to_numpy(), y_train)
    yp = reg.predict(X_test)

    r2 = r2_score(y_test, yp)
    rms = np.sqrt(mean_squared_error(y_test.to_numpy(), yp))
    print(f"test: rms={rms}, r2={r2}")
    print(reg.sexpr)

    return [rms, r2]


def all_samples(path: str, timeLimit: float, numThreads: int, stopingCriteria: float, verbose: bool = False):
    os.chdir(path)
    cnt = 0
    fit = 0
    eq = 0  # symbolic equivalent
    eq2 = 0  # symbolic equivalent
    r2_sum = 0.0
    r2arr = []
    for file in glob.glob("*.tsv"):
        rms, r2 = test_sample(file, timeLimit, numThreads,
                              stopingCriteria, verbose)

        r2_sum += r2
        cnt = cnt + 1
        if r2 > 0.999:
            fit = fit + 1

        if r2 >= 1.0-stopingCriteria:
            eq = eq + 1
        if r2 >= 1.0:
            eq2 = eq2 + 1
        r2arr.append(r2)
        r2_avg = r2_sum / cnt
        print(
            f"cnt={cnt}, fit={fit}, eq={eq}, eq2={eq2}, r2={r2_avg}, r2_sum={r2_sum} med={statistics.median(r2arr)}")


all_samples(os.path.dirname(os.path.abspath(
    __file__)) + "/test/", numThreads=8, timeLimit=5.0,  stopingCriteria=0.0, verbose=False)
