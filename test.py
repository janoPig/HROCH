from HROCH import Hroch
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import glob
import os


def test_sample(name, timeLimit, numThreads, stopingCriteria, verbose):
    print(name)
    f = pd.read_csv(name, sep="\t")

    # normalize feature names
    f.columns = f.columns.str.replace(r"[-.]", "_", regex=True)

    Y = pd.DataFrame(f, columns=['target'])
    X = f.drop(columns=['target'])

    reg = Hroch()
    reg.numThreads = numThreads
    reg.verbose = verbose
    reg.stopingCriteria = stopingCriteria
    reg.timeLimit = timeLimit
    reg.fit(X.to_numpy(), Y.to_numpy())

    yp = reg.predict(X.to_numpy())

    r2 = r2_score(Y, yp)
    rms = np.sqrt(mean_squared_error(Y.to_numpy(), yp))
    print(f"rms={rms}, r2={r2}")
    print(reg.get_panda_expr(X, reg.sexpr))

    return [rms, r2]


print(os.path.dirname(os.path.abspath(__file__)))


def all_samples(path, timeLimit=5000, numThreads=8, stopingCriteria=1e-9, verbose=True):
    os.chdir(path)
    cnt = 0
    fit = 0
    eq = 0  # symbolic equivalent
    r2_sum = 0.0
    for file in glob.glob("*.tsv"):
        rms, r2 = test_sample(file, timeLimit, numThreads,
                              stopingCriteria, verbose)
        if r2 < 0.0:
            r2 = 0.0
        r2_sum += r2
        cnt = cnt + 1
        if r2 > 0.999:
            fit = fit + 1
        if r2 == 1.0:
            eq = eq + 1
        r2_avg = r2_sum / cnt
        print(f"cnt={cnt}, fit={fit}, eq={eq}, r2{r2_avg}, r2_sum{r2_sum}")


all_samples(os.path.dirname(os.path.abspath(__file__)) + "/test/", 1000)
