from hroch import Hroch
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import glob
import os


def test_sample(name):
    print(name)
    f = pd.read_csv(name, sep="\t")

    #f = f.add_prefix('_')
    f.columns = f.columns.str.replace(r"[-.]", "_", regex=True)

    Y = pd.DataFrame(f, columns=['target'])
    X = f.drop(columns=['target'])

    reg = Hroch()
    reg.fit(X, Y)

    yp = reg.predict(X)

    r2 = r2_score(Y, yp)
    rms = np.sqrt(mean_squared_error(Y, yp))
    print(f"rms={rms}, RMSE={reg.RMSE}, r2={r2}")

    print(reg.expr)

    return [rms, r2]


def all_samples(path):
    os.chdir(path)
    cnt = 0
    fit = 0
    r2_sum = 0.0
    for file in glob.glob("*.tsv"):
        rms, r2 = test_sample(file)
        if r2 < 0.0:
            r2 = 0.0
        r2_sum += r2
        cnt = cnt + 1
        if r2 > 0.999:
            fit = fit + 1
        r2_avg = r2_sum / cnt
        print(f"cnt={cnt}, fit={fit}, r2{r2_avg}, r2_sum{r2_sum}")


all_samples("./test/")
