from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import os
from tempfile import TemporaryDirectory
import subprocess
import numpy as np
import pandas as pd
from sympy.parsing.sympy_parser import parse_expr
import sympy as sy
from math import *
import sys
from sympy import preorder_traversal
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split


def complexity(expr):
    c = 0
    for arg in preorder_traversal(expr):
        c += 1
    return c


class Hroch(BaseEstimator, RegressorMixin):

    def __init__(self, random_state=-1):
        self.random_state = random_state
        self.r2 = -99999999999999999.0

    def fit(self, X_train, y_train, timeLimit=5000, numThreads=8, verbose=False):
        with TemporaryDirectory() as temp_dir:
            fname = temp_dir + "/tmpdata.csv"

            data_merge = X_train.join(y_train)
            # randomize rows order
            data_merge = data_merge.sample(frac=1).reset_index(drop=True)
            data_merge.to_csv(fname, sep=" ", index=False)

            cwd = os.path.dirname(os.path.realpath(__file__))

            process = subprocess.Popen(
                ["bin/hroch", f"{fname}", "", f"{timeLimit}", f"{numThreads}"], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True)
            for line in iter(process.stdout.readline, b''):
                ss = line.decode("utf-8").split(";")

                try:
                    sexpr = sy.sympify(ss[0], evaluate=True)
                    expr = self.get_panda_expr(X_train, sexpr)
                    cplx = complexity(sexpr)

                    # hroch-cli compute with 32-bit float, and isn't guaranteed thah use all samples
                    # we fit data to obtain correct error
                    yp = self.predict(X_train, sexpr)
                    r2 = r2_score(y_train, yp)
                    rms = np.sqrt(mean_squared_error(y_train, yp))

                    if r2 > self.r2:
                        self.r2 = r2
                        self.is_fitted_ = True
                        self.sexpr = sexpr
                        self.cplx = cplx
                        self.streq = ss[0]  # debug
                        self.RMSE = sqrt(float(ss[2]))  # debug
                        self.expr = self.get_panda_expr(X_train, sexpr)

                        if verbose:
                            print(
                                f"[{ss[1]}] rms={rms},r2={self.r2},cplx={self.cplx}, {self.expr}")
                except:
                    print("some error!")
                if self.r2 and self.r2 >= 0.9999:
                    break

            process.stdout.close()
            process.wait()

        return self

    def get_panda_expr(self, X, sexpr):
        seq = str(sexpr)
        seq = seq.replace('asin', "arcsin")
        seq = seq.replace('acos', "arccos")
        seq = seq.replace('E', "2.718281828459045")
        seq = seq.replace('pi', "3.14159265359")
        seq = seq.replace('nan', "1.0")
        seq = seq.replace('I', "1.0")

        mapping = {'x_'+str(i): k for i, k in enumerate(X.columns)}
        new_model = seq
        for k, v in reversed(mapping.items()):
            new_model = new_model.replace(k, v)

        return new_model

    def eval_expr(self, X, sexpr=None):
        if sexpr == None:
            sexpr = self.sexpr
        Z = pd.DataFrame(index=range(X.shape[0]), columns=range(1)).fillna(0.0)
        # if self.sexpr.is_Number:
        #   Z.loc[:, 0] = float(str(self.sexpr))
        if len(sexpr.free_symbols) == 0:
            seq = str(sexpr)
            seq = seq.replace('E', "2.718281828459045")
            seq = seq.replace('pi', "3.14159265359")
            seq = seq.replace('nan', "1.0")
            seq = seq.replace('I', "1.0")

            Z.loc[:, 0] = eval(seq)  # i hope this work
        else:
            expr = self.get_panda_expr(X, sexpr)
            Z = X.eval(expr)

        return Z

    def predict(self, X_test, sexpr=None, ic=None):
        if sexpr == None:
            check_is_fitted(self)
        return self.eval_expr(X_test, sexpr)
