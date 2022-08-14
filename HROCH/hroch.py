from sklearn.base import BaseEstimator, RegressorMixin

import os
import platform
from tempfile import TemporaryDirectory
import subprocess
import numpy as np
import pandas as pd


class Hroch(BaseEstimator, RegressorMixin):

    def __init__(self):
        self.numThreads = 8
        self.timeLimit = 5*1000
        self.stopingCriteria = 0
        self.verbose = False
        self.precision = "f32"
        self.problem = "math"
        self.saveModel = False
        self.dir = TemporaryDirectory()

    def fit(self, X, y):

        x_cnt = np.shape(X)[1]

        if isinstance(X, pd.DataFrame):
            X_train = X
        elif isinstance(X, np.ndarray):
            column_names = ['x_'+str(i) for i in range(x_cnt)]
            X_train = pd.DataFrame(data=X, columns=column_names)
        else:
            raise Exception(
                'param X: wrong type (numpy.ndarray and pandas.Dataframe supported)')

        if isinstance(y, pd.DataFrame):
            y_train = y
        elif isinstance(y, np.ndarray):
            y_train = pd.DataFrame(data=y, columns=["target"])
        else:
            raise Exception(
                'param y: wrong type (numpy.ndarray and pandas.Dataframe supported)')

        # with TemporaryDirectory() as temp_dir:
        fname = self.dir.name + "/tmpdata.csv"
        fmname = self.dir.name + "/model.hrx"

        data_merge = X_train.join(y_train)
        # randomize rows order
        data_merge = data_merge.sample(frac=1).reset_index(drop=True)
        data_merge.to_csv(fname, sep=" ", index=False)

        cwd = os.path.dirname(os.path.realpath(__file__))

        cli = "./hroch"
        if platform.system() == "Windows":
            cli = "hroch.exe"

        process = subprocess.Popen([cli,
                                    "--problem", "math",
                                    "--task", "fit",
                                    "--inputFile", fname,
                                    "--precision", f"{self.precision}",
                                    "--modelFile" if self.saveModel else "--programFile", fmname,
                                    "--timeLimit", f"{self.timeLimit}",
                                    "--numThreads", f"{self.numThreads}",
                                    "--stopingCriteria", f"{self.stopingCriteria}"
                                    ],
                                   cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True)
        for line in iter(process.stdout.readline, b''):
            line = line.decode("utf-8")
            if line.find("eq= ") >= 0:
                self.sexpr = line.removeprefix("eq=")
            if self.verbose:
                print(line, end="")

        process.stdout.close()
        process.wait()

    def predict(self, X_test):
        fname = self.dir.name + "/tmpdata_test.csv"
        foname = self.dir.name + "/tmpdata_out.csv"
        fmname = self.dir.name + "/model.hrx"

        X_test.insert(len(X_test.columns), 'target', 0)
        X_test.to_csv(fname, sep=" ", index=False)

        cwd = os.path.dirname(os.path.realpath(__file__))

        cli = "./hroch"
        if platform.system() == "Windows":
            cli = "hroch.exe"

        process = subprocess.Popen([cli,
                                    "--problem", "math",
                                    "--task", "predict",
                                    "--inputFile", fname,
                                    "--programFile", fmname,
                                    "--outFile", foname,
                                    ],
                                   cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True)
        for line in iter(process.stdout.readline, b''):
            line = line.decode("utf-8")
            if self.verbose:
                print(line, end="")

        process.stdout.close()
        process.wait()
        y = np.genfromtxt(foname, delimiter=' ')
        return y
