
import os
import platform
from tempfile import TemporaryDirectory
import subprocess
import numpy as np


class PHCRegressor:
    """Parallel Hill Climbing symbolic regressor"""

    def __init__(self, numThreads: int = 8, timeLimit: float = 5.0, stoppingCriteria: float = 0.0, precision: str = "f32", problem: str = "math", saveModel: bool = False, verbose: bool = False):
        self.numThreads = numThreads
        self.timeLimit = timeLimit
        self.stoppingCriteria = stoppingCriteria
        self.verbose = verbose
        self.precision = precision
        self.problem = problem
        self.saveModel = saveModel

        self.dir = TemporaryDirectory()

    def fit(self, X: np.ndarray, y: np.ndarray):
        fnameX = self.dir.name + "/x.csv"
        fnameY = self.dir.name + "/y.csv"
        fnameM = self.dir.name + "/model.hrx"
        fnameP = self.dir.name + "/program.hrx"

        if os.path.exists(fnameX):
            os.remove(fnameX)
        if os.path.exists(fnameY):
            os.remove(fnameY)

        np.savetxt(fnameX, X, delimiter=" ")
        np.savetxt(fnameY, y, delimiter=" ")

        path = os.path.dirname(os.path.realpath(__file__))

        cli = path + "/hroch.bin"
        if platform.system() == "Windows":
            cli = path + "/hroch.exe"

        process = subprocess.Popen([cli,
                                    "--problem", self.problem,
                                    "--task", "fit",
                                    "--x", fnameX,
                                    "--y", fnameY,
                                    "--precision", f"{self.precision}",
                                    "--modelFile" if self.saveModel else "--programFile", fnameM if self.saveModel else fnameP,
                                    "--timeLimit", f"{int(round(self.timeLimit*1000.0))}",
                                    "--numThreads", f"{self.numThreads}",
                                    "--stoppingCriteria", f"{self.stoppingCriteria}"
                                    ],
                                   cwd=self.dir.name, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True)
        eq_prefix = 'eq= '
        for line in iter(process.stdout.readline, b''):
            line = line.decode("utf-8")
            if line.startswith(eq_prefix):
                self.sexpr = line[len(eq_prefix):]
            if self.verbose:
                print(line, end="")

        process.stdout.close()
        process.wait()

    def predict(self, X_test: np.ndarray):
        fnameX = self.dir.name + "/x.csv"
        fnameY = self.dir.name + "/y.csv"
        fnameM = self.dir.name + "/model.hrx"
        fnameP = self.dir.name + "/program.hrx"

        if os.path.exists(fnameX):
            os.remove(fnameX)
        if os.path.exists(fnameY):
            os.remove(fnameY)

        np.savetxt(fnameX, X_test, delimiter=" ")

        path = os.path.dirname(os.path.realpath(__file__))

        cli = path + "/hroch.bin"
        if platform.system() == "Windows":
            cli = path + "/hroch.exe"

        process = subprocess.Popen([cli,
                                    "--task", "predict",
                                    "--x", fnameX,
                                    "--y", fnameY,
                                    "--modelFile" if self.saveModel else "--programFile", fnameM if self.saveModel else fnameP,
                                    ],
                                   cwd=self.dir.name, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True)
        for line in iter(process.stdout.readline, b''):
            line = line.decode("utf-8")
            if self.verbose:
                print(line, end="")

        process.stdout.close()
        process.wait()
        y = np.genfromtxt(fnameY, delimiter=' ')
        return y

    def get_params(self):
        return {'numThreads': self.numThreads,
                'timeLimit': self.timeLimit,
                'stoppingCriteria': self.stoppingCriteria,
                'precision': self.precision,
                'problem': self.problem,
                'saveModel': self.saveModel}
