
import os
import platform
from tempfile import TemporaryDirectory
import subprocess
import numpy as np


class Hroch:
    """Hroch symbolic regressor"""

    def __init__(self, numThreads: int = 8, timeLimit: float = 5.0, stopingCriteria: float = 0.0, precision: str = "f32", problem: str = "math", saveModel: bool = False, verbose: bool = False):
        self.numThreads = numThreads
        self.timeLimit = round(timeLimit*1000.0)
        self.stopingCriteria = stopingCriteria
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

        cwd = os.path.dirname(os.path.realpath(__file__))

        cli = "hroch.bin"
        if platform.system() == "Windows":
            cli = "hroch.exe"

        process = subprocess.Popen([cli,
                                    "--problem", self.problem,
                                    "--task", "fit",
                                    "--x", fnameX,
                                    "--y", fnameY,
                                    "--precision", f"{self.precision}",
                                    "--modelFile" if self.saveModel else "--programFile", fnameM if self.saveModel else fnameP,
                                    "--timeLimit", f"{self.timeLimit}",
                                    "--numThreads", f"{self.numThreads}",
                                    "--stopingCriteria", f"{self.stopingCriteria}"
                                    ],
                                   cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True)
        for line in iter(process.stdout.readline, b''):
            line = line.decode("utf-8")
            if line.find("eq= ") >= 0:
                self.sexpr = line.removeprefix("eq=").removesuffix("\n")
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

        cwd = os.path.dirname(os.path.realpath(__file__))

        cli = "./hroch.bin"
        if platform.system() == "Windows":
            cli = "hroch.exe"

        process = subprocess.Popen([cli,
                                    "--task", "predict",
                                    "--x", fnameX,
                                    "--y", fnameY,
                                    "--modelFile" if self.saveModel else "--programFile", fnameM if self.saveModel else fnameP,
                                    ],
                                   cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True)
        for line in iter(process.stdout.readline, b''):
            line = line.decode("utf-8")
            if self.verbose:
                print(line, end="")

        process.stdout.close()
        process.wait()
        y = np.genfromtxt(fnameY, delimiter=' ')
        return y
